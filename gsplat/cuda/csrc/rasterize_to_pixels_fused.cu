#include "bindings.h"
#include "helpers.cuh"
#include "types.cuh"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace gsplat {

namespace cg = cooperative_groups;

template <uint32_t COLOR_DIM, typename S>
__inline__ __device__ void l1_loss_grad(
    const S *__restrict__ input,
    const S *__restrict__ reference,
    S scale,
    S *__restrict__ grad
) {
    GSPLAT_PRAGMA_UNROLL
    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
        S diff = input[k] - reference[k];

        char sign = diff < 0 ? -1 : 0;
        sign = diff > 0 ? 1 : sign;

        grad[k] = sign * scale;
    }
}

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/

template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_to_pixels_fused_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2<S> *__restrict__ means2d, // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,  // [C, N, 3] or [nnz, 3]
    const S *__restrict__ colors,      // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    const S *__restrict__ opacities,   // [C, N] or [nnz]
    const S *__restrict__ backgrounds, // [C, COLOR_DIM]
    const bool *__restrict__ masks,    // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    // fwd outputs
    S *__restrict__ render_colors_out, // [C, image_height, image_width, COLOR_DIM]
    S *__restrict__ render_alphas_out, // [C, image_height, image_width, 1]
    int32_t *__restrict__ last_ids_out, // [C, image_height, image_width]
    // reference input
    const S *__restrict__ ref_colors, // [C, image_height, image_width, D]
    // grad outputs
    const S *__restrict__ v_render_colors, // [C, image_height, image_width,
                                          // COLOR_DIM]
    const S *__restrict__ v_render_alphas, // [C, image_height, image_width, 1]
    // grad inputs
    vec2<S> *__restrict__ v_means2d_abs, // [C, N, 2] or [nnz, 2]
    vec2<S> *__restrict__ v_means2d,     // [C, N, 2] or [nnz, 2]
    vec3<S> *__restrict__ v_conics,      // [C, N, 3] or [nnz, 3]
    S *__restrict__ v_colors,   // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ v_opacities // [C, N] or [nnz]
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    if (render_colors_out != nullptr) {
        render_colors_out += camera_id * image_height * image_width * COLOR_DIM;
    }
    if (render_alphas_out != nullptr) {
        render_alphas_out += camera_id * image_height * image_width;
    }
    if (last_ids_out != nullptr) {
        last_ids_out += camera_id * image_height * image_width;
    }

    ref_colors += camera_id * image_height * image_width * COLOR_DIM;

    if (v_render_colors != nullptr) {
        v_render_colors += camera_id * image_height * image_width * COLOR_DIM;
    }
    if (v_render_alphas != nullptr) {
        v_render_alphas += camera_id * image_height * image_width;
    }
    if (backgrounds != nullptr) {
        backgrounds += camera_id * COLOR_DIM;
    }
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    S px = (S)j + 0.5f;
    S py = (S)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (render_colors_out != nullptr &&
        masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_colors_out[pix_id * COLOR_DIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (camera_id == C - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    // TODO[tinyml] restrict
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]
    vec3<S> *xy_opacity_batch =
        reinterpret_cast<vec3<float> *>(&id_batch[block_size]); // [block_size]
    vec3<S> *conic_batch =
        reinterpret_cast<vec3<float> *>(&xy_opacity_batch[block_size]
        ); // [block_size]
    S *__restrict__ colors_batch =
        reinterpret_cast<S *>(&conic_batch[block_size]
        ); // [block_size * COLOR_DIM]

    // Fwd output buffers for the current pixel
    S render_color[COLOR_DIM], render_alpha;
    int32_t last_id;

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    S T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    S pix_out[COLOR_DIM] = {0.f};
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                colors_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            const vec3<S> conic = conic_batch[t];
            const vec3<S> xy_opac = xy_opacity_batch[t];
            const S opac = xy_opac.z;
            const vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
            const S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                    conic.z * delta.y * delta.y) +
                            conic.y * delta.x * delta.y;
            S alpha = min(0.999f, opac * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f / 255.f) {
                continue;
            }

            const S next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            const S vis = alpha * T;
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                pix_out[k] += colors_batch[t * COLOR_DIM + k] * vis;
            }
            cur_idx = batch_start + t;

            T = next_T;
        }
    }

    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alpha = 1.0f - T;
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            render_color[k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
        // index in bin of last gaussian in this pixel
        last_id = static_cast<int32_t>(cur_idx);

        // Write back if requested
        if (render_alphas_out != nullptr) {
            render_alphas_out[pix_id] = render_alpha;
        }
        if (render_colors_out != nullptr) {
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                render_colors_out[pix_id * COLOR_DIM + k] =
                    render_color[k];
            }
        }
        if (last_ids_out != nullptr) {
            last_ids_out[pix_id] = last_id;
        }
    }

    // *************************************************************************
    // * Rasterization to Pixels Backward Pass
    // *************************************************************************
    // this is the T AFTER the last gaussian in this pixel
    S T_final = 1.0f - render_alpha;
    T = T_final;
    // the contribution from gaussians behind the current one
    S buffer[COLOR_DIM] = {0.f};
    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_id : 0;

    // df/d_out for this pixel
    S v_render_c[COLOR_DIM];
    S v_render_a;

    // Compute gradient (assuming L1 averaged over all channels of all pixels)
    S factor = 1.0f / (float) (image_height * image_width * C * COLOR_DIM);
    l1_loss_grad<COLOR_DIM, S>(
        render_color,
        &ref_colors[pix_id * COLOR_DIM],
        factor,
        v_render_c
    );
    v_render_a = 0;

    // Override gradient if provided
    if (v_render_colors != nullptr) {
        GSPLAT_PRAGMA_UNROLL
        for (uint32_t k = 0; k < COLOR_DIM; ++k) {
            v_render_c[k] = v_render_colors[pix_id * COLOR_DIM + k];
        }
    }
    if (v_render_alphas != nullptr) {
        v_render_a = v_render_alphas[pix_id];
    }

    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);
        const int32_t idx = batch_end - tr;
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [C * N] or [nnz]
            id_batch[tr] = g;
            const vec2<S> xy = means2d[g];
            const S opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};
            conic_batch[tr] = conics[g];
            GSPLAT_PRAGMA_UNROLL
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                colors_batch[tr * COLOR_DIM + k] = colors[g * COLOR_DIM + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // process gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {
            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }
            S alpha;
            S opac;
            vec2<S> delta;
            vec3<S> conic;
            S vis;

            if (valid) {
                conic = conic_batch[t];
                vec3<S> xy_opac = xy_opacity_batch[t];
                opac = xy_opac.z;
                delta = {xy_opac.x - px, xy_opac.y - py};
                S sigma = 0.5f * (conic.x * delta.x * delta.x +
                                  conic.z * delta.y * delta.y) +
                          conic.y * delta.x * delta.y;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis);
                if (sigma < 0.f || alpha < 1.f / 255.f) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }
            S v_rgb_local[COLOR_DIM] = {0.f};
            vec3<S> v_conic_local = {0.f, 0.f, 0.f};
            vec2<S> v_xy_local = {0.f, 0.f};
            vec2<S> v_xy_abs_local = {0.f, 0.f};
            S v_opacity_local = 0.f;
            // initialize everything to 0, only set if the lane is valid
            if (valid) {
                // compute the current T for this gaussian
                S ra = 1.0f / (1.0f - alpha);
                T *= ra;
                // update v_rgb for this gaussian
                const S fac = alpha * T;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_rgb_local[k] = fac * v_render_c[k];
                }
                // contribution from this pixel
                S v_alpha = 0.f;
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    v_alpha +=
                        (colors_batch[t * COLOR_DIM + k] * T - buffer[k] * ra) *
                        v_render_c[k];
                }

                v_alpha += T_final * ra * v_render_a;
                // contribution from background pixel
                if (backgrounds != nullptr) {
                    S accum = 0.f;
                    GSPLAT_PRAGMA_UNROLL
                    for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                if (opac * vis <= 0.999f) {
                    const S v_sigma = -opac * vis * v_alpha;
                    v_conic_local = {
                        0.5f * v_sigma * delta.x * delta.x,
                        v_sigma * delta.x * delta.y,
                        0.5f * v_sigma * delta.y * delta.y
                    };
                    v_xy_local = {
                        v_sigma * (conic.x * delta.x + conic.y * delta.y),
                        v_sigma * (conic.y * delta.x + conic.z * delta.y)
                    };
                    if (v_means2d_abs != nullptr) {
                        v_xy_abs_local = {abs(v_xy_local.x), abs(v_xy_local.y)};
                    }
                    v_opacity_local = vis * v_alpha;
                }

                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    buffer[k] += colors_batch[t * COLOR_DIM + k] * fac;
                }
            }
            warpSum<COLOR_DIM, S>(v_rgb_local, warp);
            warpSum<decltype(warp), S>(v_conic_local, warp);
            warpSum<decltype(warp), S>(v_xy_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum<decltype(warp), S>(v_xy_abs_local, warp);
            }
            warpSum<decltype(warp), S>(v_opacity_local, warp);
            if (warp.thread_rank() == 0) {
                int32_t g = id_batch[t]; // flatten index in [C * N] or [nnz]
                S *v_rgb_ptr = (S *)(v_colors) + COLOR_DIM * g;
                GSPLAT_PRAGMA_UNROLL
                for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                S *v_conic_ptr = (S *)(v_conics) + 3 * g;
                gpuAtomicAdd(v_conic_ptr, v_conic_local.x);
                gpuAtomicAdd(v_conic_ptr + 1, v_conic_local.y);
                gpuAtomicAdd(v_conic_ptr + 2, v_conic_local.z);

                S *v_xy_ptr = (S *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    S *v_xy_abs_ptr = (S *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }
        }
    }
}


template <uint32_t CDIM>
std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // forward outputs
    bool ret_render_colors, // [C, image_height, image_width, D]
    bool ret_render_alphas, // [C, image_height, image_width, 1]
    bool ret_last_ids,      // [C, image_height, image_width]
    const torch::Tensor &ref_colors,
    // gradients of outputs
    const at::optional<torch::Tensor> &v_render_colors, // [C, H, W, D]
    const at::optional<torch::Tensor> &v_render_alphas, // [C, H, W, 1]
    // options
    bool absgrad
) {

    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(colors);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    if (backgrounds.has_value()) {
        GSPLAT_CHECK_INPUT(backgrounds.value());
    }
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }
    if (v_render_colors.has_value()) {
        GSPLAT_CHECK_INPUT(v_render_colors->values());
    }
    if (v_render_alphas.has_value()) {
        GSPLAT_CHECK_INPUT(v_render_alphas.value());
    }

    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t n_isects = flatten_ids.size(0);
    uint32_t COLOR_DIM = colors.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor render_colors, render_alphas, last_ids;

    if (ret_render_colors) {
        render_colors = torch::empty(
            {C, image_height, image_width, COLOR_DIM},
            means2d.options().dtype(torch::kFloat32)
        );
    }
    if (ret_render_alphas) {
        render_alphas = torch::empty(
            {C, image_height, image_width, 1},
            means2d.options().dtype(torch::kFloat32)
        );
    }
    if (ret_last_ids) {
        last_ids = torch::empty(
            {C, image_height, image_width},
            means2d.options().dtype(torch::kInt32)
        );
    }

    torch::Tensor v_means2d = torch::zeros_like(means2d);
    torch::Tensor v_conics = torch::zeros_like(conics);
    torch::Tensor v_colors = torch::zeros_like(colors);
    torch::Tensor v_opacities = torch::zeros_like(opacities);
    torch::Tensor v_means2d_abs;
    if (absgrad) {
        v_means2d_abs = torch::zeros_like(means2d);
    }

    if (n_isects) {
        // int32_t id_batch, vec3<S> xy_opacity_batch, vec3<S> conic_batch
        // S colors_batch[COLOR_DIM],
        const uint32_t shared_mem =
            tile_size * tile_size *
            (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>) +
             sizeof(float) * COLOR_DIM);
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();

        if (cudaFuncSetAttribute(
                rasterize_to_pixels_fused_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shared_mem
            ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shared_mem,
                " bytes), try lowering tile_size."
            );
        }
        rasterize_to_pixels_fused_kernel<CDIM, float>
            <<<blocks, threads, shared_mem, stream>>>(
                C,
                N,
                n_isects,
                packed,
                reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
                colors.data_ptr<float>(),
                opacities.data_ptr<float>(),
                backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                        : nullptr,
                masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                // fwd outputs
                ret_render_colors ? render_colors.data_ptr<float>() : nullptr,
                ret_render_alphas ? render_alphas.data_ptr<float>() : nullptr,
                ret_last_ids ? last_ids.data_ptr<int32_t>() : nullptr,
                // reference
                ref_colors.data_ptr<float>(),
                // grad outputs
                v_render_colors.has_value()
                    ? v_render_colors.value().data_ptr<float>()
                    : nullptr,
                v_render_alphas.has_value()
                    ? v_render_alphas.value().data_ptr<float>()
                    : nullptr,
                absgrad ? reinterpret_cast<vec2<float> *>(
                              v_means2d_abs.data_ptr<float>()
                          )
                        : nullptr,
                reinterpret_cast<vec2<float> *>(v_means2d.data_ptr<float>()),
                reinterpret_cast<vec3<float> *>(v_conics.data_ptr<float>()),
                v_colors.data_ptr<float>(),
                v_opacities.data_ptr<float>()
            );
    }

    return std::make_tuple(
        render_colors, render_alphas, last_ids,
        // TODO
        torch::zeros({}, means2d.options()),
        v_means2d_abs, v_means2d, v_conics, v_colors, v_opacities
    );
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
rasterize_to_pixels_fused_tensor(
    // Gaussian parameters
    const torch::Tensor &means2d,                   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &colors,                    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &opacities,                 // [C, N] or [nnz]
    const at::optional<torch::Tensor> &backgrounds, // [C, 3]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids,  // [n_isects]
    // whether to return forward outputs
    bool ret_render_colors, // [C, image_height, image_width, D]
    bool ret_render_alphas, // [C, image_height, image_width, 1]
    bool ret_last_ids,      // [C, image_height, image_width]
    // reference
    const torch::Tensor &ref_colors,  // [C, image_height, image_width, D]
    // gradients of outputs
    const at::optional<torch::Tensor> &v_render_colors, // [C, H, W, D]
    const at::optional<torch::Tensor> &v_render_alphas, // [C, H, W, 1]
    // options
    bool absgrad
) {

    GSPLAT_CHECK_INPUT(colors);
    uint32_t COLOR_DIM = colors.size(-1);

#define __GS__CALL_(N)                                                         \
    case N:                                                                    \
        return call_kernel_with_dim<N>(                                        \
            means2d,                                                           \
            conics,                                                            \
            colors,                                                            \
            opacities,                                                         \
            backgrounds,                                                       \
            masks,                                                             \
            image_width,                                                       \
            image_height,                                                      \
            tile_size,                                                         \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            ret_render_colors,                                                 \
            ret_render_alphas,                                                 \
            ret_last_ids,                                                      \
            ref_colors,                                                        \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            absgrad                                                            \
        );

    switch (COLOR_DIM) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
    default:
        AT_ERROR("Unsupported number of channels: ", COLOR_DIM);
    }
}

} // namespace gsplat