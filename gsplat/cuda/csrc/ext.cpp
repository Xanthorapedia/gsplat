#include "bindings.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // auto diff functions
    m.def("nd_rasterize_forward", py::overload_cast<
        const std::tuple<int, int, int>,
        const std::tuple<int, int, int>,
        const std::tuple<int, int, int>,
        const float,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &
    >(&nd_rasterize_forward_tensor));
    m.def("nd_rasterize_forward", py::overload_cast<
        const std::tuple<int, int, int>,
        const std::tuple<int, int, int>,
        const std::tuple<int, int, int>,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &
    >(&nd_rasterize_forward_tensor));
    m.def("nd_rasterize_backward", &nd_rasterize_backward_tensor);
    // rasterize with visibility output
    m.def("rasterize_forward", py::overload_cast<
        const std::tuple<int, int, int>,
        const std::tuple<int, int, int>,
        const std::tuple<int, int, int>,
        const float,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &
    >(&rasterize_forward_tensor));
    m.def("rasterize_forward", py::overload_cast<
        const std::tuple<int, int, int>,
        const std::tuple<int, int, int>,
        const std::tuple<int, int, int>,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &,
        const torch::Tensor &
    >(&rasterize_forward_tensor));
    m.def("rasterize_backward", &rasterize_backward_tensor);
    m.def("project_gaussians_forward", &project_gaussians_forward_tensor);
    m.def("project_gaussians_backward", &project_gaussians_backward_tensor);
    m.def("compute_sh_forward", &compute_sh_forward_tensor);
    m.def("compute_sh_backward", &compute_sh_backward_tensor);
    // utils
    m.def("compute_cov2d_bounds", &compute_cov2d_bounds_tensor);
    m.def("map_gaussian_to_intersects", &map_gaussian_to_intersects_tensor);
    m.def("get_tile_bin_edges", &get_tile_bin_edges_tensor);
}
