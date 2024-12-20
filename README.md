# Kernel Fusion for Cache-friendly Gaussian Splatting on Mobile Devices
Fall 2024 6.5940 Final Project by Dasong Gao, Jungsoo Lee, Sarah Zhang

Although Gaussian Splatting (GS) allows for high-fidelity reconstruction of 3D scenes, the current GS implementations are not optimized to reduce DRAM accesses during training, making GS difficult to deploy on memory-constrained devices. In this project, we propose to reduce the DRAM traffic by fusing the forward and backward kernels of GS during training. Our method enables intermediate results from forward to be stored and consumed within shared memory rather than written to DRAM.


## Project Demo and Experiment Scripts
The subsequent gsplat section belows explain how to set up the gsplat package. Make sure gsplat is properly installed before running the demo.

Navigate to `./examples/benchmarks/`. Run `basic_garden_original.sh` to train a gaussian splat model on the Mip-NeRF360 Garden scene using combination of L1 and SSIM loss (traditional gaussian splatting). Run `basic_garden_no_ssim.sh` to train gaussian splat model on the same scene using only L1 loss, as our kernel fused implementation uses. Checkpoints at 7k and 30k training steps as well as a video in the scene along a fixed trajectory are saved into respective results directories. This duplicates our results from Table 3 in our paper.

The trained GSs can be viewed interactively by running ```python simple_viewer.py --ckpt <results-dir>/ckpt_<num-steps>_rank0.pt --backend gsplat```. This duplicates our results from Table 3 in our paper.

In general, training and viewing our implementation can be done with the `./examples/simple_trainer.py` script, which has usage instructions at the bottom of the file.

## Profiling Experiments
We use the [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) to profile the 51th iteration from the start of the training.

To profile the unfused (original, forward+backward) implementation:
```sh
cd examples
ncu --config-file off --export "./gsplatting/%i" --force-overwrite --kernel-name regex:^rasterize_to_pixels_ \
--launch-skip 100 --launch-count 2 --kill 1 --set full /path/to/python simple_trainer.py default --eval_steps -1 \
--disable_viewer --data_factor 4 --render_traj_path ellipse --data_dir data/360_v2/garden/ \
--result_dir results/fused/benchmark/garden/
```

Expected Results (profiled on NVIDIA Jetson Orin Nano (8GB)):

- [`rasterize_to_pixels_fwd_kernel`](gsplat/cuda/csrc/rasterize_to_pixels_fwd.cu)
<details>
  <summary>Report</summary>
  <a href="profiling/ncu-reports/orin-fwd.png">
  <img
    src=profiling/ncu-reports/orin-fwd.png
    style="width: 100%; aspect-ratio : 1.8 / 1; object-fit: cover;  object-position: 50% 26.7%;"
  />
  </a>
</details>

- [`rasterize_to_pixels_bwd_kernel`](gsplat/cuda/csrc/rasterize_to_pixels_bwd.cu)
<details>
  <summary>Report</summary>
  <a href="profiling/ncu-reports/orin-bwd.png">
  <img
    src=profiling/ncu-reports/orin-bwd.png
    style="width: 100%; aspect-ratio : 1.9 / 1; object-fit: cover;  object-position: 50% 25.7%;"
  />
  </a>
</details>

To profile the fused (our) implementation:
```sh
cd examples
ncu --config-file off --export "./gsplatting/%i" --force-overwrite --kernel-name regex:^rasterize_to_pixels_ \
--launch-skip 50 --launch-count 2 --kill 1 --set full /path/to/python simple_trainer.py default --eval_steps -1 \
--disable_viewer --data_factor 4 --render_traj_path ellipse --data_dir data/360_v2/garden/ \
--result_dir results/fused/benchmark/garden/ --fused
```

Expected Results (profiled on NVIDIA Jetson Orin Nano (8GB)):
- Expected Results ([`rasterize_to_pixels_fused_kernel`](gsplat/cuda/csrc/rasterize_to_pixels_fused.cu) profiled on NVIDIA Jetson Orin Nano (8GB)):
<details>
  <summary>Report</summary>
  <a href="profiling/ncu-reports/orin-fused.png">
  <img
    src=profiling/ncu-reports/orin-fused.png
    style="width: 100%; aspect-ratio : 1.9 / 1; object-fit: cover;  object-position: 50% 25.4%;"
  />
  </a>
</details>

# gsplat

[![Core Tests.](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/core_tests.yml)
[![Docs](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml/badge.svg?branch=main)](https://github.com/nerfstudio-project/gsplat/actions/workflows/doc.yml)

[http://www.gsplat.studio/](http://www.gsplat.studio/)

gsplat is an open-source library for CUDA accelerated rasterization of gaussians with python bindings. It is inspired by the SIGGRAPH paper [3D Gaussian Splatting for Real-Time Rendering of Radiance Fields](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), but we’ve made gsplat even faster, more memory efficient, and with a growing list of new features! 

<div align="center">
  <video src="https://github.com/nerfstudio-project/gsplat/assets/10151885/64c2e9ca-a9a6-4c7e-8d6f-47eeacd15159" width="100%" />
</div>

## Installation

**Dependence**: Please install [Pytorch](https://pytorch.org/get-started/locally/) first.

The easiest way is to install from PyPI. In this way it will build the CUDA code **on the first run** (JIT).

```bash
pip install gsplat
```

Alternatively you can install gsplat from source. In this way it will build the CUDA code during installation.

```bash
pip install git+https://github.com/nerfstudio-project/gsplat.git
```

We also provide [pre-compiled wheels](https://docs.gsplat.studio/whl) for both linux and windows on certain python-torch-CUDA combinations (please check first which versions are supported). Note this way you would have to manually install [gsplat's dependencies](https://github.com/nerfstudio-project/gsplat/blob/6022cf45a19ee307803aaf1f19d407befad2a033/setup.py#L115). For example, to install gsplat for pytorch 2.0 and cuda 11.8 you can run
```
pip install ninja numpy jaxtyping rich
pip install gsplat --index-url https://docs.gsplat.studio/whl/pt20cu118
```

To build gsplat from source on Windows, please check [this instruction](docs/INSTALL_WIN.md).

## Evaluation

This repo comes with a standalone script that reproduces the official Gaussian Splatting with exactly the same performance on PSNR, SSIM, LPIPS, and converged number of Gaussians. Powered by gsplat’s efficient CUDA implementation, the training takes up to **4x less GPU memory** with up to **15% less time** to finish than the official implementation. Full report can be found [here](https://docs.gsplat.studio/main/tests/eval.html).

```bash
pip install -r examples/requirements.txt
# download mipnerf_360 benchmark data
python examples/datasets/download_dataset.py
# run batch evaluation
bash examples/benchmarks/basic.sh
```

## Examples

We provide a set of examples to get you started! Below you can find the details about
the examples (requires to install some exta dependencies via `pip install -r examples/requirements.txt`)

- [Train a 3D Gaussian splatting model on a COLMAP capture.](https://docs.gsplat.studio/main/examples/colmap.html)
- [Fit a 2D image with 3D Gaussians.](https://docs.gsplat.studio/main/examples/image.html)
- [Render a large scene in real-time.](https://docs.gsplat.studio/main/examples/large_scale.html)


## Development and Contribution

This repository was born from the curiosity of people on the Nerfstudio team trying to understand a new rendering technique. We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software.

This project is developed by the following wonderful contributors (unordered):

- [Angjoo Kanazawa](https://people.eecs.berkeley.edu/~kanazawa/) (UC Berkeley): Mentor of the project.
- [Matthew Tancik](https://www.matthewtancik.com/about-me) (Luma AI): Mentor of the project.
- [Vickie Ye](https://people.eecs.berkeley.edu/~vye/) (UC Berkeley): Project lead. v0.1 lead.
- [Matias Turkulainen](https://maturk.github.io/) (Aalto University): Core developer.
- [Ruilong Li](https://www.liruilong.cn/) (UC Berkeley): Core developer. v1.0 lead.
- [Justin Kerr](https://kerrj.github.io/) (UC Berkeley): Core developer.
- [Brent Yi](https://github.com/brentyi) (UC Berkeley): Core developer.
- [Zhuoyang Pan](https://panzhy.com/) (ShanghaiTech University): Core developer.
- [Jianbo Ye](http://www.jianboye.org/) (Amazon): Core developer.

We also have a white paper with about the project with benchmarking and mathematical supplement with conventions and derivations, available [here](https://arxiv.org/abs/2409.06765). If you find this library useful in your projects or papers, please consider citing:

```
@article{ye2024gsplatopensourcelibrarygaussian,
    title={gsplat: An Open-Source Library for {Gaussian} Splatting}, 
    author={Vickie Ye and Ruilong Li and Justin Kerr and Matias Turkulainen and Brent Yi and Zhuoyang Pan and Otto Seiskari and Jianbo Ye and Jeffrey Hu and Matthew Tancik and Angjoo Kanazawa},
    year={2024},
    eprint={2409.06765},
    journal={arXiv preprint arXiv:2409.06765},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2409.06765}, 
}
```

We welcome contributions of any kind and are open to feedback, bug-reports, and improvements to help expand the capabilities of this software. Please check [docs/DEV.md](docs/DEV.md) for more info about development.
