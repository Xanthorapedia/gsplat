"""Tests for the functions in the CUDA extension.

Usage:
```bash
pytest <THIS_PY_FILE> -s
```
"""

from typing import Optional

import pytest
import torch
import torch.nn.functional as F
device = torch.device("cuda:0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA device")
@pytest.mark.parametrize("per_view_color", [True, False])
@pytest.mark.parametrize("sh_degree", [None, 3])
@pytest.mark.parametrize("render_mode", ["RGB", "RGB+D", "D"])
@pytest.mark.parametrize("packed", [True, False])
def test_fused_rasterization(
    per_view_color: bool, sh_degree: Optional[int], render_mode: str, packed: bool
):
    # TODO[tinyml]: Replace rasterization with fused rasterization
    from gsplat.rendering import _rasterization, rasterization

    torch.manual_seed(42)

    C, N = 2, 10_000
    means = torch.rand(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    scales = torch.rand(N, 3, device=device)
    opacities = torch.rand(N, device=device)
    if per_view_color:
        if sh_degree is None:
            colors = torch.rand(C, N, 3, device=device)
        else:
            colors = torch.rand(C, N, (sh_degree + 1) ** 2, 3, device=device)
    else:
        if sh_degree is None:
            colors = torch.rand(N, 3, device=device)
        else:
            colors = torch.rand(N, (sh_degree + 1) ** 2, 3, device=device)

    width, height = 300, 200
    focal = 300.0
    Ks = torch.tensor(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        device=device,
    ).expand(C, -1, -1)
    viewmats = torch.eye(4, device=device).expand(C, -1, -1)

    renders, alphas, meta = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=packed,
    )

    if render_mode == "D":
        assert renders.shape == (C, height, width, 1)
    elif render_mode == "RGB":
        assert renders.shape == (C, height, width, 3)
    elif render_mode == "RGB+D":
        assert renders.shape == (C, height, width, 4)

    _renders, _alphas, _meta = _rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
    )
    torch.testing.assert_close(renders, _renders, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(alphas, _alphas, rtol=1e-4, atol=1e-4)

    # TODO[tinyml]: Tests for backward (L1 loss only)
    if render_mode == "D":
        dim=1
    elif render_mode == "RGB":
        dim=3 
    elif render_mode == "RGB+D":
        dim=4   
    ref_color= torch.randn(C,height,width,dim)
    l1_loss= F.l1_loss(renders, ref_color) 
    l1_loss.backward()
    means_grad=means.grad
    quats_grad=quats.grad
    scales_grad=scales.grad
    opacities_grad= opacities.grad
    colors_grad=colors.grad
    renders_fused, alphas_fused, meta_fused = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        sh_degree=sh_degree,
        render_mode=render_mode,
        packed=packed,
        ref_color=ref_color
    )
    l1_loss_fused= meta_fused["render_loss"]
    l1_loss_fused.backward()
    means_grad_fused=means.grad
    quats_grad_fused=quats.grad
    scales_grad_fused=scales.grad
    opacities_grad_fused= opacities.grad
    colors_grad_fused=colors.grad
    torch.testing.assert_close(l1_loss, l1_loss_fused, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(means_grad, means_grad_fused, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(quats_grad,quats_grad_fused,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(scales_grad,scales_grad_fused,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(opacities_grad,opacities_grad_fused,rtol=1e-4,atol=1e-4)
    torch.testing.assert_close(colors_grad,colors_grad_fused,rtol=1e-4,atol=1e-4)




    



