"""Feature-augmented 3D Gaussian field.

This is the heart of the "add, don't replace" idea. We take an existing trained
splat (positions, quats, scales, opacities, SH colors) and attach ONE new
per-Gaussian attribute: a low-dim language latent. That latent is rendered with
the *same* differentiable rasterizer as color -- gsplat happily alpha-blends any
per-Gaussian channel you hand it as `colors`.

We keep geometry frozen and optimize only the latent. This is the common
two-stage setup: reconstruct RGB first (you already did this), then distill
features on top without disturbing the geometry.

Assumes gsplat >= 1.0 (`from gsplat import rasterization`). The rasterization
call renders arbitrary-dim colors when you pass `sh_degree=None` and
`colors` of shape [N, D].
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RenderResult:
    image: torch.Tensor        # [H, W, 3] RGB (from frozen SH)
    features: torch.Tensor     # [H, W, latent_dim] rendered latent
    alpha: torch.Tensor        # [H, W, 1]


class FeatureField(nn.Module):
    """Wraps a frozen splat and adds a trainable per-Gaussian latent."""

    def __init__(
        self,
        means: torch.Tensor,       # [N, 3]
        quats: torch.Tensor,       # [N, 4]
        scales: torch.Tensor,      # [N, 3]
        opacities: torch.Tensor,   # [N]
        sh_colors: torch.Tensor,   # [N, K, 3] spherical harmonics
        latent_dim: int = 3,
        sh_degree: int = 3,
    ):
        super().__init__()
        # Geometry + appearance are frozen buffers, not parameters.
        self.register_buffer("means", means)
        self.register_buffer("quats", quats)
        self.register_buffer("scales", scales)
        self.register_buffer("opacities", opacities)
        self.register_buffer("sh_colors", sh_colors)
        self.sh_degree = sh_degree

        # The one new trainable thing.
        n = means.shape[0]
        self.latents = nn.Parameter(torch.zeros(n, latent_dim))
        nn.init.normal_(self.latents, std=0.01)

    @property
    def num_gaussians(self) -> int:
        return self.means.shape[0]

    def render(
        self,
        viewmat: torch.Tensor,     # [4, 4] world->camera
        K: torch.Tensor,           # [3, 3] intrinsics
        width: int,
        height: int,
        render_rgb: bool = False,
    ) -> RenderResult:
        from gsplat import rasterization

        viewmats = viewmat[None]   # [1, 4, 4]
        Ks = K[None]               # [1, 3, 3]

        # --- render the latent feature channel (the new path) ------------
        feat_img, alpha, _ = rasterization(
            means=self.means,
            quats=self.quats,
            scales=self.scales,
            opacities=self.opacities,
            colors=self.latents,          # arbitrary-dim "color" = our latent
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=None,               # colors are raw, not SH
            render_mode="RGB",
        )

        rgb = None
        if render_rgb:
            rgb, _, _ = rasterization(
                means=self.means,
                quats=self.quats,
                scales=self.scales,
                opacities=self.opacities,
                colors=self.sh_colors,
                viewmats=viewmats,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=self.sh_degree,
                render_mode="RGB",
            )
            rgb = rgb[0]

        return RenderResult(
            image=rgb,
            features=feat_img[0],         # [H, W, latent_dim]
            alpha=alpha[0],
        )

    # --- construction from a saved splat ---------------------------------

    @classmethod
    def from_ply(cls, path: str, latent_dim: int = 3, sh_degree: int = 3, device="cuda"):
        """Load a splat .ply (INRIA/gsplat convention) into a FeatureField.

        Kept deliberately small; if you train with nerfstudio/gsplat you likely
        already have a loader -- reuse it and just pass the tensors to __init__.
        """
        from vgrep3d.field.ply_io import load_splat_ply

        g = load_splat_ply(path, device=device)
        return cls(
            means=g["means"],
            quats=g["quats"],
            scales=g["scales"],
            opacities=g["opacities"],
            sh_colors=g["sh_colors"],
            latent_dim=latent_dim,
            sh_degree=sh_degree,
        )
