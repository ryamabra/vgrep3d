"""Distill 2D SigLIP 2 feature maps into the 3D feature field.

Geometry is frozen; we optimize only the per-Gaussian latents so that, when
rendered from each training view, the blended latent map matches the encoded
SigLIP target for that view.

Target pipeline per view:
    raw SigLIP map [H, W, 1152]  --AE.encode-->  target latent [H, W, L]
    field.render(view)           --------------> rendered latent [H, W, L]
    loss = 1 - cosine(rendered, target)   (masked to pixels with a target)

Only pixels that got a SAM mask carry a target; everything else is ignored so
empty background does not wash out the signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vgrep3d.field.autoencoder import FeatureAutoencoder
from vgrep3d.field.feature_field import FeatureField


def train_feature_field(
    field: FeatureField,
    autoencoder: FeatureAutoencoder,
    cameras: list[dict],          # each: {viewmat, K, width, height, feature_path}
    epochs: int = 30,
    lr: float = 1e-2,
    device: str = "cuda",
) -> FeatureField:
    field = field.to(device)
    autoencoder = autoencoder.to(device).eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam([field.latents], lr=lr)

    for epoch in range(epochs):
        order = np.random.permutation(len(cameras))
        running = 0.0
        for idx in order:
            cam = cameras[idx]
            target_raw = torch.from_numpy(np.load(cam["feature_path"])).to(device)
            valid = target_raw.norm(dim=-1) > 1e-6           # [H, W]
            if valid.sum() == 0:
                continue

            with torch.no_grad():
                target_latent = autoencoder.encode(target_raw)  # [H, W, L]

            out = field.render(
                viewmat=cam["viewmat"].to(device),
                K=cam["K"].to(device),
                width=cam["width"],
                height=cam["height"],
            )
            rendered = out.features                            # [H, W, L]

            r = rendered[valid]
            t = target_latent[valid]
            loss = (1 - F.cosine_similarity(r, t, dim=-1)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        print(f"[field] epoch {epoch + 1:3d}/{epochs}  loss {running / len(cameras):.5f}")
    return field


def save_field(field: FeatureField, path: str | Path) -> None:
    torch.save({"latents": field.latents.detach().cpu()}, path)


def load_latents(field: FeatureField, path: str | Path, device="cuda") -> FeatureField:
    state = torch.load(path, map_location=device)
    field.latents.data = state["latents"].to(device)
    return field
