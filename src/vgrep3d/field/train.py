"""Distill 2D SigLIP 2 feature maps into the 3D feature field.

Geometry is frozen; we optimize only the per-Gaussian latents so that, when
rendered from each training view, the blended latent map matches the encoded
SigLIP target for that view.

This version adds three reliability features needed for long-running Modal
jobs that can hit a function timeout mid-training:

  * views_per_epoch: sample a random subset of cameras each epoch instead of
    every view every epoch. Cuts wall time roughly proportionally with no
    change to what's being optimized (still SGD over the same objective).

  * save_every / save_path: write latents.pt periodically during training,
    not just at the end, so a timeout doesn't lose all prior progress.

  * train_feature_field(..., start_epoch=N) combined with load_latents lets a
    fresh invocation resume from the last checkpoint instead of restarting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from vgrep3d.field.autoencoder import FeatureAutoencoder
from vgrep3d.field.feature_field import FeatureField
from vgrep3d.preprocess.features import load_feature_map


def train_feature_field(
    field: FeatureField,
    autoencoder: FeatureAutoencoder,
    cameras: list[dict],
    epochs: int = 30,
    lr: float = 1e-2,
    device: str = "cuda",
    views_per_epoch: int | None = None,
    save_every: int | None = None,
    save_path: str | Path | None = None,
    start_epoch: int = 0,
) -> FeatureField:
    field = field.to(device)
    autoencoder = autoencoder.to(device).eval()
    for p in autoencoder.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam([field.latents], lr=lr)
    n_views = views_per_epoch or len(cameras)

    for epoch in range(start_epoch, epochs):
        pool = np.random.permutation(len(cameras))[:n_views]
        running = 0.0
        for idx in pool:
            cam = cameras[idx]
            target_raw = torch.from_numpy(load_feature_map(cam["feature_path"]))
            valid = target_raw.norm(dim=-1) > 1e-6
            if valid.sum() == 0:
                continue

            with torch.no_grad():
                t = autoencoder.encode(target_raw[valid].to(device))

            out = field.render(
                viewmat=cam["viewmat"].to(device),
                K=cam["K"].to(device),
                width=cam["width"],
                height=cam["height"],
            )
            rendered = out.features
            r = rendered[valid.to(device)]

            loss = (1 - F.cosine_similarity(r, t, dim=-1)).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        print(f"[field] epoch {epoch + 1:3d}/{epochs}  "
              f"loss {running / max(len(pool), 1):.5f}  ({len(pool)} views)")

        if save_every and save_path and (epoch + 1) % save_every == 0:
            save_field(field, save_path)
            print(f"[field] checkpoint saved at epoch {epoch + 1} -> {save_path}")

    if save_path:
        save_field(field, save_path)
    return field


def save_field(field: FeatureField, path: str | Path) -> None:
    torch.save({"latents": field.latents.detach().cpu()}, path)


def load_latents(field: FeatureField, path: str | Path, device="cuda") -> FeatureField:
    state = torch.load(path, map_location=device)
    loaded = state["latents"]
    n_field, n_loaded = field.means.shape[0], loaded.shape[0]
    if n_field != n_loaded:
        raise ValueError(
            f"latents.pt has {n_loaded:,} rows but the loaded checkpoint has "
            f"{n_field:,} Gaussians (field.means). These must match exactly -- "
            f"the field was almost certainly trained with a different "
            f"max_gaussians cap than this call is using. Use the same "
            f"max_gaussians value (or None) everywhere: index, query, and "
            f"export must all agree, or the two tensors silently misalign "
            f"and produce out-of-bounds indexing / CUDA device-side asserts "
            f"downstream instead of this clear error."
        )
    field.latents.data = loaded.to(device)
    return field
