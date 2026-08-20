"""Per-scene feature autoencoder.

Storing a raw ~1152-d SigLIP 2 vector on every one of millions of Gaussians is
infeasible on a single A10G. Following LangSplat, we learn a small per-scene
autoencoder that maps the SigLIP feature to a low-dim latent (default 3-16).
We store the *latent* on each Gaussian and decode back up only at query time.

Why per-scene (not a global AE): the compression only has to be lossless enough
for the objects present in *this* scene, so a tiny scene-specific bottleneck
beats a general-purpose one and trains in a couple of minutes.

Usage:
    ae = FeatureAutoencoder(in_dim=1152, latent_dim=3)
    train_autoencoder(ae, feature_maps_dir)   # fits on all mask embeddings
    z = ae.encode(feat)                        # [.., 1152] -> [.., 3]
    f = ae.decode(z)                           # [.., 3]    -> [.., 1152]
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAutoencoder(nn.Module):
    def __init__(self, in_dim: int = 1152, latent_dim: int = 3, hidden: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, hidden),
            nn.GELU(),
            nn.Linear(hidden, in_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        # Decoded features are compared with SigLIP embeddings via cosine sim,
        # so we L2-normalize the output.
        return F.normalize(self.decoder(z), dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def _load_all_embeddings(feature_dir: str | Path) -> torch.Tensor:
    """Collect the unique non-zero embeddings across all feature maps.

    Feature maps are dense [H, W, D] with many pixels sharing a mask's vector,
    so we dedup to keep the AE training set small and balanced.
    """
    feature_dir = Path(feature_dir)
    chunks = []
    for p in sorted(feature_dir.glob("*.npy")):
        fmap = np.load(p).reshape(-1, np.load(p).shape[-1])
        nonzero = fmap[np.linalg.norm(fmap, axis=-1) > 1e-6]
        if len(nonzero) == 0:
            continue
        uniq = np.unique(np.round(nonzero, 4), axis=0)
        chunks.append(torch.from_numpy(uniq).float())
    if not chunks:
        raise RuntimeError(f"No non-zero features found in {feature_dir}")
    return torch.cat(chunks, dim=0)


def train_autoencoder(
    ae: FeatureAutoencoder,
    feature_dir: str | Path,
    epochs: int = 200,
    batch_size: int = 4096,
    lr: float = 1e-3,
    device: str = "cuda",
) -> FeatureAutoencoder:
    data = _load_all_embeddings(feature_dir).to(device)
    ae = ae.to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=lr)
    n = len(data)
    for epoch in range(epochs):
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, batch_size):
            batch = data[perm[i : i + batch_size]]
            recon = ae(batch)
            # cosine loss: we only care about direction (used for cosine-sim query)
            loss = (1 - F.cosine_similarity(recon, batch, dim=-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
        if epoch % 20 == 0 or epoch == epochs - 1:
            print(f"[AE] epoch {epoch:4d}  cosine-loss {total / n:.5f}")
    return ae
