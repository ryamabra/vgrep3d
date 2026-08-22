"""Load Gaussians directly from a gsplat simple_trainer checkpoint (.pt).

FIX (critical): when max_gaussians < the checkpoint's true Gaussian count, the
original version subsampled with an UNSEEDED torch.randperm. Every separate
script invocation (index, query, export, visualize -- each a fresh Modal
container) drew a different random subset, so array position i referred to a
different physical Gaussian each time. Trained latents.pt (learned per
position) then got silently misapplied to the wrong Gaussians on every
subsequent load -- this produced confident-looking but spatially meaningless
detections. Fixed by subsampling with a fixed seed, so the same 500K subset
is loaded identically every time, in every process, forever.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

SUBSAMPLE_SEED = 1234  # do not change without retraining -- changing this
                        # changes which Gaussians "index 0..N" refers to


@dataclass
class GaussianModel:
    means: torch.Tensor
    opacities: torch.Tensor
    quats: torch.Tensor
    scales: torch.Tensor
    sh: torch.Tensor
    sh_degree: int

    @property
    def num_points(self) -> int:
        return self.means.shape[0]


def load_gsplat_checkpoint(
    ckpt_path: str | Path,
    device: str | torch.device = "cpu",
    max_gaussians: Optional[int] = None,
) -> GaussianModel:
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not isinstance(ck, dict) or "splats" not in ck:
        raise ValueError(f"Unexpected checkpoint format. Keys: {list(ck.keys()) if isinstance(ck, dict) else type(ck)}")

    splats = ck["splats"]
    required = ["means", "opacities", "quats", "scales", "sh0"]
    missing = [k for k in required if k not in splats]
    if missing:
        raise KeyError(f"Missing keys: {missing}")

    means = splats["means"].float()
    opacities = splats["opacities"].float()
    quats = splats["quats"].float()
    scales = splats["scales"].float()
    sh0 = splats["sh0"].float()
    shN = splats.get("shN", None)

    N = means.shape[0]

    if max_gaussians is not None and N > max_gaussians:
        # FIXED: seeded generator, independent of global RNG state, so this
        # is byte-identical across every process/container invocation.
        gen = torch.Generator().manual_seed(SUBSAMPLE_SEED)
        idx = torch.randperm(N, generator=gen)[:max_gaussians]
        means, opacities, quats, scales, sh0 = means[idx], opacities[idx], quats[idx], scales[idx], sh0[idx]
        if shN is not None:
            shN = shN[idx]
        N = max_gaussians

    # activations
    if opacities.min() < 0 or opacities.max() > 1.0:
        opacities = torch.sigmoid(opacities)
    if scales.min() <= 0:
        scales = torch.exp(scales)
    quats = F.normalize(quats, dim=-1)

    if shN is not None:
        sh = torch.cat([sh0, shN], dim=1)
    else:
        sh = sh0

    K = sh.shape[1]
    sh_degree = int(round(K ** 0.5 - 1))
    if (sh_degree + 1) ** 2 != K:
        sh_degree = 3 if K >= 16 else 0

    device = torch.device(device)
    return GaussianModel(
        means=means.to(device),
        opacities=opacities.to(device),
        quats=quats.to(device),
        scales=scales.to(device),
        sh=sh.to(device),
        sh_degree=sh_degree,
    )


def load_checkpoint(path: str | Path, device="cpu", max_gaussians=None) -> GaussianModel:
    return load_gsplat_checkpoint(path, device=device, max_gaussians=max_gaussians)
