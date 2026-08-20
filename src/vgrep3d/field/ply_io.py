"""Load a 3DGS .ply (INRIA/gsplat convention) into tensors.

Handles the standard property layout: xyz, opacity, scale_0..2, rot_0..3,
f_dc_0..2 (SH DC term) and f_rest_* (higher-order SH). If your trainer saved a
different format, replace this with its loader -- FeatureField only needs the
five tensors returned here.
"""

from __future__ import annotations

import numpy as np
import torch


def load_splat_ply(path: str, device: str = "cuda") -> dict:
    from plyfile import PlyData

    ply = PlyData.read(path)
    v = ply["vertex"]

    xyz = np.stack([v["x"], v["y"], v["z"]], axis=-1)
    opacity = np.asarray(v["opacity"])
    scales = np.stack([v[f"scale_{i}"] for i in range(3)], axis=-1)
    quats = np.stack([v[f"rot_{i}"] for i in range(4)], axis=-1)

    dc = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=-1)  # [N, 3]
    rest_names = sorted(
        (p.name for p in v.properties if p.name.startswith("f_rest_")),
        key=lambda s: int(s.split("_")[-1]),
    )
    if rest_names:
        rest = np.stack([v[n] for n in rest_names], axis=-1)  # [N, 3*(K-1)]
        rest = rest.reshape(len(xyz), -1, 3)                  # [N, K-1, 3]
        sh = np.concatenate([dc[:, None, :], rest], axis=1)   # [N, K, 3]
    else:
        sh = dc[:, None, :]                                   # [N, 1, 3]

    def t(a, dtype=torch.float32):
        return torch.from_numpy(np.ascontiguousarray(a)).to(dtype).to(device)

    # gsplat expects activated scales/opacities; the .ply stores raw (pre-act).
    return {
        "means": t(xyz),
        "quats": torch.nn.functional.normalize(t(quats), dim=-1),
        "scales": torch.exp(t(scales)),
        "opacities": torch.sigmoid(t(opacity)),
        "sh_colors": t(sh),
    }
