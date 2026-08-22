"""Open-vocabulary 3D query against a trained feature field. (hardened)

Same public interface as before (encode_text / relevance / locate_3d /
render_heatmap). Robustness changes:

  * encode_text(): get_text_features() can return a ModelOutput object
    (e.g. BaseModelOutputWithPooling) instead of a plain tensor in newer
    transformers releases -- same issue as get_image_features() in
    preprocess/features.py. _extract_text_features() below normalizes it.

  * relevance(): per-query normalization uses 1st/99th percentiles instead of
    raw min/max, so a single outlier Gaussian can't rescale the whole scene.

  * locate_3d(): drops spatially scattered hits with an iterative MAD filter
    before fitting the box, and fits the AABB from percentile-clipped
    extents, so a handful of stray high-relevance Gaussians can't inflate it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vgrep3d.field.autoencoder import FeatureAutoencoder
from vgrep3d.field.feature_field import FeatureField


def _extract_text_features(raw) -> torch.Tensor:
    """Normalize get_text_features() output to a plain [N, D] tensor.

    Handles both a bare tensor return and a ModelOutput-style object."""
    if torch.is_tensor(raw):
        return raw
    for attr in ("text_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(raw, attr, None)
        if val is not None:
            if val.dim() == 3:
                val = val[:, 0, :]
            return val
    if isinstance(raw, (tuple, list)) and len(raw) > 0 and torch.is_tensor(raw[0]):
        return raw[0]
    raise TypeError(f"Unrecognized get_text_features() output type: {type(raw)}")


def _keep_dense_core(pts: torch.Tensor, n_mad: float = 3.0, iters: int = 2) -> torch.Tensor:
    """Return a boolean mask over `pts` (M,3) keeping the spatially coherent core."""
    keep = torch.ones(len(pts), dtype=torch.bool, device=pts.device)
    for _ in range(iters):
        p = pts[keep]
        if len(p) < 8:
            break
        med = p.median(dim=0).values
        mad = (p - med).abs().median(dim=0).values + 1e-9
        z = (pts - med).abs() / (1.4826 * mad)
        new_keep = (z < n_mad).all(dim=1) & keep
        if new_keep.sum() < 8 or new_keep.sum() == keep.sum():
            keep = new_keep if new_keep.sum() >= 8 else keep
            break
        keep = new_keep
    return keep


def _robust_aabb(pts: torch.Tensor, lo: float = 2.0, hi: float = 98.0):
    q = torch.tensor([lo / 100.0, hi / 100.0], device=pts.device, dtype=pts.dtype)
    mn = torch.quantile(pts, q[0], dim=0)
    mx = torch.quantile(pts, q[1], dim=0)
    return mn, mx


class Query3D:
    def __init__(
        self,
        field: FeatureField,
        autoencoder: FeatureAutoencoder,
        siglip_model: str = "google/siglip2-so400m-patch14-384",
        device: str = "cuda",
    ):
        self.field = field.to(device).eval()
        self.ae = autoencoder.to(device).eval()
        self.device = device
        self._model = None
        self._processor = None
        self._siglip_name = siglip_model

        with torch.no_grad():
            self.gauss_feats = self.ae.decode(self.field.latents.to(device))  # [N, D]

    def _load_text_tower(self):
        if self._model is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._siglip_name)
        self._model = AutoModel.from_pretrained(self._siglip_name).eval().to(self.device)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        self._load_text_tower()
        inputs = self._processor(
            text=[text], return_tensors="pt", padding="max_length", max_length=64
        ).to(self.device)
        raw = self._model.get_text_features(**inputs)
        feat = _extract_text_features(raw)
        return F.normalize(feat, dim=-1)[0]  # [D]

    @torch.no_grad()
    def relevance(self, text: str) -> torch.Tensor:
        """Per-Gaussian relevance in [0, 1], robustly normalized per query."""
        t = self.encode_text(text)                       # [D]
        sim = self.gauss_feats @ t                        # [N] cosine (both normed)
        lo = torch.quantile(sim, 0.01)
        hi = torch.quantile(sim, 0.99)
        rel = ((sim - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
        return rel

    @torch.no_grad()
    def locate_3d(self, text: str, threshold: float = 0.6, min_hits: int = 16) -> dict:
        rel = self.relevance(text)                        # [N]
        hit_idx = torch.where(rel >= threshold)[0]
        if len(hit_idx) < min_hits:                       # threshold too strict -> top slice
            k = max(min_hits, int(0.01 * len(rel)))
            hit_idx = torch.topk(rel, k).indices
        if len(hit_idx) == 0:
            return {"found": False, "relevance": rel}

        pts = self.field.means[hit_idx]                   # [M, 3]
        core = _keep_dense_core(pts)                      # spatial coherence filter
        core_idx = hit_idx[core]
        pts_core = self.field.means[core_idx]
        if len(pts_core) < 8:
            pts_core, core_idx = pts, hit_idx              # fall back to unfiltered

        w = rel[core_idx][:, None]
        centroid = (pts_core * w).sum(0) / w.sum()
        mn, mx = _robust_aabb(pts_core)
        return {
            "found": True,
            "centroid": centroid.cpu(),
            "aabb_min": mn.cpu(),
            "aabb_max": mx.cpu(),
            "num_gaussians": int(len(core_idx)),
            "indices": core_idx.cpu(),
            "relevance": rel,
        }

    @torch.no_grad()
    def render_heatmap(self, text: str, viewmat, K, width, height) -> torch.Tensor:
        """Relevance rendered from a camera -> [H, W] in [0, 1]."""
        rel = self.relevance(text)                        # [N]
        saved = self.field.latents
        try:
            self.field.latents = torch.nn.Parameter(rel[:, None], requires_grad=False)
            out = self.field.render(viewmat, K, width, height)
            heat = out.features[..., 0]
        finally:
            self.field.latents = saved
        return heat
