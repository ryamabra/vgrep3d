"""Open-vocabulary 3D query against a trained feature field.

Two query modes:

  locate_3d(text)  -> per-Gaussian relevance + the 3D centroid/AABB of the hit.
                      This is the "where is the fire extinguisher" answer: a
                      point/box in the scene's coordinate frame.

  render_heatmap(text, view) -> a 2D relevance heatmap from a given camera, for
                      making the demo video / README gif.

Both work by:
  1. encode text with the SigLIP 2 text tower  -> [D]
  2. decode each Gaussian's latent up to [D]    (autoencoder.decode)
  3. cosine similarity -> relevance in [-1, 1]
  4. threshold (relevance normalized per-query, LERF-style)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from vgrep3d.field.autoencoder import FeatureAutoencoder
from vgrep3d.field.feature_field import FeatureField


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

        # Precompute decoded per-Gaussian features once.
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
        feat = self._model.get_text_features(**inputs)
        return F.normalize(feat, dim=-1)[0]  # [D]

    @torch.no_grad()
    def relevance(self, text: str) -> torch.Tensor:
        """Per-Gaussian relevance in [0, 1], normalized per query."""
        t = self.encode_text(text)                       # [D]
        sim = self.gauss_feats @ t                        # [N] cosine (both normed)
        # per-query min-max normalize so the threshold is scene-agnostic
        sim = (sim - sim.min()) / (sim.max() - sim.min() + 1e-8)
        return sim

    @torch.no_grad()
    def locate_3d(self, text: str, threshold: float = 0.6) -> dict:
        rel = self.relevance(text)                        # [N]
        hit = rel >= threshold
        if hit.sum() == 0:
            return {"found": False, "relevance": rel}
        pts = self.field.means[hit]                       # [M, 3]
        weights = rel[hit][:, None]
        centroid = (pts * weights).sum(0) / weights.sum()
        return {
            "found": True,
            "centroid": centroid.cpu(),
            "aabb_min": pts.min(0).values.cpu(),
            "aabb_max": pts.max(0).values.cpu(),
            "num_gaussians": int(hit.sum()),
            "relevance": rel,
        }

    @torch.no_grad()
    def render_heatmap(self, text: str, viewmat, K, width, height) -> torch.Tensor:
        """Relevance rendered from a camera -> [H, W] in [0, 1]."""
        rel = self.relevance(text)                        # [N]
        # stash relevance as a 1-channel "color" and splat it
        saved = self.field.latents
        try:
            self.field.latents = torch.nn.Parameter(rel[:, None], requires_grad=False)
            out = self.field.render(viewmat, K, width, height)
            heat = out.features[..., 0]
        finally:
            self.field.latents = saved
        return heat
