"""Extract 2D SigLIP 2 feature maps from training images.

Pipeline per image:
    1. SAM automatic mask generator -> list of binary masks
    2. For each mask, crop the masked region and run the SigLIP 2 image tower
       -> one embedding per mask (shape [D])
    3. Paint each embedding back onto its mask's pixels
       -> a dense feature map of shape [H, W, D]

These maps are the supervision target for the 3D feature field. We store them
compressed (see autoencoder.py) rather than raw, because a full [H, W, 1152]
float map per frame is large.

Design notes
------------
- We use SAM's *automatic* mask generator so this stays open-vocabulary and
  annotation-free, matching the LangSplat recipe. Swap in SAM 2 if you want
  cross-frame mask tracking later (that is one lever on the multi-view
  consistency problem).
- Overlapping masks: SAM returns overlapping masks at multiple scales. We keep
  the highest-scoring mask per pixel (see `_flatten_masks`). LangSplat instead
  keeps three scale levels separately; single-level is simpler and a fine
  starting point.
- This module needs a GPU and downloaded weights. It is written to run on your
  A10G Modal setup, not in a CPU-only sandbox.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


@dataclasses.dataclass
class FeatureExtractorConfig:
    siglip_model: str = "google/siglip2-so400m-patch14-384"
    sam_checkpoint: str = "sam_vit_h_4b8939.pth"
    sam_model_type: str = "vit_h"
    device: str = "cuda"
    # Minimum mask area (in pixels) to bother embedding. Filters SAM noise.
    min_mask_area: int = 400
    # Batch size for running SigLIP over mask crops.
    crop_batch_size: int = 32


class FeatureExtractor:
    """Turns an RGB image into a dense [H, W, D] SigLIP 2 feature map."""

    def __init__(self, cfg: FeatureExtractorConfig):
        self.cfg = cfg
        self._siglip = None
        self._siglip_processor = None
        self._sam = None
        self.feature_dim: int | None = None

    # --- lazy loaders so importing this module is cheap -------------------

    def _load_siglip(self):
        if self._siglip is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._siglip_processor = AutoProcessor.from_pretrained(self.cfg.siglip_model)
        self._siglip = (
            AutoModel.from_pretrained(self.cfg.siglip_model)
            .eval()
            .to(self.cfg.device)
        )
        self.feature_dim = self._siglip.config.text_config.projection_size

    def _load_sam(self):
        if self._sam is not None:
            return
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[self.cfg.sam_model_type](
            checkpoint=self.cfg.sam_checkpoint
        ).to(self.cfg.device)
        self._sam = SamAutomaticMaskGenerator(sam)

    # --- core -------------------------------------------------------------

    @torch.no_grad()
    def embed_crops(self, crops: list[np.ndarray]) -> torch.Tensor:
        """Run SigLIP 2 image tower over a list of RGB crops -> [len, D]."""
        self._load_siglip()
        out = []
        for i in range(0, len(crops), self.cfg.crop_batch_size):
            batch = crops[i : i + self.cfg.crop_batch_size]
            inputs = self._siglip_processor(images=batch, return_tensors="pt").to(
                self.cfg.device
            )
            feats = self._siglip.get_image_features(**inputs)
            feats = F.normalize(feats, dim=-1)
            out.append(feats.float().cpu())
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """image: uint8 [H, W, 3] RGB -> float32 [H, W, D] feature map."""
        self._load_sam()
        self._load_siglip()

        masks = self._sam.generate(image)
        masks = [m for m in masks if m["area"] >= self.cfg.min_mask_area]
        if not masks:
            # No masks -> zero map. The field will get no feature signal here.
            return np.zeros((*image.shape[:2], self.feature_dim), dtype=np.float32)

        crops, seg_masks = [], []
        for m in masks:
            seg = m["segmentation"]
            ys, xs = np.where(seg)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crop = image[y0:y1, x0:x1].copy()
            crops.append(crop)
            seg_masks.append(seg)

        embeddings = self.embed_crops(crops).numpy()  # [M, D]

        # Paint embeddings back. Later (higher-scoring) masks overwrite earlier
        # ones; SAM returns masks roughly small->large, so this keeps finer
        # objects on top of the background they sit in.
        H, W = image.shape[:2]
        feat_map = np.zeros((H, W, self.feature_dim), dtype=np.float32)
        order = np.argsort([m["area"] for m in masks])[::-1]  # large first
        for idx in order:
            feat_map[seg_masks[idx]] = embeddings[idx]
        return feat_map


def extract_dataset(
    image_dir: str | Path,
    out_dir: str | Path,
    cfg: FeatureExtractorConfig | None = None,
) -> None:
    """Extract feature maps for every image in a directory."""
    import cv2

    cfg = cfg or FeatureExtractorConfig()
    extractor = FeatureExtractor(cfg)
    image_dir, out_dir = Path(image_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        feat_map = extractor(rgb)
        np.save(out_dir / f"{p.stem}.npy", feat_map)
        print(f"[{i + 1}/{len(paths)}] {p.name} -> {feat_map.shape}")
