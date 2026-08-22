"""Extract 2D SigLIP 2 feature maps from training images.

Pipeline per image:
    1. SAM automatic mask generator -> list of binary masks
    2. For each mask, crop the masked region and run the SigLIP 2 image tower
       -> one embedding per mask (shape [D])
    3. Record which mask owns each pixel (an id map) + the per-mask embedding
       table. This is the supervision target for the 3D feature field.

Storage is compact: a mask-id map [H, W] int32 plus an embedding table
[num_masks, D] float16, saved as .npz. load_feature_map() reconstructs the
dense [H, W, D] map the training loop expects.

Note on transformers versions: in newer transformers releases,
`get_image_features(...)` can return a ModelOutput object (e.g.
BaseModelOutputWithPooling) instead of a plain tensor, depending on the model
config. `_extract_image_features` below handles both cases.
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
    min_mask_area: int = 400
    crop_batch_size: int = 32


def _extract_image_features(raw) -> torch.Tensor:
    """Normalize get_image_features() output to a plain [N, D] tensor.

    Handles both a bare tensor return and a ModelOutput-style object that
    carries the embedding under one of a few common attribute names.
    """
    if torch.is_tensor(raw):
        return raw
    for attr in ("image_embeds", "pooler_output", "last_hidden_state"):
        val = getattr(raw, attr, None)
        if val is not None:
            if val.dim() == 3:
                val = val[:, 0, :]
            return val
    if isinstance(raw, (tuple, list)) and len(raw) > 0 and torch.is_tensor(raw[0]):
        return raw[0]
    raise TypeError(f"Unrecognized get_image_features() output type: {type(raw)}")


class FeatureExtractor:
    """Turns an RGB image into a compact (id_map, table) feature representation."""

    def __init__(self, cfg: FeatureExtractorConfig):
        self.cfg = cfg
        self._siglip = None
        self._siglip_processor = None
        self._sam = None
        self.feature_dim: int | None = None

    def _load_siglip(self):
        if self._siglip is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._siglip_processor = AutoProcessor.from_pretrained(self.cfg.siglip_model)
        self._siglip = (
            AutoModel.from_pretrained(self.cfg.siglip_model).eval().to(self.cfg.device)
        )
        with torch.no_grad():
            dummy = np.zeros((16, 16, 3), dtype=np.uint8)
            inp = self._siglip_processor(images=[dummy], return_tensors="pt").to(
                self.cfg.device
            )
            raw = self._siglip.get_image_features(**inp)
            feat = _extract_image_features(raw)
        self.feature_dim = int(feat.shape[-1])
        print(f"[features] probed SigLIP image-feature dim = {self.feature_dim}")

    def _load_sam(self):
        if self._sam is not None:
            return
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[self.cfg.sam_model_type](
            checkpoint=self.cfg.sam_checkpoint
        ).to(self.cfg.device)
        self._sam = SamAutomaticMaskGenerator(sam)

    @torch.no_grad()
    def embed_crops(self, crops: list[np.ndarray]) -> np.ndarray:
        self._load_siglip()
        out = []
        for i in range(0, len(crops), self.cfg.crop_batch_size):
            batch = crops[i : i + self.cfg.crop_batch_size]
            inputs = self._siglip_processor(images=batch, return_tensors="pt").to(
                self.cfg.device
            )
            raw = self._siglip.get_image_features(**inputs)
            feats = _extract_image_features(raw)
            feats = F.normalize(feats, dim=-1)
            out.append(feats.float().cpu())
        return torch.cat(out, dim=0).numpy()

    @torch.no_grad()
    def extract(self, image: np.ndarray) -> dict:
        self._load_sam()
        self._load_siglip()
        H, W = image.shape[:2]

        masks = self._sam.generate(image)
        masks = [m for m in masks if m["area"] >= self.cfg.min_mask_area]
        if not masks:
            return {
                "ids": np.full((H, W), -1, dtype=np.int32),
                "table": np.zeros((0, self.feature_dim), dtype=np.float16),
            }

        crops, seg_masks = [], []
        for m in masks:
            seg = m["segmentation"]
            ys, xs = np.where(seg)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            crops.append(image[y0:y1, x0:x1].copy())
            seg_masks.append(seg)

        embeddings = self.embed_crops(crops).astype(np.float16)

        id_map = np.full((H, W), -1, dtype=np.int32)
        order = np.argsort([m["area"] for m in masks])[::-1]
        for idx in order:
            id_map[seg_masks[idx]] = idx
        return {"ids": id_map, "table": embeddings}


def load_feature_map(path: str | Path) -> np.ndarray:
    """Reconstruct the dense [H, W, D] float32 feature map from a compact .npz."""
    data = np.load(path)
    ids, table = data["ids"], data["table"]
    H, W = ids.shape
    D = table.shape[1]
    out = np.zeros((H, W, D), dtype=np.float32)
    valid = ids >= 0
    if table.shape[0] > 0:
        out[valid] = table[ids[valid]].astype(np.float32)
    return out


def extract_dataset(
    image_dir: str | Path,
    out_dir: str | Path,
    cfg: FeatureExtractorConfig | None = None,
) -> None:
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
        rec = extractor.extract(rgb)
        np.savez_compressed(out_dir / f"{p.stem}.npz", ids=rec["ids"], table=rec["table"])
        print(f"[{i + 1}/{len(paths)}] {p.name} -> ids{rec['ids'].shape} "
              f"table{rec['table'].shape}")
