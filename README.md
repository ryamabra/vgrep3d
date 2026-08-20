# vgrep3d

**Local-first, open-vocabulary 3D search over Gaussian splats.**

`vgrep` let you search a folder of images by text, locally, with SigLIP 2.
`vgrep3d` lifts that idea into 3D: index a Gaussian-splat scene once, then ask
where things are and get an answer in the scene's coordinate frame.

```bash
vgrep3d index  scenes/office
vgrep3d query  scenes/office "fire extinguisher"
# "fire extinguisher"
#   centroid: (1.243, 0.512, -2.881)
#   aabb:     min=[1.1, 0.2, -3.0]  max=[1.4, 0.9, -2.7]
#   support:  418 gaussians
```

## What it does

It attaches a language feature to every Gaussian *without touching the geometry
you already trained*. Your splat still renders color exactly as before; a second,
parallel channel carries a SigLIP 2 latent that you can query by text.

```
                 ┌─ RGB (SH colors, frozen) ──────────────► image
 3D Gaussians ───┤
                 └─ latent (trainable) ──[decode]──► SigLIP feature ─┐
                                                                     │ cosine
 text prompt ──[SigLIP 2 text tower]──► text embedding ─────────────┘
                                                                     ▼
                                                        per-Gaussian relevance
                                                        → 3D centroid / AABB / heatmap
```

## How it works (4 additions, 0 replacements)

1. **New attribute.** Each Gaussian gets a low-dim latent (default 3-D) alongside
   its color. Raw SigLIP vectors (~1152-D) are too big to store per-Gaussian, so
   a small per-scene autoencoder compresses them.
2. **Same rasterizer, extra channel.** The latent is alpha-blended by the exact
   `gsplat` rasterization used for color — just pointed at different data.
3. **New supervision.** Training images are preprocessed with SAM + SigLIP 2 into
   2D feature maps; a cosine loss distills them into the field. Geometry stays
   frozen (two-stage: RGB first, features second).
4. **New query path (inference only).** Encode text with SigLIP 2, cosine-sim
   against the decoded per-Gaussian features, threshold → 3D localization.

## Install

```bash
pip install -e .
# SAM weights (ViT-H):
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

Needs a CUDA GPU. Developed against an A10G. `gsplat` must be built with CUDA.

## Scene layout

```
scenes/office/
  images/          training RGB frames
  cameras.json     [{viewmat:4x4, K:3x3, width, height, image_stem}, ...]
  point_cloud.ply  your trained splat (INRIA/gsplat .ply)
```

`index` writes everything else into `scenes/office/vgrep3d/`.

## Status & scope

This is an engineering-first reimplementation in the LangSplat / Feature-3DGS
lineage, built to be small, single-GPU, and CLI-driven rather than to chase a
leaderboard. The SigLIP 2 backbone (vs the usual CLIP) gives stronger
open-vocab segmentation features; note that a SAM + SigLIP 2 feature-collection
step also appears in SceneSplat, so the backbone choice is a sensible default,
not the novel contribution.

Known rough edges (good next-work targets):
- **Multi-view feature inconsistency.** Per-view SAM+SigLIP maps disagree across
  frames; this is an open problem (see VALA, SceneSplat). Swapping SAM → SAM 2
  for mask tracking is the first lever to try.
- Single mask scale (LangSplat uses three).
- No relevance calibration beyond per-query min-max normalization.

## Evaluation

Point it at LERF-OVS or 3D-OVS and report localization accuracy / mIoU. A
CLIP-vs-SigLIP-2 ablation on the same scenes is the cleanest quantitative story.

## License

MIT.
