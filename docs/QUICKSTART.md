# Quick Start

## Prerequisites

```bash
pip install modal gsplat open-clip-torch transformers plyfile opencv-python
modal setup   # authenticate with Modal
```

## Step 1 — Prepare your scene

You need a folder of images of the scene you want to index. The more varied the camera angles, the better the 3D localization.

```
my_scene/
└── images/
    ├── 0001.jpg
    ├── 0002.jpg
    └── ...
```

## Step 2 — Train the Gaussian Splat

```bash
modal run modal/run_gsplat_train.py --scene my_scene --images /path/to/my_scene
```

This runs COLMAP for camera pose estimation, then trains a 3D Gaussian Splatting model for 7000 iterations.

## Step 3 — Build the semantic feature field

```bash
modal run modal/run_index.py::index --scene my_scene --epochs 30
```

This:
- Extracts SigLIP 2 features from every training image
- Trains a per-scene autoencoder (1152-dim → 16-dim)
- Distills the feature field by training each Gaussian's latent to reconstruct the SigLIP features from its visible views

Runtime: ~2–5 hours depending on scene size and GPU.

## Step 4 — Query in 3D

```bash
modal run modal/run_index.py::query --scene my_scene --prompt "red car"
```

Output:
```
"red car"
  centroid: (-0.347, 0.372, 0.142)
  aabb:     min=[-2.912, -1.078, -0.262]  max=[3.885, 1.94, 0.725]
  support:  473054 gaussians
```

## Step 5 — Render semantic heatmap

```bash
modal run modal/heatmap.py --scene my_scene --prompt "red car"
modal volume get gaussian-outputs my_scene/heatmap_red_car.mp4 ./heatmap_red_car.mp4
```

## Step 6 — Export for SuperSplat viewer

```bash
modal run scripts/convert_ply.py --scene my_scene
modal volume get gaussian-outputs my_scene/my_scene.ply ./my_scene.ply
# Drag my_scene.ply into https://supersplat.playcanvas.com
```

## Step 7 — 2D object detection (workaround for tight boxes)

```bash
modal run modal/run_detect.py --scene my_scene --prompt "car" --max-images 30
modal volume get gaussian-outputs my_scene/vgrep3d/detect/ ./detect_out/
```

---

## Tips

- **Better results**: capture your scene by walking around objects with a handheld camera, not driving past them
- **Threshold tuning**: use `--threshold 0.5` to `0.7` for stricter matching; default 0.6
- **Scene size**: scenes up to ~1M Gaussians fit on A10G; larger scenes need batched decoding
