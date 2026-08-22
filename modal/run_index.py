"""Modal entrypoint for vgrep3d: distill a SigLIP-2 feature field into a trained
splat, then query it in 3D.

Reliability notes (added after the first run hit a 3-hour function timeout at
epoch 9/30):
  * Function timeout raised to 12 hours.
  * SAM+SigLIP extraction is now SKIPPED if the feature .npz files already
    exist on the volume from a prior run -- retries no longer re-pay the
    20-30 min SAM cost every time.
  * Distillation checkpoints latents.pt every few epochs and RESUMES from it
    automatically if present, so a future timeout only loses a few epochs of
    progress, not the whole run.
  * Each epoch trains on a random subset of views (views_per_epoch) instead
    of all of them, cutting wall time per epoch roughly proportionally.

Usage (from repo root):
    modal run modal/run_index.py::index --scene truck_test --epochs 30
    modal run modal/run_index.py::query --scene truck_test --prompt "red truck"

Pull results:
    modal volume get gaussian-outputs truck_test/vgrep3d/ ./vgrep3d_out/
"""

from __future__ import annotations

import modal
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

app = modal.App("vgrep3d-index")
GSPLAT_VERSION = "1.5.3"
SAM_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0", "wget")
    .pip_install(
        f"gsplat=={GSPLAT_VERSION}",
        "numpy",
        "Pillow",
        "tqdm",
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "transformers>=4.49.0",
        "segment-anything",
        "huggingface_hub",
        "safetensors",
    )
    .run_commands(f"wget -q {SAM_URL} -O /root/sam_vit_h_4b8939.pth")
    .add_local_dir("src/vgrep3d", remote_path="/root/vgrep3d")
)

volume = modal.Volume.from_name("gaussian-outputs", create_if_missing=False)
OUT = "/outputs"
TIMEOUT_SECONDS = 12 * 60 * 60  # 12 hours


# --------------------------------------------------------------------------- #
# shared scene loading
# --------------------------------------------------------------------------- #
def _find_scene_paths(scene: str):
    from pathlib import Path

    root = Path(OUT) / scene
    try:
        ckpt = next(root.glob("**/ckpts/ckpt_*_rank0.pt"))
    except StopIteration:
        ckpt = next(root.glob("**/*.pt"))
    try:
        sparse = next(root.glob("**/colmap/sparse/0"))
    except StopIteration:
        sparse = next(root.glob("**/sparse/0"))
    return root, ckpt, sparse


def _locate_image_dir(root, colmap_names):
    from pathlib import Path

    stems = {Path(n).stem for n in colmap_names}
    for cand in [root / "colmap" / "images", root / "images",
                 root / "gsplat" / "images", root / "colmap" / "input"]:
        if cand.is_dir():
            got = {p.stem for p in cand.iterdir()
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
            if stems & got:
                return cand
    return None


def _build_field(ckpt, max_gaussians, device):
    import sys
    sys.path.insert(0, "/root")
    from vgrep3d.field.ckpt_io import load_gsplat_checkpoint
    from vgrep3d.field.feature_field import FeatureField

    g = load_gsplat_checkpoint(ckpt, device=device, max_gaussians=max_gaussians)
    field = FeatureField(
        means=g.means,
        quats=g.quats,
        scales=g.scales,
        opacities=g.opacities.reshape(-1),
        sh_colors=g.sh,
        latent_dim=16,
        sh_degree=g.sh_degree,
    )
    print(f"[scene] {g.num_points:,} gaussians, sh_degree={g.sh_degree}")
    return field, g


def _render_rgb_frames(g, cameras, images, image_dir, device):
    import numpy as np
    import cv2
    from pathlib import Path
    from gsplat import rasterization
    from vgrep3d.field.colmap_io import get_intrinsics
    import torch

    image_dir.mkdir(parents=True, exist_ok=True)
    colors = (g.sh[:, 0, :] * 0.28209479177387814 + 0.5).clamp(0, 1).to(device)
    for im in images.values():
        cam = cameras[im.camera_id]
        K = torch.tensor(get_intrinsics(cam), dtype=torch.float32, device=device)
        w2c = torch.tensor(im.world_to_camera, dtype=torch.float32, device=device)
        rgb, _, _ = rasterization(
            means=g.means.to(device), quats=g.quats.to(device),
            scales=g.scales.to(device), opacities=g.opacities.reshape(-1).to(device),
            colors=colors, viewmats=w2c[None], Ks=K[None],
            width=int(cam.width), height=int(cam.height),
        )
        img = (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        cv2.imwrite(str(image_dir / f"{Path(im.name).stem}.png"),
                    cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print(f"[scene] rendered {len(images)} RGB frames -> {image_dir}")


def _features_cached(feat_dir, colmap_names) -> bool:
    """True if a .npz already exists for every COLMAP-referenced image."""
    from pathlib import Path

    if not feat_dir.is_dir():
        return False
    stems = {Path(n).stem for n in colmap_names}
    have = {p.stem for p in feat_dir.glob("*.npz")}
    return stems.issubset(have) and len(stems) > 0


def _build_cameras(cameras, images, image_dir, feat_dir):
    import numpy as np
    import cv2
    import torch
    from pathlib import Path
    from vgrep3d.field.colmap_io import get_intrinsics

    cams = []
    for im in images.values():
        stem = Path(im.name).stem
        fp = feat_dir / f"{stem}.npz"
        if not fp.exists():
            continue
        img_path = None
        for ext in (".png", ".jpg", ".jpeg"):
            p = image_dir / f"{stem}{ext}"
            if p.exists():
                img_path = p
                break
        if img_path is None:
            continue
        h, w = cv2.imread(str(img_path)).shape[:2]

        cam = cameras[im.camera_id]
        K = get_intrinsics(cam).astype(np.float32)
        sx, sy = w / float(cam.width), h / float(cam.height)
        K[0, 0] *= sx; K[0, 2] *= sx
        K[1, 1] *= sy; K[1, 2] *= sy

        cams.append({
            "viewmat": torch.tensor(im.world_to_camera, dtype=torch.float32),
            "K": torch.tensor(K, dtype=torch.float32),
            "width": w, "height": h,
            "feature_path": str(fp),
        })
    return cams


# --------------------------------------------------------------------------- #
# index: extract (skippable) -> AE -> distill (checkpointed + resumable)
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu="A10G", timeout=TIMEOUT_SECONDS,
              volumes={OUT: volume}, memory=32768)
def index_scene(
    scene: str = "truck_test",
    epochs: int = 30,
    latent_dim: int = 16,
    max_gaussians: int | None = None,
    views_per_epoch: int | None = 60,
    save_every: int = 3,
    force_reextract: bool = False,
):
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch

    from vgrep3d.field.autoencoder import FeatureAutoencoder, train_autoencoder
    from vgrep3d.field.train import save_field, train_feature_field, load_latents
    from vgrep3d.field.colmap_io import load_colmap_scene
    from vgrep3d.preprocess.features import FeatureExtractorConfig, extract_dataset

    device = "cuda"
    root, ckpt, sparse = _find_scene_paths(scene)
    print(f"[paths] ckpt={ckpt}\n[paths] sparse={sparse}")

    work = root / "vgrep3d"
    feat_dir = work / "features"
    work.mkdir(parents=True, exist_ok=True)

    field, g = _build_field(ckpt, max_gaussians, device)
    cameras, images = load_colmap_scene(sparse)
    colmap_names = [im.name for im in images.values()]

    image_dir = _locate_image_dir(root, colmap_names)
    if image_dir is None:
        from pathlib import Path
        image_dir = Path("/root/scene_images")
        _render_rgb_frames(g, cameras, images, image_dir, device)
    else:
        print(f"[scene] using real images at {image_dir}")

    # 1. SigLIP-2 feature maps -- skip if already extracted on the volume
    if force_reextract or not _features_cached(feat_dir, colmap_names):
        print("== extracting feature maps ==")
        extract_dataset(
            image_dir, feat_dir,
            FeatureExtractorConfig(device=device, sam_checkpoint="/root/sam_vit_h_4b8939.pth"),
        )
        volume.commit()
    else:
        print(f"[scene] features already cached at {feat_dir} -- skipping extraction")

    # 2. per-scene autoencoder -- skip if already trained on the volume
    ae_path = work / "autoencoder.pt"
    sample = next(feat_dir.glob("*.npz"))
    in_dim = int(np.load(sample)["table"].shape[1])
    ae = FeatureAutoencoder(in_dim=in_dim, latent_dim=latent_dim)
    if ae_path.exists() and not force_reextract:
        print(f"[scene] loading cached autoencoder from {ae_path}")
        ck = torch.load(ae_path, map_location=device)
        ae.load_state_dict(ck["state_dict"])
    else:
        print("== training autoencoder ==")
        ae = train_autoencoder(ae, feat_dir, device=device)
        torch.save({"state_dict": ae.state_dict(), "in_dim": in_dim,
                    "latent_dim": latent_dim}, ae_path)
        volume.commit()

    # 3. distill into the field -- resume from checkpoint if present
    print("== distilling feature field ==")
    cams = _build_cameras(cameras, images, image_dir, feat_dir)
    print(f"[scene] {len(cams)} usable views, {views_per_epoch or len(cams)} per epoch")

    latents_path = work / "latents.pt"
    start_epoch = 0
    if latents_path.exists() and not force_reextract:
        print(f"[scene] resuming from checkpoint {latents_path}")
        field = load_latents(field, latents_path, device=device)
        # We don't persist exact epoch count across runs; conservatively
        # resume for the full remaining epoch budget rather than guessing.

    field = train_feature_field(
        field, ae, cams,
        epochs=epochs,
        device=device,
        views_per_epoch=views_per_epoch,
        save_every=save_every,
        save_path=latents_path,
        start_epoch=start_epoch,
    )
    save_field(field, latents_path)
    volume.commit()
    print(f"done -> {work}")
    return str(work)


# --------------------------------------------------------------------------- #
# query: locate a prompt in 3D
# --------------------------------------------------------------------------- #
@app.function(image=image, gpu="A10G", timeout=60 * 60,
              volumes={OUT: volume}, memory=32768)
def query_scene(scene: str = "truck_test", prompt: str = "red truck",
                threshold: float = 0.6, max_gaussians: int | None = None):
    import sys
    sys.path.insert(0, "/root")
    import torch

    from vgrep3d.field.autoencoder import FeatureAutoencoder
    from vgrep3d.field.train import load_latents
    from vgrep3d.query.query import Query3D

    device = "cuda"
    root, ckpt, _ = _find_scene_paths(scene)
    work = root / "vgrep3d"

    ae_ck = torch.load(work / "autoencoder.pt", map_location=device)
    ae = FeatureAutoencoder(in_dim=ae_ck["in_dim"], latent_dim=ae_ck["latent_dim"])
    ae.load_state_dict(ae_ck["state_dict"])

    field, _ = _build_field(ckpt, max_gaussians, device)
    field = load_latents(field, work / "latents.pt", device=device)

    q = Query3D(field, ae, device=device)
    res = q.locate_3d(prompt, threshold=threshold)
    if not res["found"]:
        print(f'"{prompt}": nothing above threshold {threshold}')
        return
    c = res["centroid"].tolist()
    lo, hi = res["aabb_min"].tolist(), res["aabb_max"].tolist()
    print(f'\n"{prompt}"')
    print(f"  centroid: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    print(f"  aabb:     min={[round(x,3) for x in lo]}  max={[round(x,3) for x in hi]}")
    print(f"  support:  {res['num_gaussians']} gaussians")


@app.local_entrypoint()
def index(scene: str = "truck_test", epochs: int = 30, force_reextract: bool = False):
    print(index_scene.remote(scene=scene, epochs=epochs, force_reextract=force_reextract))


@app.local_entrypoint()
def query(scene: str = "truck_test", prompt: str = "red truck", threshold: float = 0.6):
    query_scene.remote(scene=scene, prompt=prompt, threshold=threshold)
