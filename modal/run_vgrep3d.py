"""Self-contained Modal entrypoint – no local package mount needed."""

from __future__ import annotations
from pathlib import Path
import modal

app = modal.App("vgrep3d")
GSPLAT_VERSION = "1.5.3"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        f"gsplat=={GSPLAT_VERSION}",
        "numpy",
        "Pillow",
        "tqdm",
        "opencv-python-headless",
        "imageio[ffmpeg]",
    )
)

volume = modal.Volume.from_name("gaussian-outputs", create_if_missing=False)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60,
    volumes={"/outputs": volume},
    memory=32768,
)
def run_search(scene: str = "chair30k", prompt: str = "chair", max_gaussians: int | None = 500_000):
    import struct
    import torch
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path

    # ---------- minimal COLMAP reader ----------
    def qvec_to_rotmat(qvec):
        w, x, y, z = qvec
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
        ], dtype=np.float64)

    def read_next(fid, n, fmt):
        return struct.unpack("<" + fmt, fid.read(n))

    def load_colmap(sparse_dir: Path):
        # cameras
        cameras = {}
        with open(sparse_dir / "cameras.bin", "rb") as f:
            num = read_next(f, 8, "Q")[0]
            for _ in range(num):
                cid = read_next(f, 4, "i")[0]
                model_id = read_next(f, 4, "i")[0]
                w = read_next(f, 8, "Q")[0]
                h = read_next(f, 8, "Q")[0]
                nparams = {0: 3, 1: 4, 4: 8}.get(model_id, 4)
                params = np.array(read_next(f, 8 * nparams, "d" * nparams))
                cameras[cid] = {"width": w, "height": h, "params": params, "model_id": model_id}

        # images
        images = {}
        with open(sparse_dir / "images.bin", "rb") as f:
            num = read_next(f, 8, "Q")[0]
            for _ in range(num):
                iid = read_next(f, 4, "i")[0]
                qvec = np.array(read_next(f, 32, "dddd"))
                tvec = np.array(read_next(f, 24, "ddd"))
                cid = read_next(f, 4, "i")[0]
                name = b""
                while True:
                    c = f.read(1)
                    if c == b"\x00":
                        break
                    name += c
                name = name.decode()
                n2d = read_next(f, 8, "Q")[0]
                f.read(24 * n2d)
                R = qvec_to_rotmat(qvec)
                w2c = np.eye(4)
                w2c[:3, :3] = R
                w2c[:3, 3] = tvec
                images[iid] = {"name": name, "camera_id": cid, "w2c": w2c}

        images = dict(sorted(images.items(), key=lambda kv: kv[1]["name"]))
        return cameras, images

    # ---------- load Gaussians from checkpoint ----------
    scene_root = Path("/outputs") / scene
    ckpt_path = next(scene_root.glob("**/ckpts/ckpt_*_rank0.pt"))
    sparse_dir = next(scene_root.glob("**/colmap/sparse/0"))

    print(f"Checkpoint: {ckpt_path}")
    print(f"COLMAP:     {sparse_dir}")

    device = torch.device("cuda")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    splats = ck["splats"]

    means = splats["means"].float()
    opacities = splats["opacities"].float()
    quats = splats["quats"].float()
    scales = splats["scales"].float()
    sh0 = splats["sh0"].float()
    shN = splats.get("shN")

    N = means.shape[0]
    if max_gaussians and N > max_gaussians:
        idx = torch.randperm(N)[:max_gaussians]
        means, opacities, quats, scales, sh0 = [t[idx] for t in (means, opacities, quats, scales, sh0)]
        if shN is not None:
            shN = shN[idx]
        N = max_gaussians

    if opacities.min() < 0 or opacities.max() > 1:
        opacities = torch.sigmoid(opacities)
    if scales.min() <= 0:
        scales = torch.exp(scales)
    quats = F.normalize(quats, dim=-1)

    print(f"Loaded {N:,} Gaussians")

    cameras, images = load_colmap(sparse_dir)
    print(f"{len(cameras)} camera(s), {len(images)} images")

    # ---------- simple trajectory render ----------
    from gsplat.rendering import rasterization
    import imageio

    out_dir = scene_root / "vgrep3d" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / f"heatmap_{prompt.replace(' ', '_')}.mp4"

    frames = []
    sorted_ims = list(images.values())
    step = max(1, len(sorted_ims) // 40)

    colors = (sh0.squeeze(1) * 0.28209479177387814 + 0.5).clamp(0, 1).to(device)

    for i, im in enumerate(sorted_ims[::step]):
        cam = cameras[im["camera_id"]]
        params = cam["params"]
        if len(params) >= 4:
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        else:
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32, device=device)

        w2c = torch.from_numpy(im["w2c"]).float().to(device)

        render_colors, _, _ = rasterization(
            means=means.to(device),
            quats=quats.to(device),
            scales=scales.to(device),
            opacities=opacities.to(device),
            colors=colors,
            viewmats=w2c.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=int(cam["width"]),
            height=int(cam["height"]),
        )
        rgb = (render_colors[0].cpu().numpy() * 255).astype(np.uint8)
        frames.append(rgb)
        if i % 5 == 0:
            print(f"  frame {i}")

    imageio.mimwrite(str(video_path), frames, fps=12, quality=8)
    print(f"✓ Wrote {video_path}")
    volume.commit()
    return str(video_path)


@app.local_entrypoint()
def main(scene: str = "chair30k", prompt: str = "chair", max_gaussians: int = 500000):
    path = run_search.remote(scene=scene, prompt=prompt, max_gaussians=max_gaussians)
    print(f"\nDone → {path}")
    print(f"Pull with:\n  modal volume get gaussian-outputs {scene}/vgrep3d/demo/ ./demo_out/")
