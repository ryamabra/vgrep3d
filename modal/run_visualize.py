"""Modal entrypoint: visualize a vgrep3d detection.

Renders the trained Gaussian scene from several camera views with:
  - the RGB scene (frozen SH colors)
  - a red relevance heatmap overlaid (per-Gaussian text-relevance, rendered
    through the same rasterizer as color -- this is the actual signal the
    field uses to answer the query)
  - the located 3D AABB drawn as a green wireframe box, reprojected into
    each view from locate_3d()'s min/max corners

Usage (from repo root):
    modal run modal/run_visualize.py --scene truck_test --prompt "red truck"

Pull the result:
    modal volume get gaussian-outputs truck_test/vgrep3d/viz/ ./viz_out/
"""

from __future__ import annotations

import modal

app = modal.App("vgrep3d-viz")
GSPLAT_VERSION = "1.5.3"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        f"gsplat=={GSPLAT_VERSION}",
        "numpy",
        "opencv-python-headless",
        "imageio[ffmpeg]",
        "transformers>=4.49.0",
    )
    .add_local_dir("src/vgrep3d", remote_path="/root/vgrep3d")
)

volume = modal.Volume.from_name("gaussian-outputs", create_if_missing=False)
OUT = "/outputs"


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


# 8 corners of an AABB, indexed by bit pattern (ix, iy, iz)
def _aabb_corners(mn, mx):
    import numpy as np
    xs, ys, zs = (mn[0], mx[0]), (mn[1], mx[1]), (mn[2], mx[2])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)


_BOX_EDGES = [
    (0, 1), (2, 3), (4, 5), (6, 7),   # along z
    (0, 2), (1, 3), (4, 6), (5, 7),   # along y
    (0, 4), (1, 5), (2, 6), (3, 7),   # along x
]


def _project(pts_world, w2c, K):
    """pts_world: [N,3] numpy -> uv [N,2], in_front [N] bool."""
    import numpy as np
    N = pts_world.shape[0]
    homog = np.concatenate([pts_world, np.ones((N, 1), dtype=np.float32)], axis=1)
    cam = (w2c @ homog.T).T[:, :3]
    z = cam[:, 2]
    in_front = z > 1e-4
    zc = np.where(in_front, z, 1.0)
    proj = (K @ (cam / zc[:, None]).T).T
    return proj[:, :2], in_front


def _draw_box(img, corners_world, w2c, K, color=(0, 255, 0), thickness=2):
    import cv2
    uv, in_front = _project(corners_world, w2c, K)
    for a, b in _BOX_EDGES:
        if in_front[a] and in_front[b]:
            pa = tuple(int(x) for x in uv[a])
            pb = tuple(int(x) for x in uv[b])
            cv2.line(img, pa, pb, color, thickness, cv2.LINE_AA)
    return img


@app.function(image=image, gpu="A10G", timeout=30 * 60,
              volumes={OUT: volume}, memory=32768)
def visualize_scene(
    scene: str = "truck_test",
    prompt: str = "red truck",
    threshold: float = 0.6,
    num_views: int = 12,
    max_gaussians: int | None = None,
):
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import cv2
    import imageio
    import torch

    from vgrep3d.field.ckpt_io import load_gsplat_checkpoint
    from vgrep3d.field.feature_field import FeatureField
    from vgrep3d.field.autoencoder import FeatureAutoencoder
    from vgrep3d.field.train import load_latents
    from vgrep3d.field.colmap_io import load_colmap_scene, get_intrinsics
    from vgrep3d.query.query import Query3D

    device = "cuda"
    root, ckpt, sparse = _find_scene_paths(scene)
    work = root / "vgrep3d"

    g = load_gsplat_checkpoint(ckpt, device=device, max_gaussians=max_gaussians)
    field = FeatureField(
        means=g.means, quats=g.quats, scales=g.scales,
        opacities=g.opacities.reshape(-1), sh_colors=g.sh,
        latent_dim=16, sh_degree=g.sh_degree,
    )
    field = load_latents(field, work / "latents.pt", device=device)

    ae_ck = torch.load(work / "autoencoder.pt", map_location=device)
    ae = FeatureAutoencoder(in_dim=ae_ck["in_dim"], latent_dim=ae_ck["latent_dim"])
    ae.load_state_dict(ae_ck["state_dict"])

    q = Query3D(field, ae, device=device)
    res = q.locate_3d(prompt, threshold=threshold)
    if not res["found"]:
        print(f'"{prompt}": nothing above threshold {threshold} -- nothing to draw')
        return None
    mn, mx = res["aabb_min"].numpy(), res["aabb_max"].numpy()
    corners = _aabb_corners(mn, mx)
    print(f"[viz] box min={mn.tolist()} max={mx.tolist()} "
          f"support={res['num_gaussians']} gaussians")

    cameras, images = load_colmap_scene(sparse)
    all_images = list(images.values())
    step = max(1, len(all_images) // num_views)
    chosen = all_images[::step][:num_views]
    print(f"[viz] rendering {len(chosen)} of {len(all_images)} views")

    colors_rgb = (g.sh[:, 0, :] * 0.28209479177387814 + 0.5).clamp(0, 1).to(device)

    frames = []
    for im in chosen:
        cam = cameras[im.camera_id]
        K = get_intrinsics(cam).astype(np.float32)
        w2c_t = torch.tensor(im.world_to_camera, dtype=torch.float32, device=device)
        K_t = torch.tensor(K, dtype=torch.float32, device=device)
        w, h = int(cam.width), int(cam.height)

        from gsplat import rasterization
        rgb, _, _ = rasterization(
            means=field.means, quats=field.quats, scales=field.scales,
            opacities=field.opacities, colors=colors_rgb,
            viewmats=w2c_t[None], Ks=K_t[None], width=w, height=h,
        )
        img = (rgb[0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)

        heat = q.render_heatmap(prompt, w2c_t, K_t, w, h)  # [H,W] in [0,1]
        heat = heat.clamp(0, 1).cpu().numpy()
        red_overlay = np.zeros_like(img)
        red_overlay[..., 0] = (heat * 255).astype(np.uint8)  # red channel
        img = cv2.addWeighted(img, 1.0, red_overlay, 0.5, 0)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_bgr = _draw_box(img_bgr, corners, im.world_to_camera, K,
                             color=(0, 255, 0), thickness=2)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(img)

    out_dir = work / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"detect_{prompt.replace(' ', '_')}.mp4"
    imageio.mimwrite(str(out_path), frames, fps=2, quality=8)
    volume.commit()
    print(f"[viz] wrote {out_path}")
    return str(out_path)


@app.local_entrypoint()
def main(scene: str = "truck_test", prompt: str = "red truck",
         threshold: float = 0.6, num_views: int = 12):
    path = visualize_scene.remote(scene=scene, prompt=prompt,
                                   threshold=threshold, num_views=num_views)
    if path:
        print(f"\nDone -> {path}")
        print(f"Pull with:\n  modal volume get gaussian-outputs "
              f"{scene}/vgrep3d/viz/ ./viz_out/")
