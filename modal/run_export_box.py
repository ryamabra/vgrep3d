"""Export a vgrep3d detection as a clean wireframe box overlay -- the original
splat is written UNCHANGED (no recoloring of real Gaussians), and a small set
of new synthetic Gaussians are appended tracing the 12 edges of the detected
AABB as a thin, solid, bright rectangle. This avoids the "green cloud" look
of tinting every matched Gaussian and instead gives a clean static marker,
like a bounding-box annotation you'd see in a 2D detector, but in 3D and
viewable natively in SuperSplat.

Usage (from repo root):
    modal run modal/run_export_box.py --scene truck_test --prompt "red truck"

Pull the result:
    modal volume get gaussian-outputs truck_test/vgrep3d/box_red_truck.ply ./box_red_truck.ply

Then open superspl.at/editor and drag the .ply in.
"""

from __future__ import annotations

import modal

app = modal.App("vgrep3d-export-box")
GSPLAT_VERSION = "1.5.3"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("git")
    .pip_install(f"gsplat=={GSPLAT_VERSION}", "numpy", "transformers>=4.49.0")
    .add_local_dir("src/vgrep3d", remote_path="/root/vgrep3d")
)

volume = modal.Volume.from_name("gaussian-outputs", create_if_missing=False)
OUT = "/outputs"
DC_FACTOR = 0.28209479177387814

_BOX_EDGES = [(0, 1), (2, 3), (4, 5), (6, 7),
              (0, 2), (1, 3), (4, 6), (5, 7),
              (0, 4), (1, 5), (2, 6), (3, 7)]


def _find_scene_paths(scene: str):
    from pathlib import Path
    root = Path(OUT) / scene
    try:
        ckpt = next(root.glob("**/ckpts/ckpt_*_rank0.pt"))
    except StopIteration:
        ckpt = next(root.glob("**/*.pt"))
    return root, ckpt


def _aabb_corners(mn, mx):
    import numpy as np
    xs, ys, zs = (mn[0], mx[0]), (mn[1], mx[1]), (mn[2], mx[2])
    return np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)


def _sample_box_edges(mn, mx, points_per_unit=40):
    """Points strictly along the 12 edges of the AABB -- verified to lie
    exactly on edges (>=2 coordinates at the min/max extreme), not faces."""
    import numpy as np
    corners = _aabb_corners(mn, mx)
    pts = []
    for a, b in _BOX_EDGES:
        length = float(np.linalg.norm(corners[b] - corners[a]))
        n = max(2, int(length * points_per_unit))
        t = np.linspace(0, 1, n)[:, None]
        pts.append(corners[a] + t * (corners[b] - corners[a]))
    return np.concatenate(pts, axis=0).astype(np.float32)


def _rgb_to_dc(rgb):
    import numpy as np
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / DC_FACTOR


def _write_ply(path, means, sh0, shN, opacity_raw, scale_raw, quats):
    import numpy as np
    N = means.shape[0]
    n_rest = 0 if shN is None else shN.shape[1] * shN.shape[2]
    props = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(n_rest)]
    props += ["opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "\n".join([
        "ply", "format binary_little_endian 1.0", f"element vertex {N}",
        *[f"property float {p}" for p in props], "end_header", "",
    ]).encode("ascii")
    dtype = np.dtype([(p, "<f4") for p in props])
    rec = np.zeros(N, dtype=dtype)
    rec["x"], rec["y"], rec["z"] = means[:, 0], means[:, 1], means[:, 2]
    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = sh0[:, 0], sh0[:, 1], sh0[:, 2]
    if n_rest > 0:
        flat = np.transpose(shN, (0, 2, 1)).reshape(N, -1)
        for i in range(n_rest):
            rec[f"f_rest_{i}"] = flat[:, i]
    rec["opacity"] = opacity_raw
    rec["scale_0"], rec["scale_1"], rec["scale_2"] = scale_raw[:, 0], scale_raw[:, 1], scale_raw[:, 2]
    rec["rot_0"], rec["rot_1"], rec["rot_2"], rec["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    with open(path, "wb") as f:
        f.write(header)
        f.write(rec.tobytes())


@app.function(image=image, gpu="A10G", timeout=30 * 60,
              volumes={OUT: volume}, memory=32768)
def export_box_ply(
    scene: str = "truck_test",
    prompt: str = "red truck",
    threshold: float = 0.6,
    max_gaussians: int | None = None,
    box_color=(0.0, 1.0, 0.0),
    thickness_frac: float = 0.003,   # fraction of the box's max dimension
    points_per_unit: float = 40.0,
):
    import sys
    sys.path.insert(0, "/root")
    import numpy as np
    import torch

    from vgrep3d.field.ckpt_io import load_gsplat_checkpoint
    from vgrep3d.field.feature_field import FeatureField
    from vgrep3d.field.autoencoder import FeatureAutoencoder
    from vgrep3d.field.train import load_latents
    from vgrep3d.query.query import Query3D

    device = "cuda"
    root, ckpt = _find_scene_paths(scene)
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
        print(f'"{prompt}": nothing above threshold {threshold} -- nothing to export')
        return None
    mn, mx = res["aabb_min"].numpy(), res["aabb_max"].numpy()
    print(f"[box] aabb min={mn.tolist()} max={mx.tolist()} "
          f"support={res['num_gaussians']} gaussians (context only, not drawn)")

    # -- original splat, completely untouched --
    means = g.means.cpu().numpy().astype(np.float32)
    opac = g.opacities.reshape(-1).cpu().numpy().astype(np.float32)
    opac = np.clip(opac, 1e-6, 1 - 1e-6)
    opacity_raw = np.log(opac / (1 - opac))
    scale_raw = np.log(g.scales.cpu().numpy().astype(np.float32))
    quats = g.quats.cpu().numpy().astype(np.float32)
    sh = g.sh.cpu().numpy().astype(np.float32)
    sh0 = sh[:, 0, :].copy()
    shN = sh[:, 1:, :].copy() if sh.shape[1] > 1 else None

    # -- synthetic wireframe box: brand new points, original data untouched --
    box_pts = _sample_box_edges(mn, mx, points_per_unit=points_per_unit)
    n_box = box_pts.shape[0]
    box_dim = float(max(mx - mn))
    box_scale = max(box_dim * thickness_frac, 1e-4)
    box_scale_raw = np.full((n_box, 3), np.log(box_scale), dtype=np.float32)
    box_opacity_raw = np.full(n_box, np.log(0.98 / 0.02), dtype=np.float32)
    box_quats = np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (n_box, 1))
    box_dc = _rgb_to_dc(box_color)
    box_sh0 = np.tile(box_dc, (n_box, 1)).astype(np.float32)
    box_shN = None
    if shN is not None:
        box_shN = np.zeros((n_box, shN.shape[1], shN.shape[2]), dtype=np.float32)

    all_means = np.concatenate([means, box_pts], axis=0)
    all_sh0 = np.concatenate([sh0, box_sh0], axis=0)
    all_shN = None
    if shN is not None:
        all_shN = np.concatenate([shN, box_shN], axis=0)
    all_opacity_raw = np.concatenate([opacity_raw, box_opacity_raw], axis=0)
    all_scale_raw = np.concatenate([scale_raw, box_scale_raw], axis=0)
    all_quats = np.concatenate([quats, box_quats], axis=0)

    out_path = work / f"box_{prompt.replace(' ', '_')}.ply"
    _write_ply(out_path, all_means, all_sh0, all_shN,
               all_opacity_raw, all_scale_raw, all_quats)
    volume.commit()
    print(f"[box] wrote {out_path} "
          f"({means.shape[0]:,} original + {n_box:,} box-wireframe points)")
    return str(out_path)


@app.local_entrypoint()
def export(scene: str = "truck_test", prompt: str = "red truck", threshold: float = 0.6):
    path = export_box_ply.remote(scene=scene, prompt=prompt, threshold=threshold)
    if path:
        fname = path.split("/")[-1]
        print(f"\nDone -> {path}")
        print(f"Pull with:\n  modal volume get gaussian-outputs "
              f"{scene}/vgrep3d/{fname} ./{fname}")
        print("Then open https://superspl.at/editor and drag the .ply in.")
