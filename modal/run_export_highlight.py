"""Export a vgrep3d detection as a standard 3DGS .ply, viewable directly in
SuperSplat (https://superspl.at/editor) -- the detected Gaussians are tinted
bright green in-place, so you can orbit the actual splat and see exactly
which Gaussians the field identified, rather than a flat rendered video.

Usage (from repo root):
    modal run modal/run_export_highlight.py --scene truck_test --prompt "red truck"

Pull the result:
    modal volume get gaussian-outputs truck_test/vgrep3d/highlight_red_truck.ply ./highlight_red_truck.ply

Then open superspl.at/editor and drag the .ply in.
"""

from __future__ import annotations

import modal

app = modal.App("vgrep3d-export")
GSPLAT_VERSION = "1.5.3"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("git")
    .pip_install(f"gsplat=={GSPLAT_VERSION}", "numpy", "transformers>=4.49.0")
    .add_local_dir("src/vgrep3d", remote_path="/root/vgrep3d")
)

volume = modal.Volume.from_name("gaussian-outputs", create_if_missing=False)
OUT = "/outputs"

DC_FACTOR = 0.28209479177387814  # sqrt(1/(4*pi)), the SH0 <-> RGB conversion constant


def _find_scene_paths(scene: str):
    from pathlib import Path

    root = Path(OUT) / scene
    try:
        ckpt = next(root.glob("**/ckpts/ckpt_*_rank0.pt"))
    except StopIteration:
        ckpt = next(root.glob("**/*.pt"))
    return root, ckpt


def _rgb_to_dc(rgb):
    import numpy as np
    return (np.asarray(rgb, dtype=np.float32) - 0.5) / DC_FACTOR


def _write_ply(path, means, sh0, shN, opacity_raw, scale_raw, quats):
    """Write a standard binary_little_endian 3DGS .ply.

    means      [N,3]   float32
    sh0        [N,3]   float32  (DC-term SH coefficients, raw)
    shN        [N,K,3] float32  (higher-order SH coefficients, raw) or None
    opacity_raw[N]      float32  (pre-sigmoid)
    scale_raw  [N,3]   float32  (pre-exp, i.e. log-scale)
    quats      [N,4]   float32  (w,x,y,z)
    """
    import numpy as np

    N = means.shape[0]
    n_rest = 0 if shN is None else shN.shape[1] * shN.shape[2]

    props = ["x", "y", "z", "nx", "ny", "nz",
             "f_dc_0", "f_dc_1", "f_dc_2"]
    props += [f"f_rest_{i}" for i in range(n_rest)]
    props += ["opacity", "scale_0", "scale_1", "scale_2",
              "rot_0", "rot_1", "rot_2", "rot_3"]

    header = "\n".join([
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {N}",
        *[f"property float {p}" for p in props],
        "end_header",
        "",
    ]).encode("ascii")

    dtype = np.dtype([(p, "<f4") for p in props])
    rec = np.zeros(N, dtype=dtype)
    rec["x"], rec["y"], rec["z"] = means[:, 0], means[:, 1], means[:, 2]
    # normals unused by 3DGS viewers, left zero
    rec["f_dc_0"], rec["f_dc_1"], rec["f_dc_2"] = sh0[:, 0], sh0[:, 1], sh0[:, 2]
    if n_rest > 0:
        # channel-major flatten: all bands for R, then G, then B
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
def export_highlighted_ply(
    scene: str = "truck_test",
    prompt: str = "red truck",
    threshold: float = 0.6,
    max_gaussians: int | None = None,
    highlight_color=(0.0, 1.0, 0.0),
    boost_opacity: bool = True,
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

    # diagnostic: was the checkpoint actually larger than max_gaussians?
    # (if so, subsampling occurred and cross-run index correspondence
    # is only valid because it's all happening inside this one process/call)
    raw_ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    raw_n = raw_ck["splats"]["means"].shape[0]
    print(f"[export] checkpoint has {raw_n:,} raw gaussians "
          f"(cap={max_gaussians})"
          + ("  -- NOTE: subsampled, see caveat below" if max_gaussians and raw_n > max_gaussians else ""))

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
    idx = res["indices"].numpy()
    print(f"[export] highlighting {len(idx):,} gaussians for \"{prompt}\"")

    # -- invert activations back to raw ply-storage form --
    means = g.means.cpu().numpy().astype(np.float32)
    opac = g.opacities.reshape(-1).cpu().numpy().astype(np.float32)
    opac = np.clip(opac, 1e-6, 1 - 1e-6)
    opacity_raw = np.log(opac / (1 - opac))                 # inverse sigmoid
    scale_raw = np.log(g.scales.cpu().numpy().astype(np.float32))  # inverse exp
    quats = g.quats.cpu().numpy().astype(np.float32)

    sh = g.sh.cpu().numpy().astype(np.float32)  # [N, K, 3], raw (no activation needed)
    sh0 = sh[:, 0, :].copy()
    shN = sh[:, 1:, :].copy() if sh.shape[1] > 1 else None

    # tint the identified gaussians
    dc_target = _rgb_to_dc(highlight_color)
    sh0[idx] = dc_target
    if shN is not None:
        shN[idx] = 0.0  # flat color, no view-dependent effects on the highlight
    if boost_opacity:
        boosted = np.log(0.98 / 0.02)
        opacity_raw[idx] = boosted

    out_path = work / f"highlight_{prompt.replace(' ', '_')}.ply"
    _write_ply(out_path, means, sh0, shN, opacity_raw, scale_raw, quats)
    volume.commit()
    print(f"[export] wrote {out_path}")
    print(f"[export] centroid: {res['centroid'].tolist()}")
    return str(out_path)


@app.local_entrypoint()
def export(scene: str = "truck_test", prompt: str = "red truck", threshold: float = 0.6):
    path = export_highlighted_ply.remote(scene=scene, prompt=prompt, threshold=threshold)
    if path:
        fname = path.split("/")[-1]
        print(f"\nDone -> {path}")
        print(f"Pull with:\n  modal volume get gaussian-outputs "
              f"{scene}/vgrep3d/{fname} ./{fname}")
        print("Then open https://superspl.at/editor and drag the .ply in.")
