import modal
from pathlib import Path

app = modal.App("vgrep3d-heatmap")
vol = modal.Volume.from_name("gaussian-outputs")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("gsplat==1.5.3", "open-clip-torch", "transformers", "opencv-python-headless", "Pillow")
    .add_local_dir(
        Path.home() / "Downloads/vgrep3d_extracted/src/vgrep3d",
        remote_path="/root/vgrep3d"
    )
)

@app.function(image=image, volumes={"/outputs": vol}, gpu="A10G", timeout=600)
def render_heatmap(scene: str = "driving_test", prompt: str = "red car"):
    import sys; sys.path.insert(0, "/root")
    import torch, torch.nn as nn, torch.nn.functional as F
    import numpy as np, cv2
    from pathlib import Path
    from gsplat import rasterization

    device = "cuda"
    work       = Path(f"/outputs/{scene}/vgrep3d")
    ckpt_path  = next(Path(f"/outputs/{scene}/gsplat/ckpts").glob("*.pt"))
    sparse_dir = Path(f"/outputs/{scene}/colmap/sparse/0")
    image_dir  = Path(f"/outputs/{scene}/colmap/input")

    ck = torch.load(ckpt_path, map_location=device)
    splats   = ck["splats"]
    g_means  = splats["means"].to(device)
    g_quats  = splats["quats"].to(device)
    g_scales = splats["scales"].to(device)
    g_opacs  = splats["opacities"].to(device)
    g_sh0    = splats["sh0"].to(device)
    if g_sh0.ndim == 3: g_sh0 = g_sh0[:, 0, :]
    N = len(g_means)
    print(f"Loaded {N} gaussians")

    # Build decoder and decode latents
    ae_ck = torch.load(work / "autoencoder.pt", map_location=device)
    in_dim, latent_dim, hidden = ae_ck["in_dim"], ae_ck["latent_dim"], 256
    state = ae_ck["state_dict"]
    decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden//2), nn.GELU(),
        nn.Linear(hidden//2, hidden), nn.GELU(),
        nn.Linear(hidden, in_dim),
    ).to(device)
    decoder.load_state_dict({k.replace("decoder.",""):v for k,v in state.items() if k.startswith("decoder.")})
    decoder.eval()

    latents_ck = torch.load(work / "latents.pt", map_location=device)
    latents = (latents_ck["latents"] if isinstance(latents_ck, dict) else latents_ck).float()

    # Decode in batches to save memory
    feats_list = []
    with torch.no_grad():
        for i in range(0, N, 8192):
            feats_list.append(F.normalize(decoder(latents[i:i+8192]), dim=-1).cpu())
    gauss_feats = torch.cat(feats_list, dim=0)  # keep on CPU
    del latents, decoder; torch.cuda.empty_cache()

    # Encode text (CPU-side then move)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli")
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
    with torch.no_grad():
        raw = clip_model.encode_text(tokenizer([prompt]).to(device))
        t = (raw.float() if torch.is_tensor(raw) else raw.text_embeds.float()).cpu()
        t = F.normalize(t, dim=-1)
    del clip_model; torch.cuda.empty_cache()

    # Relevance on CPU
    sim = (gauss_feats @ t.T).squeeze(-1)
    lo, hi = torch.quantile(sim, 0.01), torch.quantile(sim, 0.99)
    rel = ((sim - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    print(f"Relevance: {rel.min():.3f} – {rel.max():.3f}")
    del gauss_feats, t; torch.cuda.empty_cache()

    # Heat colors on GPU
    colors_rgb = (g_sh0 * 0.28209479177387814 + 0.5).clamp(0, 1)
    r = rel.to(device)[:, None]
    # Use jet colormap: low=blue, mid=green, high=red
    heat = torch.zeros_like(colors_rgb)
    heat[:, 0] = rel.to(device).clamp(0,1)  # R
    heat[:, 1] = (1 - (rel.to(device) - 0.5).abs() * 2).clamp(0,1)  # G
    heat[:, 2] = (1 - rel.to(device)).clamp(0,1)  # B
    # Only show heat where relevance is high, keep original elsewhere
    heat_colors = torch.where(r > 0.6, heat, colors_rgb)
    del r, heat, colors_rgb; torch.cuda.empty_cache()

    from vgrep3d.field.colmap_io import read_cameras_binary, read_images_binary, get_intrinsics
    colmap_cams   = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    colmap_images = read_images_binary(str(sparse_dir / "images.bin"))

    out_dir = Path(f"/outputs/{scene}/heatmap_{prompt.replace(' ','_')}")
    out_dir.mkdir(exist_ok=True)
    imgs = sorted(colmap_images.values(), key=lambda x: x.name)[:30]

    for im in imgs:
        cam  = colmap_cams[im.camera_id]
        w, h = int(cam.width) // 2, int(cam.height) // 2  # half res to save VRAM
        K    = torch.tensor(get_intrinsics(cam), dtype=torch.float32, device=device)
        K[0] /= 2; K[1] /= 2  # scale intrinsics for half res
        w2c  = torch.tensor(im.world_to_camera, dtype=torch.float32, device=device)
        with torch.no_grad():
            rgb, _, _ = rasterization(
                means=g_means, quats=g_quats, scales=g_scales,
                opacities=g_opacs.reshape(-1), colors=heat_colors,
                viewmats=w2c[None], Ks=K[None], width=w, height=h,
            )
        img_bgr = cv2.cvtColor((rgb[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        orig = image_dir / im.name
        if orig.exists():
            o = cv2.imread(str(orig))
            if o is not None:
                img_bgr = cv2.addWeighted(cv2.resize(o,(w,h)), 0.6, img_bgr, 0.4, 0)
        cv2.imwrite(str(out_dir / f"{Path(im.name).stem}.jpg"), img_bgr)

    frames = sorted(out_dir.glob("*.jpg"))
    if frames:
        s = cv2.imread(str(frames[0])); fh, fw = s.shape[:2]
        vid = Path(f"/outputs/{scene}/heatmap_{prompt.replace(' ','_')}.mp4")
        vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (fw, fh))
        for f in frames: vw.write(cv2.imread(str(f)))
        vw.release()
        print(f"Video -> {vid}")
    vol.commit()
    print(f"Done. {len(frames)} frames.")

@app.local_entrypoint()
def main(scene: str = "driving_test", prompt: str = "red car"):
    render_heatmap.remote(scene=scene, prompt=prompt)
