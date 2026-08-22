import modal
from pathlib import Path

app = modal.App("vgrep3d-snapshot")
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
def snapshot(scene: str = "driving_test", prompt: str = "red car"):
    import sys; sys.path.insert(0, "/root")
    import torch, torch.nn as nn, torch.nn.functional as F
    import numpy as np, cv2
    from pathlib import Path
    from gsplat import rasterization

    device = "cuda"
    work       = Path(f"/outputs/{scene}/vgrep3d")
    ckpt_path  = next(Path(f"/outputs/{scene}/gsplat/ckpts").glob("*.pt"))
    sparse_dir = Path(f"/outputs/{scene}/colmap/sparse/0")

    ck = torch.load(ckpt_path, map_location=device)
    splats   = ck["splats"]
    g_means  = splats["means"].to(device)
    g_quats  = splats["quats"].to(device)
    g_scales = splats["scales"].to(device)
    g_opacs  = splats["opacities"].to(device)
    g_sh0    = splats["sh0"].to(device)
    if g_sh0.ndim == 3: g_sh0 = g_sh0[:, 0, :]
    N = len(g_means)

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
    feats_list = []
    with torch.no_grad():
        for i in range(0, N, 8192):
            feats_list.append(F.normalize(decoder(latents[i:i+8192]), dim=-1).cpu())
    gauss_feats = torch.cat(feats_list, dim=0)
    del latents, decoder; torch.cuda.empty_cache()

    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli")
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")
    with torch.no_grad():
        raw = clip_model.encode_text(tokenizer([prompt]).to(device))
        t = (raw.float() if torch.is_tensor(raw) else raw.text_embeds.float()).cpu()
        t = F.normalize(t, dim=-1)
    del clip_model; torch.cuda.empty_cache()

    sim = (gauss_feats @ t.T).squeeze(-1)
    lo, hi = torch.quantile(sim, 0.01), torch.quantile(sim, 0.99)
    rel = ((sim - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    del gauss_feats, t; torch.cuda.empty_cache()

    # Object centroid from top hits
    hit_mask = rel > 0.7
    if hit_mask.sum() < 50:
        hit_mask = rel > torch.quantile(rel, 0.95)
    hit_means_cpu = g_means[hit_mask.to(device)].cpu()
    centroid_cpu = hit_means_cpu.mean(dim=0)
    print(f"Centroid: {centroid_cpu.numpy().round(3)}, hits: {hit_mask.sum()}")

    from vgrep3d.field.colmap_io import read_cameras_binary, read_images_binary, get_intrinsics
    colmap_cams   = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    colmap_images = read_images_binary(str(sparse_dir / "images.bin"))

    # Pick camera where centroid has positive depth and projects near center
    # Sort images by name and try middle third — those frames face the road
    img_list = sorted(colmap_images.values(), key=lambda x: x.name)
    n = len(img_list)
    candidates = img_list[n//3: 2*n//3]  # middle third
    best_im, best_score = None, -1e9
    for im in candidates:
        w2c = np.array(im.world_to_camera)  # (4,4)
        c_h = np.array([*centroid_cpu.numpy(), 1.0])
        c_cam = w2c @ c_h  # (4,)
        depth = c_cam[2]
        if depth < 0.5:
            continue
        cam = colmap_cams[im.camera_id]
        K = get_intrinsics(cam)
        u = c_cam[0]/depth * K[0,0] + K[0,2]
        v = c_cam[1]/depth * K[1,1] + K[1,2]
        W, H = cam.width/2, cam.height/2
        # Score: centroid close to image center, not too deep
        score = -abs(u/2 - W/2)/(W/2) - abs(v/2 - H/2)/(H/2) - 0.01*depth
        if score > best_score:
            best_score = score
            best_im = im
    print(f"Best view: {best_im.name}")

    cam = colmap_cams[best_im.camera_id]
    W, H = int(cam.width)//2, int(cam.height)//2
    K_np = get_intrinsics(cam)
    K_np[0] /= 2; K_np[1] /= 2
    K = torch.tensor(K_np, dtype=torch.float32, device=device)
    w2c = torch.tensor(best_im.world_to_camera, dtype=torch.float32, device=device)

    colors_rgb = (g_sh0 * 0.28209479177387814 + 0.5).clamp(0, 1)
    r = rel.to(device)
    heat = torch.zeros_like(colors_rgb)
    heat[:,0]=r; heat[:,1]=(1-(r-0.5).abs()*2).clamp(0,1); heat[:,2]=(1-r)
    heat_colors = torch.where((r>0.65)[:,None], heat, colors_rgb)

    with torch.no_grad():
        rgb_blend, _, _ = rasterization(
            means=g_means, quats=g_quats, scales=g_scales,
            opacities=g_opacs.reshape(-1), colors=heat_colors,
            viewmats=w2c[None], Ks=K[None], width=W, height=H,
        )
    img = cv2.cvtColor((rgb_blend[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    # Project hit gaussians to 2D
    w2c_np = np.array(best_im.world_to_camera)
    pts_h = np.concatenate([hit_means_cpu.numpy(), np.ones((len(hit_means_cpu),1))], axis=1)
    pts_cam = (w2c_np @ pts_h.T).T  # (M,4)
    depth_vals = pts_cam[:, 2]
    in_front = depth_vals > 0.1
    pts_cam = pts_cam[in_front]
    print(f"Points in front of camera: {in_front.sum()}")

    if in_front.sum() > 10:
        u = pts_cam[:,0]/pts_cam[:,2] * K_np[0,0] + K_np[0,2]
        v = pts_cam[:,1]/pts_cam[:,2] * K_np[1,1] + K_np[1,2]
        u = np.clip(u, 0, W-1); v = np.clip(v, 0, H-1)
        x1,x2 = int(np.percentile(u,3)), int(np.percentile(u,97))
        y1,y2 = int(np.percentile(v,3)), int(np.percentile(v,97))
        print(f"2D box: ({x1},{y1})->({x2},{y2})")
        colors_map = {"red car":(0,0,255),"white car":(255,100,0),"stop sign":(0,200,0),"yellow sign":(0,200,200),"motorcycle":(200,0,200)}
        bc = colors_map.get(prompt,(0,255,255))
        cv2.rectangle(img,(x1,y1),(x2,y2),bc,3)
        lbl = prompt.upper()
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.8,2)
        cv2.rectangle(img,(x1,y1-th-10),(x1+tw+8,y1),bc,-1)
        cv2.putText(img,lbl,(x1+4,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    out = Path(f"/outputs/{scene}/snapshot_{prompt.replace(' ','_')}.jpg")
    cv2.imwrite(str(out), img)
    print(f"Saved -> {out}")
    vol.commit()

@app.local_entrypoint()
def main(scene: str = "driving_test", prompt: str = "red car"):
    snapshot.remote(scene=scene, prompt=prompt)
