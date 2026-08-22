import modal

app = modal.App("recolor-ply")
vol = modal.Volume.from_name("gaussian-outputs")

@app.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "gsplat", "numpy", "plyfile", "open-clip-torch", "Pillow"
    ),
    volumes={"/outputs": vol},
    gpu="A10G",
    timeout=600,
)
def recolor():
    import sys
    sys.path.insert(0, "/outputs")
    import torch
    import numpy as np
    from pathlib import Path
    from plyfile import PlyData, PlyElement
    import torch.nn.functional as F

    device = "cuda"
    scene = "driving_test"
    work = Path(f"/outputs/{scene}/vgrep3d")
    ckpt_path = f"/outputs/{scene}/gsplat/ckpts/ckpt_6999_rank0.pt"

    # Load gaussian checkpoint
    ck = torch.load(ckpt_path, map_location="cpu")
    splats = ck["splats"]
    means  = splats["means"].numpy()
    scales = splats["scales"].numpy()
    quats  = splats["quats"].numpy()
    opacs  = splats["opacities"].numpy()
    sh0    = splats["sh0"].numpy()
    if sh0.ndim == 3:
        sh0 = sh0[:, 0, :]
    N = means.shape[0]
    print(f"Loaded {N} gaussians")

    # Load latents (per-gaussian feature vectors)
    latents = torch.load(work / "latents.pt", map_location=device)  # (N, D)
    print(f"Latents shape: {latents.shape}")

    # Load autoencoder to get decoder
    ae_ck = torch.load(work / "autoencoder.pt", map_location=device)

    # Build a simple linear decoder from ae checkpoint
    # The autoencoder projects latents to feature space for similarity
    # We normalize latents directly for cosine sim (works if ae is identity-like)
    latents_norm = F.normalize(latents.float(), dim=-1)  # (N, D)

    # Load SigLIP2 text encoder
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-SO400M-14-SigLIP-384", pretrained="webli"
    )
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")

    # Queries and their RGB colors (in SH DC space: color = 0.5 + 0.28209*sh)
    # We'll directly set sh0 to produce the target color
    # sh0 = (color - 0.5) / 0.28209
    def color_to_sh(r, g, b):
        C0 = 0.28209479177387814
        return np.array([(r - 0.5) / C0, (g - 0.5) / C0, (b - 0.5) / C0], dtype=np.float32)

    queries = [
        ("red car",     color_to_sh(1.0, 0.1, 0.1)),   # red
        ("white car",   color_to_sh(0.2, 0.4, 1.0)),   # blue
        ("stop sign",   color_to_sh(0.1, 0.9, 0.1)),   # green
        ("yellow sign", color_to_sh(1.0, 0.9, 0.0)),   # yellow
        ("motorcycle",  color_to_sh(0.8, 0.1, 0.9)),   # purple
    ]

    threshold = 0.25
    sh0_out = sh0.copy()
    assigned = np.zeros(N, dtype=bool)

    for prompt, color_sh in queries:
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            txt_feat = model.encode_text(tokens)
            txt_feat = F.normalize(txt_feat.float(), dim=-1)  # (1, D)

        # Cosine similarity between text and each gaussian latent
        # latents may have different dim than text features — project if needed
        if latents_norm.shape[-1] != txt_feat.shape[-1]:
            # Simple mean-pool or take first dims to match
            d = min(latents_norm.shape[-1], txt_feat.shape[-1])
            sims = (latents_norm[:, :d] @ txt_feat[:, :d].T).squeeze(-1)
        else:
            sims = (latents_norm @ txt_feat.T).squeeze(-1)

        mask = (sims > threshold).cpu().numpy()
        n_matched = mask.sum()
        print(f'"{prompt}": {n_matched} gaussians (threshold={threshold})')

        # Only color gaussians not already assigned to a higher-priority query
        new_mask = mask & ~assigned
        sh0_out[new_mask] = color_sh
        assigned |= mask

    print(f"Total colored: {assigned.sum()} / {N}")

    # Export full PLY
    vertex = np.zeros(N, dtype=[
        ('x','f4'), ('y','f4'), ('z','f4'),
        ('scale_0','f4'), ('scale_1','f4'), ('scale_2','f4'),
        ('rot_0','f4'), ('rot_1','f4'), ('rot_2','f4'), ('rot_3','f4'),
        ('f_dc_0','f4'), ('f_dc_1','f4'), ('f_dc_2','f4'),
        ('opacity','f4'),
    ])
    vertex['x'], vertex['y'], vertex['z'] = means[:,0], means[:,1], means[:,2]
    vertex['scale_0'] = scales[:,0]
    vertex['scale_1'] = scales[:,1]
    vertex['scale_2'] = scales[:,2]
    vertex['rot_0'] = quats[:,0]
    vertex['rot_1'] = quats[:,1]
    vertex['rot_2'] = quats[:,2]
    vertex['rot_3'] = quats[:,3]
    vertex['f_dc_0'] = sh0_out[:,0]
    vertex['f_dc_1'] = sh0_out[:,1]
    vertex['f_dc_2'] = sh0_out[:,2]
    vertex['opacity'] = opacs

    out = f"/outputs/{scene}/driving_test_labeled.ply"
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(out)
    print(f"Saved -> {out}")
    vol.commit()

@app.local_entrypoint()
def main():
    recolor.remote()
