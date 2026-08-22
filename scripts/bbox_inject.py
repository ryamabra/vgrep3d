import modal

app = modal.App("bbox-final")
vol = modal.Volume.from_name("gaussian-outputs")

@app.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "numpy", "plyfile", "open-clip-torch", "Pillow", "transformers"
    ),
    volumes={"/outputs": vol},
    gpu="A10G",
    timeout=600,
)
def make_final_bbox():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import numpy as np
    from pathlib import Path
    from plyfile import PlyData, PlyElement

    device = "cuda"
    scene = "driving_test"
    work = Path(f"/outputs/{scene}/vgrep3d")
    ckpt_path = f"/outputs/{scene}/gsplat/ckpts/ckpt_6999_rank0.pt"

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

    # Load latents
    latents_ck = torch.load(work / "latents.pt", map_location=device)
    latents = (latents_ck["latents"] if isinstance(latents_ck, dict) else latents_ck).float()
    print(f"Latents: {latents.shape}")

    # Load AE and reconstruct EXACTLY as FeatureAutoencoder does
    ae_ck = torch.load(work / "autoencoder.pt", map_location=device)
    in_dim     = ae_ck["in_dim"]
    latent_dim = ae_ck["latent_dim"]
    hidden     = 256
    state      = ae_ck["state_dict"]
    print(f"AE: in_dim={in_dim}, latent_dim={latent_dim}, hidden={hidden}")

    # Decoder: latent_dim -> hidden//2 -> hidden -> in_dim  (GELU activations)
    decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden // 2),
        nn.GELU(),
        nn.Linear(hidden // 2, hidden),
        nn.GELU(),
        nn.Linear(hidden, in_dim),
    ).to(device)
    dec_state = {k.replace("decoder.", ""): v for k, v in state.items() if k.startswith("decoder.")}
    decoder.load_state_dict(dec_state)
    decoder.eval()

    # Decode and L2-normalize (matches ae.decode exactly)
    batch_size = 8192
    feats_list = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            z = latents[i:i+batch_size]
            f = F.normalize(decoder(z), dim=-1)
            feats_list.append(f)
    gauss_feats = torch.cat(feats_list, dim=0)  # (N, in_dim)
    print(f"Gaussian features: {gauss_feats.shape}")

    # Load SigLIP text encoder
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")

    def encode_text(prompt):
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            raw = model.encode_text(tokens)
        # Handle ModelOutput
        if torch.is_tensor(raw):
            t = raw
        else:
            for attr in ("text_embeds", "pooler_output", "last_hidden_state"):
                val = getattr(raw, attr, None)
                if val is not None:
                    t = val[:, 0, :] if val.dim() == 3 else val
                    break
        return F.normalize(t.float(), dim=-1)  # (1, D)

    def relevance(prompt):
        t = encode_text(prompt)  # (1, D)
        sim = (gauss_feats @ t.T).squeeze(-1)  # (N,)
        lo = torch.quantile(sim, 0.01)
        hi = torch.quantile(sim, 0.99)
        rel = ((sim - lo) / (hi - lo + 1e-8)).clamp(0.0, 1.0)
        return rel

    def keep_dense_core(pts, n_mad=3.0, iters=2):
        keep = torch.ones(len(pts), dtype=torch.bool, device=pts.device)
        for _ in range(iters):
            p = pts[keep]
            if len(p) < 8:
                break
            med = p.median(dim=0).values
            mad = (p - med).abs().median(dim=0).values + 1e-9
            z = (pts - med).abs() / (1.4826 * mad)
            new_keep = (z < n_mad).all(dim=1) & keep
            keep = new_keep
        return keep

    def robust_aabb(pts):
        mn = torch.quantile(pts, 0.02, dim=0)
        mx = torch.quantile(pts, 0.98, dim=0)
        return mn, mx

    def color_to_sh(r, g, b):
        C0 = 0.28209479177387814
        return [(r - 0.5) / C0, (g - 0.5) / C0, (b - 0.5) / C0]

    def make_edge_gaussians(mn, mx, rgb, n_per_edge=300, scale=0.012):
        mn, mx = np.array(mn), np.array(mx)
        corners = np.array([[mn[0],mn[1],mn[2]],[mx[0],mn[1],mn[2]],
                             [mn[0],mx[1],mn[2]],[mx[0],mx[1],mn[2]],
                             [mn[0],mn[1],mx[2]],[mx[0],mn[1],mx[2]],
                             [mn[0],mx[1],mx[2]],[mx[0],mx[1],mx[2]]])
        edges = [(0,1),(2,3),(4,5),(6,7),(0,2),(1,3),(4,6),(5,7),(0,4),(1,5),(2,6),(3,7)]
        pts = []
        for a, b in edges:
            t = np.linspace(0, 1, n_per_edge)
            pts.append(corners[a] + t[:,None] * (corners[b] - corners[a]))
        pts = np.vstack(pts).astype(np.float32)
        n = len(pts)
        sh = color_to_sh(*rgb)
        sc = np.full((n, 3), np.log(scale), dtype=np.float32)
        qt = np.zeros((n, 4), dtype=np.float32); qt[:, 0] = 1.0
        op = np.full(n, 8.0, dtype=np.float32)
        col = np.tile(sh, (n, 1)).astype(np.float32)
        return pts, sc, qt, op, col

    queries = [
        ("red car",     [1.0, 0.15, 0.15]),
        ("white car",   [0.3,  0.5,  1.0]),
        ("stop sign",   [0.1,  1.0,  0.1]),
        ("yellow sign", [1.0,  0.9,  0.0]),
        ("motorcycle",  [0.8,  0.1,  0.9]),
    ]

    threshold = 0.6
    min_hits  = 16

    all_means  = [means]
    all_scales = [scales]
    all_quats  = [quats]
    all_opacs  = [opacs]
    all_sh0    = [sh0]

    means_t = torch.from_numpy(means).to(device)

    for prompt, rgb in queries:
        rel = relevance(prompt)
        hit_idx = torch.where(rel >= threshold)[0]
        if len(hit_idx) < min_hits:
            k = max(min_hits, int(0.01 * len(rel)))
            hit_idx = torch.topk(rel, k).indices
        pts = means_t[hit_idx]
        core = keep_dense_core(pts)
        core_idx = hit_idx[core]
        if len(core_idx) < 8:
            core_idx = hit_idx
        pts_core = means_t[core_idx]
        mn, mx = robust_aabb(pts_core)
        mn = mn.cpu().numpy()
        mx = mx.cpu().numpy()
        print(f'"{prompt}": {len(core_idx)} core gaussians, bbox {mn.round(3)} -> {mx.round(3)}')

        edge_pts, sc, qt, op, col = make_edge_gaussians(mn, mx, rgb)
        all_means.append(edge_pts)
        all_scales.append(sc)
        all_quats.append(qt)
        all_opacs.append(op)
        all_sh0.append(col)

    means_out  = np.vstack(all_means)
    scales_out = np.vstack(all_scales)
    quats_out  = np.vstack(all_quats)
    opacs_out  = np.concatenate(all_opacs)
    sh0_out    = np.vstack(all_sh0)
    M = len(means_out)
    print(f"Total gaussians: {M}")

    vertex = np.zeros(M, dtype=[
        ('x','f4'),('y','f4'),('z','f4'),
        ('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),
        ('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4'),
        ('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
        ('opacity','f4'),
    ])
    vertex['x'], vertex['y'], vertex['z'] = means_out[:,0], means_out[:,1], means_out[:,2]
    vertex['scale_0']=scales_out[:,0]; vertex['scale_1']=scales_out[:,1]; vertex['scale_2']=scales_out[:,2]
    vertex['rot_0']=quats_out[:,0]; vertex['rot_1']=quats_out[:,1]
    vertex['rot_2']=quats_out[:,2]; vertex['rot_3']=quats_out[:,3]
    vertex['f_dc_0']=sh0_out[:,0]; vertex['f_dc_1']=sh0_out[:,1]; vertex['f_dc_2']=sh0_out[:,2]
    vertex['opacity']=opacs_out

    out = f"/outputs/{scene}/driving_test_bbox.ply"
    PlyData([PlyElement.describe(vertex, 'vertex')]).write(out)
    print(f"Saved -> {out}")
    vol.commit()

@app.local_entrypoint()
def main():
    make_final_bbox.remote()
