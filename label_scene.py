import modal

app = modal.App("bbox-clustered")
vol = modal.Volume.from_name("gaussian-outputs")

@app.function(
    image=modal.Image.debian_slim().pip_install(
        "torch", "numpy", "plyfile", "open-clip-torch", "Pillow", "transformers", "scikit-learn"
    ),
    volumes={"/outputs": vol},
    gpu="A10G",
    timeout=600,
)
def make_final_bbox():
    import torch
    import numpy as np
    from pathlib import Path
    from plyfile import PlyData, PlyElement
    import torch.nn.functional as F
    import torch.nn as nn
    from sklearn.cluster import DBSCAN

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

    latents_ck = torch.load(work / "latents.pt", map_location=device)
    latents = (latents_ck["latents"] if isinstance(latents_ck, dict) else latents_ck).float()

    ae_ck = torch.load(work / "autoencoder.pt", map_location=device)
    state = ae_ck["state_dict"] if "state_dict" in ae_ck else ae_ck
    enc_keys = {k: v for k, v in state.items() if k.startswith("encoder.")}
    enc_indices = sorted(set(int(k.split(".")[1]) for k in enc_keys.keys()))
    enc_layers = []
    for idx, i in enumerate(enc_indices):
        w, b = f"encoder.{i}.weight", f"encoder.{i}.bias"
        if w in enc_keys:
            out_f, in_f = enc_keys[w].shape
            lin = nn.Linear(in_f, out_f)
            lin.weight.data = enc_keys[w]
            if b in enc_keys:
                lin.bias.data = enc_keys[b]
            enc_layers.append(lin)
            if idx != len(enc_indices) - 1:
                enc_layers.append(nn.ReLU())
    encoder = nn.Sequential(*enc_layers).to(device).eval()
    latents_norm = F.normalize(latents, dim=-1)

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms("ViT-SO400M-14-SigLIP-384", pretrained="webli")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-SO400M-14-SigLIP-384")

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

    all_means  = [means]
    all_scales = [scales]
    all_quats  = [quats]
    all_opacs  = [opacs]
    all_sh0    = [sh0]

    TOP_K = 8000

    for prompt, rgb in queries:
        tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            txt_feat = F.normalize(model.encode_text(tokens).float(), dim=-1)
            txt_latent = F.normalize(encoder(txt_feat), dim=-1)
        sims = (latents_norm @ txt_latent.T).squeeze(-1)
        topk_idx = torch.topk(sims, TOP_K).indices.cpu().numpy()
        top_pts = means[topk_idx]  # (TOP_K, 3)

        # DBSCAN to find the densest cluster = the actual object
        db = DBSCAN(eps=0.3, min_samples=20).fit(top_pts)
        labels = db.labels_
        unique, counts = np.unique(labels[labels >= 0], return_counts=True)
        if len(unique) == 0:
            # fallback: just use percentile bbox
            mn = np.percentile(top_pts, 5, axis=0)
            mx = np.percentile(top_pts, 95, axis=0)
            print(f'"{prompt}": no cluster found, fallback bbox')
        else:
            # Pick the largest cluster
            best_label = unique[np.argmax(counts)]
            cluster_pts = top_pts[labels == best_label]
            mn = cluster_pts.min(axis=0)
            mx = cluster_pts.max(axis=0)
            # Small padding
            pad = (mx - mn) * 0.05
            mn -= pad; mx += pad
            print(f'"{prompt}": cluster {len(cluster_pts)} pts, bbox {mn.round(2)} -> {mx.round(2)}')

        pts, sc, qt, op, col = make_edge_gaussians(mn, mx, rgb)
        all_means.append(pts)
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
    print(f"Total: {M}")

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
