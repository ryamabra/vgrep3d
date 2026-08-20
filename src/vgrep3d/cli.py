"""vgrep3d command-line interface.

Carries the vgrep idiom into 3D:

    vgrep3d index  <scene_dir>            build the feature field for a scene
    vgrep3d query  <scene_dir> "prompt"   locate the prompt in 3D

A <scene_dir> is expected to contain:
    images/            training RGB frames
    cameras.json       per-frame {viewmat, K, width, height}  (your splat's poses)
    point_cloud.ply    the trained splat

index writes into <scene_dir>/vgrep3d/:
    features/*.npy     2D SigLIP maps
    autoencoder.pt     per-scene AE
    latents.pt         trained per-Gaussian latents
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_cameras(scene: Path) -> list[dict]:
    raw = json.loads((scene / "cameras.json").read_text())
    cams = []
    for i, c in enumerate(raw):
        stem = c.get("image_stem", f"{i:05d}")
        cams.append(
            {
                "viewmat": torch.tensor(c["viewmat"], dtype=torch.float32),
                "K": torch.tensor(c["K"], dtype=torch.float32),
                "width": int(c["width"]),
                "height": int(c["height"]),
                "feature_path": str(scene / "vgrep3d" / "features" / f"{stem}.npy"),
            }
        )
    return cams


def cmd_index(args):
    from vgrep3d.field.autoencoder import FeatureAutoencoder, train_autoencoder
    from vgrep3d.field.feature_field import FeatureField
    from vgrep3d.field.train import save_field, train_feature_field
    from vgrep3d.preprocess.features import FeatureExtractorConfig, extract_dataset

    scene = Path(args.scene)
    work = scene / "vgrep3d"
    feat_dir = work / "features"
    work.mkdir(exist_ok=True)

    # 1. 2D feature maps
    if args.reextract or not feat_dir.exists():
        print("== extracting SigLIP 2 feature maps ==")
        extract_dataset(
            scene / "images",
            feat_dir,
            FeatureExtractorConfig(device=args.device, siglip_model=args.siglip),
        )

    # 2. per-scene autoencoder
    print("== training feature autoencoder ==")
    # infer feature dim from a saved map
    sample = next(feat_dir.glob("*.npy"))
    in_dim = np.load(sample).shape[-1]
    ae = FeatureAutoencoder(in_dim=in_dim, latent_dim=args.latent_dim)
    ae = train_autoencoder(ae, feat_dir, device=args.device)
    torch.save({"state_dict": ae.state_dict(), "in_dim": in_dim,
                "latent_dim": args.latent_dim}, work / "autoencoder.pt")

    # 3. feature field
    print("== distilling features into the 3D field ==")
    field = FeatureField.from_ply(
        str(scene / "point_cloud.ply"),
        latent_dim=args.latent_dim,
        device=args.device,
    )
    cams = _load_cameras(scene)
    field = train_feature_field(field, ae, cams, epochs=args.epochs, device=args.device)
    save_field(field, work / "latents.pt")
    print(f"done -> {work}")


def cmd_query(args):
    from vgrep3d.field.autoencoder import FeatureAutoencoder
    from vgrep3d.field.feature_field import FeatureField
    from vgrep3d.field.train import load_latents
    from vgrep3d.query.query import Query3D

    scene = Path(args.scene)
    work = scene / "vgrep3d"

    ae_ckpt = torch.load(work / "autoencoder.pt", map_location=args.device)
    ae = FeatureAutoencoder(in_dim=ae_ckpt["in_dim"], latent_dim=ae_ckpt["latent_dim"])
    ae.load_state_dict(ae_ckpt["state_dict"])

    field = FeatureField.from_ply(
        str(scene / "point_cloud.ply"),
        latent_dim=ae_ckpt["latent_dim"],
        device=args.device,
    )
    field = load_latents(field, work / "latents.pt", device=args.device)

    q = Query3D(field, ae, siglip_model=args.siglip, device=args.device)
    res = q.locate_3d(args.prompt, threshold=args.threshold)
    if not res["found"]:
        print(f'"{args.prompt}": no region above threshold {args.threshold}')
        return
    c = res["centroid"].tolist()
    lo, hi = res["aabb_min"].tolist(), res["aabb_max"].tolist()
    print(f'"{args.prompt}"')
    print(f"  centroid: ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f})")
    print(f"  aabb:     min={[round(x,3) for x in lo]}  max={[round(x,3) for x in hi]}")
    print(f"  support:  {res['num_gaussians']} gaussians")


def main(argv=None):
    p = argparse.ArgumentParser(prog="vgrep3d")
    p.add_argument("--device", default="cuda")
    p.add_argument("--siglip", default="google/siglip2-so400m-patch14-384")
    sub = p.add_subparsers(required=True)

    pi = sub.add_parser("index", help="build the feature field for a scene")
    pi.add_argument("scene")
    pi.add_argument("--latent-dim", type=int, default=3)
    pi.add_argument("--epochs", type=int, default=30)
    pi.add_argument("--reextract", action="store_true")
    pi.set_defaults(func=cmd_index)

    pq = sub.add_parser("query", help="locate a text prompt in 3D")
    pq.add_argument("scene")
    pq.add_argument("prompt")
    pq.add_argument("--threshold", type=float, default=0.6)
    pq.set_defaults(func=cmd_query)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
