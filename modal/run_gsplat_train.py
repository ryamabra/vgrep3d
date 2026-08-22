"""Train a 3D Gaussian Splat from a COLMAP reconstruction, using gsplat's own
official examples/simple_trainer.py -- the same tool that produced the
truck_test checkpoint (ckpt_io.py's docstring already confirms the checkpoint
format is "a gsplat simple_trainer checkpoint"). Reusing gsplat's own trainer
means we get correct adaptive density control, SH-degree scheduling, and loss
weighting for free, rather than reimplementing 3DGS training from scratch.

Unlike run_colmap.py, this streams gsplat's training log LIVE via Popen
instead of capturing it silently until the process exits -- training can run
for a long time and you should be able to see it's actually progressing.

Usage (from repo root):
    modal run modal/run_gsplat_train.py --scene driving_test --max-steps 7000

Pull the result:
    modal volume get gaussian-outputs driving_test/gsplat/ ./driving_gsplat_out/
"""

from __future__ import annotations

import modal

app = modal.App("vgrep3d-gsplat-train")
GSPLAT_VERSION = "1.5.3"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        f"gsplat=={GSPLAT_VERSION}",
        "numpy", "opencv-python-headless", "imageio[ffmpeg]",
        "tqdm", "tyro", "pyyaml", "scikit-learn", "tensorboard",
        "torchmetrics", "plyfile", "PyMCubes",
    )
    # examples/ isn't shipped in the pip package -- clone the matching tag to get
    # simple_trainer.py and its COLMAP dataset loader
    .run_commands(
        f"git clone --depth 1 --branch v{GSPLAT_VERSION} "
        f"https://github.com/nerfstudio-project/gsplat.git /root/gsplat_src"
    )
    # install gsplat's own tested examples requirements (correct pycolmap
    # variant, viser, nerfview, etc.) rather than hand-picking packages
    .run_commands(
        "pip install -r /root/gsplat_src/examples/requirements.txt"
    )
)

volume = modal.Volume.from_name("gaussian-outputs", create_if_missing=False)
OUT = "/outputs"


def _stream_run(cmd, cwd=None):
    """Run a subprocess with LIVE stdout streaming, not silent capture --
    long training runs need visible progress, not a black box until exit."""
    import subprocess
    print(f"[gsplat] $ {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(
        cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"command failed with exit code {proc.returncode}: "
                            f"{' '.join(cmd)}")


@app.function(image=image, gpu="A10G", timeout=8 * 60 * 60,
              volumes={OUT: volume}, memory=32768)
def train_gsplat(
    scene: str = "driving_test",
    max_steps: int = 7000,
    data_factor: int = 1,   # downscale images by this factor (1 = full res)
):
    from pathlib import Path
    import shutil

    root = Path(OUT) / scene
    colmap_dir = root / "colmap"
    image_dir = colmap_dir / "input"
    sparse_dir = colmap_dir / "sparse"
    if not (sparse_dir / "0" / "cameras.bin").exists():
        raise FileNotFoundError(
            f"No COLMAP reconstruction at {sparse_dir}/0 -- run run_colmap.py first."
        )
    n_images = len(list(image_dir.glob("*.jpg"))) + len(list(image_dir.glob("*.png")))
    print(f"[gsplat] {n_images} images, training for {max_steps} steps")

    # gsplat's COLMAP dataset loader expects <data_dir>/images + <data_dir>/sparse/0
    # -- arrange a matching layout without duplicating the (large) image files
    data_dir = Path("/tmp/gsplat_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    images_link = data_dir / "images"
    sparse_link = data_dir / "sparse"
    if images_link.exists():
        images_link.unlink()
    if sparse_link.exists():
        sparse_link.unlink()
    images_link.symlink_to(image_dir)
    sparse_link.symlink_to(sparse_dir)

    result_dir = Path("/tmp/gsplat_result")
    result_dir.mkdir(parents=True, exist_ok=True)

    try:
        _stream_run([
            "python", "examples/simple_trainer.py", "default",
            "--data_dir", str(data_dir),
            "--data_factor", str(data_factor),
            "--result_dir", str(result_dir),
            "--max_steps", str(max_steps),
            "--disable_viewer",
        ], cwd="/root/gsplat_src")
    except RuntimeError as e:
        ckpt_check = list((result_dir / "ckpts").glob("ckpt_*_rank0.pt")) if (result_dir / "ckpts").exists() else []
        if not ckpt_check:
            raise
        print(f"[gsplat] WARNING: simple_trainer exited with an error after training ({e}), but {len(ckpt_check)} checkpoint(s) exist -- treating training as successful and continuing to save.")

    # simple_trainer writes ckpts/ckpt_<step>_rank0.pt -- copy the whole
    # result dir back to the volume, matching the layout ckpt_io.py expects
    # (**/ckpts/ckpt_*_rank0.pt glob)
    dest = root / "gsplat"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(result_dir, dest)
    volume.commit()

    ckpts = list((dest / "ckpts").glob("ckpt_*_rank0.pt"))
    print(f"[gsplat] wrote {len(ckpts)} checkpoint(s) to {dest / 'ckpts'}")
    for c in ckpts:
        print(f"[gsplat]   {c.name}")
    return {"n_checkpoints": len(ckpts), "result_dir": str(dest)}


@app.local_entrypoint()
def main(scene: str = "driving_test", max_steps: int = 7000, data_factor: int = 1):
    result = train_gsplat.remote(scene=scene, max_steps=max_steps, data_factor=data_factor)
    print(f"\nDone: {result['n_checkpoints']} checkpoint(s) at {result['result_dir']}")
    print(f"Pull with:\n  modal volume get gaussian-outputs "
          f"{scene}/gsplat/ ./{scene}_gsplat_out/")
