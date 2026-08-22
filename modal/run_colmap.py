"""Run COLMAP structure-from-motion on a directory of images already sitting
on the volume, producing sparse/0/{cameras.bin, images.bin, points3D.bin} --
the exact binary format colmap_io.py already reads, so nothing downstream
needs to change once this succeeds.

Uses SEQUENTIAL matching (not exhaustive), since the driving frames are an
ordered video sequence -- this is both much faster than exhaustive pairwise
matching (293 images exhaustively is ~42.8K pairs; sequential only compares
each frame to its ~N neighbors) and gives COLMAP the ordering as a strong
prior, which matters more the closer the capture is to a degenerate
near-straight-line trajectory (a known weak case for unordered SfM).

Usage (from repo root):
    modal run modal/run_colmap.py --scene driving_test

Pull the result:
    modal volume get gaussian-outputs driving_test/colmap/sparse/ ./driving_colmap_out/
"""

from __future__ import annotations

import modal

app = modal.App("vgrep3d-colmap")

image = (
    modal.Image.from_registry("colmap/colmap:latest", add_python="3.11")
    .pip_install("numpy")
)

volume = modal.Volume.from_name("gaussian-outputs", create_if_missing=False)
OUT = "/outputs"


@app.function(image=image, gpu="A10G", timeout=2 * 60 * 60,
              volumes={OUT: volume}, memory=32768)
def run_colmap(
    scene: str = "driving_test",
    sequential_overlap: int = 15,
    camera_model: str = "OPENCV",
):
    import subprocess
    import shutil
    from pathlib import Path

    root = Path(OUT) / scene
    image_dir = root / "colmap" / "input"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"No images found at {image_dir}")
    n_images = len(list(image_dir.glob("*.jpg"))) + len(list(image_dir.glob("*.png")))
    print(f"[colmap] {n_images} images in {image_dir}")

    work = Path("/tmp/colmap_work")
    work.mkdir(parents=True, exist_ok=True)
    db_path = work / "database.db"
    sparse_dir = work / "sparse"
    sparse_dir.mkdir(exist_ok=True)

    def run(cmd, label):
        print(f"[colmap] == {label} ==")
        print(f"[colmap] $ {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        # COLMAP logs progress to stdout; tail it so long steps aren't silent
        tail = "\n".join(result.stdout.strip().splitlines()[-40:])
        print(tail)
        if result.returncode != 0:
            print("[colmap] STDERR:\n" + result.stderr[-4000:])
            raise RuntimeError(f"{label} failed with exit code {result.returncode}")
        return result

    # 1. feature extraction (GPU SIFT)
    run([
        "colmap", "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.camera_model", camera_model,
        "--ImageReader.single_camera", "1",   # one physical camera captured all frames
        "--FeatureExtraction.use_gpu", "1",
    ], "feature_extractor")

    # 2. sequential matching -- video-ordered frames, not unordered photos
    run([
        "colmap", "sequential_matcher",
        "--database_path", str(db_path),
        "--SequentialMatching.overlap", str(sequential_overlap),
        "--FeatureMatching.use_gpu", "1",
    ], "sequential_matcher")

    # 3. sparse reconstruction (incremental SfM -> camera poses + sparse points)
    run([
        "colmap", "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
    ], "mapper")

    # find the largest reconstructed model (mapper can produce sparse/0, sparse/1, ...
    # if the scene fragments into disconnected components -- common with weak
    # parallax / degenerate trajectories)
    models = sorted(sparse_dir.glob("*"), key=lambda p: p.name)
    if not models:
        raise RuntimeError("mapper produced no reconstruction at all -- "
                            "SfM failed to register any coherent camera set")
    print(f"[colmap] mapper produced {len(models)} model(s): "
          f"{[m.name for m in models]}")

    best_model, best_n = None, -1
    for m in models:
        analysis = subprocess.run(
            ["colmap", "model_analyzer", "--path", str(m)],
            capture_output=True, text=True,
        )
        log = analysis.stdout + analysis.stderr
        print(f"[colmap] -- model {m.name} --\n{log}")
        n_reg = 0
        for line in log.splitlines():
            if "Registered images" in line:
                try:
                    n_reg = int(line.split(":")[-1].strip().split()[0])
                except (ValueError, IndexError):
                    pass
        if n_reg > best_n:
            best_model, best_n = m, n_reg

    print(f"[colmap] best model: {best_model.name} with {best_n}/{n_images} "
          f"images registered ({100 * best_n / max(n_images, 1):.1f}%)")
    if best_n < n_images * 0.5:
        print("[colmap] WARNING: fewer than half the frames registered. "
              "This is consistent with weak parallax / near-degenerate "
              "forward-only trajectory -- expect gaps and floaters in the "
              "trained splat, especially off the captured camera path.")

    dest = root / "colmap" / "sparse" / "0"
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(best_model, dest)
    volume.commit()
    print(f"[colmap] wrote {dest}")
    return {"n_images": n_images, "n_registered": best_n}


@app.function(image=image, timeout=5 * 60)
def check_colmap():
    """Diagnostic: print the installed COLMAP version and the real flag
    names feature_extractor accepts, instead of guessing across versions."""
    import subprocess
    ver = subprocess.run(["colmap", "-h"], capture_output=True, text=True)
    print("[colmap] colmap -h:\n" + (ver.stdout + ver.stderr)[:2000])
    help_out = subprocess.run(["colmap", "feature_extractor", "-h"],
                               capture_output=True, text=True)
    text = help_out.stdout + help_out.stderr
    print("[colmap] feature_extractor -h (full):\n" + text)
    gpu_lines = [l for l in text.splitlines() if "gpu" in l.lower()]
    print("[colmap] GPU-related flags found:\n" + "\n".join(gpu_lines))

    match_help = subprocess.run(["colmap", "sequential_matcher", "-h"],
                                 capture_output=True, text=True)
    match_text = match_help.stdout + match_help.stderr
    match_gpu_lines = [l for l in match_text.splitlines() if "gpu" in l.lower()]
    print("[colmap] sequential_matcher GPU-related flags found:\n" + "\n".join(match_gpu_lines))


@app.local_entrypoint()
def check():
    check_colmap.remote()


@app.local_entrypoint()
def main(scene: str = "driving_test", sequential_overlap: int = 15):
    result = run_colmap.remote(scene=scene, sequential_overlap=sequential_overlap)
    print(f"\nDone: {result['n_registered']}/{result['n_images']} images registered")
    print(f"Pull with:\n  modal volume get gaussian-outputs "
          f"{scene}/colmap/sparse/ ./{scene}_colmap_out/")
