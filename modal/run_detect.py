"""Open-vocabulary detection with boxes + labels + confidence."""

from __future__ import annotations
from pathlib import Path
import modal

app = modal.App("vgrep3d-detect")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("git", "ffmpeg", "libgl1-mesa-glx", "libglib2.0-0")
    .pip_install(
        "transformers>=4.40.0",
        "timm",
        "Pillow",
        "opencv-python-headless",
        "numpy",
        "imageio[ffmpeg]",
        "tqdm",
        "accelerate",
    )
)

volume = modal.Volume.from_name("gaussian-outputs", create_if_missing=False)


@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 30,
    volumes={"/outputs": volume},
    memory=32768,
)
def detect(scene: str = "playroom_test", prompt: str = "dell monitor", max_images: int = 30):
    import torch
    import numpy as np
    from PIL import Image, ImageDraw
    from pathlib import Path
    import imageio
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

    device = "cuda"
    model_id = "IDEA-Research/grounding-dino-tiny"

    print(f"Loading {model_id} ...")
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

    img_dir = Path("/outputs") / scene / "colmap" / "input"
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))[:max_images]
    print(f"Using {len(img_paths)} images")

    out_dir = Path("/outputs") / scene / "vgrep3d" / "detect"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    text = prompt.lower().strip()

    for i, p in enumerate(img_paths):
        image = Image.open(p).convert("RGB")
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.15,
            text_threshold=0.15,
            target_sizes=[image.size[::-1]],
        )[0]

        draw = ImageDraw.Draw(image)
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"]

        for box, score, label in zip(boxes, scores, labels):
            x0, y0, x1, y1 = box
            draw.rectangle([x0, y0, x1, y1], outline="lime", width=4)
            caption = f"{label} {score:.2f}"
            draw.rectangle([x0, y0 - 22, x0 + len(caption) * 9, y0], fill="lime")
            draw.text((x0 + 3, y0 - 20), caption, fill="black")

        out_path = out_dir / f"det_{i:03d}.jpg"
        image.save(out_path, quality=90)
        frames.append(np.array(image))

        if i % 5 == 0:
            print(f"  processed {i}/{len(img_paths)}  – found {len(boxes)} boxes")

    video_path = out_dir / f"detect_{prompt.replace(' ', '_')}.mp4"
    imageio.mimwrite(str(video_path), frames, fps=6, quality=8)
    print(f"✓ Wrote {video_path}")
    volume.commit()
    return str(video_path)


@app.local_entrypoint()
def main(scene: str = "playroom_test", prompt: str = "dell monitor", max_images: int = 30):
    path = detect.remote(scene=scene, prompt=prompt, max_images=max_images)
    print(f"\nDone → {path}")
    print(f"Pull with:\n  modal volume get gaussian-outputs {scene}/vgrep3d/detect/ ./detect_out/")
