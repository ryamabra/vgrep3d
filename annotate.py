import modal
from pathlib import Path

app = modal.App("vgrep3d-annotate")
vol = modal.Volume.from_name("gaussian-outputs")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("opencv-python-headless", "numpy")
)

@app.function(image=image, volumes={"/outputs": vol}, timeout=120)
def annotate(scene: str = "driving_test", prompt: str = "red car"):
    import cv2, numpy as np
    from pathlib import Path

    frames_dir = Path(f"/outputs/{scene}/heatmap_{prompt.replace(' ','_')}")
    frames = sorted(frames_dir.glob("*.jpg"))
    print(f"Found {len(frames)} frames")

    colors_map = {
        "red car": (0,0,255), "white car": (255,100,0),
        "stop sign": (0,200,0), "yellow sign": (0,200,200),
        "motorcycle": (200,0,200),
    }
    bc = colors_map.get(prompt, (0,255,255))

    best_frame, best_score, best_box = None, -1, None

    for f in frames:
        img = cv2.imread(str(f))
        if img is None: continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Detect red/hot pixels (high value, low saturation in heat overlay = red)
        # Red in BGR: high B=low, G=low, R=high
        b, g, r = cv2.split(img)
        # Hot pixel mask: red channel much higher than blue
        hot = ((r.astype(int) - b.astype(int)) > 80) & (r > 150)
        hot_count = hot.sum()
        if hot_count > best_score:
            best_score = hot_count
            best_frame = img.copy()
            # Find bounding box of hot pixels
            ys, xs = np.where(hot)
            if len(xs) > 20:
                x1,x2 = int(np.percentile(xs,3)), int(np.percentile(xs,97))
                y1,y2 = int(np.percentile(ys,3)), int(np.percentile(ys,97))
                best_box = (x1,y1,x2,y2)

    if best_frame is not None and best_box is not None:
        x1,y1,x2,y2 = best_box
        cv2.rectangle(best_frame,(x1,y1),(x2,y2),bc,3)
        lbl = prompt.upper()
        (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,0.9,2)
        cv2.rectangle(best_frame,(x1,y1-th-12),(x1+tw+8,y1),bc,-1)
        cv2.putText(best_frame,lbl,(x1+4,y1-6),cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2)
        print(f"Box: {best_box}, hot pixels: {best_score}")
        out = Path(f"/outputs/{scene}/annotated_{prompt.replace(' ','_')}.jpg")
        cv2.imwrite(str(out), best_frame)
        print(f"Saved -> {out}")
        vol.commit()

@app.local_entrypoint()
def main(scene: str = "driving_test", prompt: str = "red car"):
    annotate.remote(scene=scene, prompt=prompt)
