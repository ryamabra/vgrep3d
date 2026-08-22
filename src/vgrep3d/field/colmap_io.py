"""Minimal COLMAP binary reader (cameras.bin + images.bin)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ImagePose:
    id: int
    name: str
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    R: np.ndarray
    t: np.ndarray
    world_to_camera: np.ndarray


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)


def read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    return struct.unpack("<" + fmt, data)


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id = read_next_bytes(f, 4, "i")[0]
            model_id = read_next_bytes(f, 4, "i")[0]
            width = read_next_bytes(f, 8, "Q")[0]
            height = read_next_bytes(f, 8, "Q")[0]
            num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 8}.get(model_id, 4)
            params = np.array(read_next_bytes(f, 8 * num_params, "d" * num_params))
            model_name = {0: "SIMPLE_PINHOLE", 1: "PINHOLE", 4: "OPENCV"}.get(model_id, "PINHOLE")
            cameras[camera_id] = Camera(id=camera_id, model=model_name, width=width, height=height, params=params)
    return cameras


def read_images_binary(path: Path) -> Dict[int, ImagePose]:
    images = {}
    with open(path, "rb") as f:
        num_images = read_next_bytes(f, 8, "Q")[0]
        for _ in range(num_images):
            image_id = read_next_bytes(f, 4, "i")[0]
            qvec = np.array(read_next_bytes(f, 32, "dddd"))
            tvec = np.array(read_next_bytes(f, 24, "ddd"))
            camera_id = read_next_bytes(f, 4, "i")[0]
            name = ""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name += c.decode("utf-8")
            num_points2D = read_next_bytes(f, 8, "Q")[0]
            f.read(24 * num_points2D)

            R = qvec_to_rotmat(qvec)
            t = tvec
            w2c = np.eye(4)
            w2c[:3, :3] = R
            w2c[:3, 3] = t

            images[image_id] = ImagePose(
                id=image_id, name=name, qvec=qvec, tvec=tvec, camera_id=camera_id,
                R=R, t=t, world_to_camera=w2c
            )
    return images


def load_colmap_scene(sparse_dir: str | Path) -> Tuple[Dict[int, Camera], Dict[int, ImagePose]]:
    sparse_dir = Path(sparse_dir)
    cameras = read_cameras_binary(sparse_dir / "cameras.bin")
    images = read_images_binary(sparse_dir / "images.bin")
    images = dict(sorted(images.items(), key=lambda kv: kv[1].name))
    return cameras, images


def get_intrinsics(camera: Camera) -> np.ndarray:
    p = camera.params
    if camera.model == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy = p[1], p[2]
    else:
        fx, fy, cx, cy = p[0], p[1], p[2], p[3]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
