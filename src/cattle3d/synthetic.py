"""Synthetic MmCows barn scene used to test the reconstruction without the dataset.

The layout matches the real visual_data tree: four cameras, OpenCV projection
XML in centimeters, normalized bbox labels, and a behavior CSV. The pictured
object is a stylized cow head, so the loader, localizer, and 3DGS trainer can
be checked end to end.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from cattle3d.cameras import Camera, intrinsics, look_at, project_points
from cattle3d.gaussians import GaussianModel
from cattle3d.mmcows import CAMERA_NAMES, Detection, write_detections
from cattle3d.projection import compose_projection, write_projection_xml
from cattle3d.render import render_gaussians

IMAGE_W = 192
IMAGE_H = 120
HEAD_CENTER = np.array([1.20, -0.40, 1.45], dtype=np.float64)
COW_ID = 3
DATE = "0725"
TIMESTAMP = 1721902638


def write_dataset(root: str | Path, seed: int = 0) -> Path:
    """Write a tiny MmCows-format capture of a still cow head. Returns ``root``."""
    root = Path(root)
    rng = np.random.default_rng(seed)
    model = stylized_head(HEAD_CENTER, rng, count=100)
    cameras = surround_cameras(HEAD_CENTER)
    stem = f"{TIMESTAMP}_14-57-18"
    for index, camera in enumerate(cameras):
        image = _render_uint8(model, camera)
        name = CAMERA_NAMES[index]
        image_path = root / "images" / DATE / name / f"{stem}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        detection = _detection_from_model(model, camera)
        write_detections(root / "labels" / "combined" / DATE / name / f"{stem}.txt", [detection])
        projection = compose_projection(camera.K, camera.R, camera.t * 100.0)
        write_projection_xml(
            root / "proj_mat" / DATE / f"proj_mat_cam{index + 1}.xml",
            projection,
            camera.K,
        )
    calib = root / "proj_mat" / DATE / "calibration.json"
    calib.write_text(json.dumps({"width": IMAGE_W, "height": IMAGE_H, "unit": "cm"}))
    behavior = root / "behavior_labels" / "individual" / f"C{COW_ID:02d}_{DATE}.csv"
    behavior.parent.mkdir(parents=True, exist_ok=True)
    with behavior.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["datetime", "timestamp", "behavior"])
        writer.writerow(["2024-07-25 14:57:18", TIMESTAMP, 2])
    (root / "source.json").write_text(
        json.dumps({"cow_id": COW_ID, "date": DATE, "head_center_m": HEAD_CENTER.tolist(), "timestamp": TIMESTAMP})
    )
    return root


def stylized_head(center: np.ndarray, rng: np.random.Generator, count: int = 220) -> GaussianModel:
    """Ellipsoid head with a pale blaze, dark eyes, and a muzzle."""
    center = np.asarray(center, dtype=np.float64)
    radii = np.array([0.26, 0.18, 0.22])
    raw = rng.normal(size=(count, 3))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    raw *= rng.uniform(0.35, 1.0, size=(count, 1))
    points = center + raw * radii
    colors = np.tile(np.array([0.45, 0.30, 0.16], dtype=np.float64), (count, 1))
    local = points - center
    blaze = (np.abs(local[:, 1]) < 0.045) & (local[:, 2] > -0.02)
    colors[blaze] = np.array([0.92, 0.88, 0.80])
    left_eye = center + np.array([0.16, 0.07, 0.05])
    right_eye = center + np.array([0.16, -0.07, 0.05])
    muzzle = center + np.array([0.22, 0.0, -0.05])
    colors[np.linalg.norm(points - left_eye, axis=1) < 0.045] = np.array([0.05, 0.04, 0.03])
    colors[np.linalg.norm(points - right_eye, axis=1) < 0.045] = np.array([0.05, 0.04, 0.03])
    muzzle_mask = np.linalg.norm(points - muzzle, axis=1) < 0.07
    colors[muzzle_mask] = np.array([0.72, 0.45, 0.42])
    model = GaussianModel.from_points(points.astype(np.float32), colors.astype(np.float32), opacity=0.85)
    # Overlap neighboring splats so the rendered head is a surface, not a point cloud.
    with torch.no_grad():
        model.log_scales += float(np.log(1.8))
    return model


def surround_cameras(center: np.ndarray) -> list[Camera]:
    """Four isometric views around ``center``, matching the MmCows camera count."""
    center = np.asarray(center, dtype=np.float64)
    K = intrinsics(IMAGE_W, IMAGE_H, focal=210.0)
    cameras = []
    for index, degrees in enumerate((35.0, 145.0, 215.0, 325.0)):
        angle = np.deg2rad(degrees)
        eye = center + np.array([2.5 * np.cos(angle), 2.5 * np.sin(angle), 0.85])
        cameras.append(look_at(eye, center, np.array([0.0, 0.0, 1.0]), IMAGE_W, IMAGE_H, K, camera_id=index + 1))
    return cameras


def _render_uint8(model: GaussianModel, camera: Camera) -> np.ndarray:
    bg = torch.zeros(3)
    with torch.no_grad():
        image = render_gaussians(model.cpu(), camera, bg, tile=16, max_per_tile=128)
    return (image.clamp(0, 1).cpu().numpy() * 255.0).round().astype(np.uint8)


def _detection_from_model(model: GaussianModel, camera: Camera) -> Detection:
    means = model.means.detach().cpu().numpy()
    u, v, z = project_points(camera, means)
    keep = z > 0.05
    u, v = u[keep], v[keep]
    pad_u = 0.08 * (u.max() - u.min() + 1.0)
    pad_v = 0.08 * (v.max() - v.min() + 1.0)
    left, right = float(u.min() - pad_u), float(u.max() + pad_u)
    top, bottom = float(v.min() - pad_v), float(v.max() + pad_v)
    left, right = max(0.0, left), min(float(camera.width - 1), right)
    top, bottom = max(0.0, top), min(float(camera.height - 1), bottom)
    width = max(right - left, 1.0)
    height = max(bottom - top, 1.0)
    return Detection(
        cow_id=COW_ID,
        cx=((left + right) / 2.0) / camera.width,
        cy=((top + bottom) / 2.0) / camera.height,
        w=width / camera.width,
        h=height / camera.height,
        behavior=2,
    )
