"""Select a still cow in an MmCows capture and fit a 3D Gaussian head."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from cattle3d.cameras import Camera, project_points
from cattle3d.config import PipelineConfig
from cattle3d.export import render_turntable, save_comparison, save_gaussians_ply
from cattle3d.gaussians import GaussianModel
from cattle3d.localize import localize_bbox_centers
from cattle3d.mmcows import BEHAVIOR_NAMES, Detection, FrameRecord, MmCowsVision, read_rgb
from cattle3d.render import render_gaussians
from cattle3d.train import TrainView, train_gaussians

logger = logging.getLogger(__name__)


@dataclass
class _Shot:
    frame: FrameRecord
    detection: Detection
    camera: Camera


@dataclass
class _Stamp:
    timestamp: int
    point: np.ndarray
    behavior: int | None
    shots: list[_Shot]


def reconstruct(visual_data: str | Path, out_dir: str | Path, cfg: PipelineConfig, sensor_data: str | Path | None = None) -> dict:
    """Fit a head splat for ``cfg.cow_id`` and write it under ``out_dir``."""
    dataset = MmCowsVision(visual_data, label_set=cfg.label_set)
    extra = [Path(sensor_data)] if sensor_data else []
    behavior = dataset.behavior_table(cfg.date, cfg.cow_id, extra_roots=extra)
    if cfg.allowed_behaviors and not behavior:
        logger.warning("No behavior CSV for cow %s on %s; selecting frames by motion only", cfg.cow_id, cfg.date)
    frames = dataset.iter_frames(cfg.date)
    if not frames:
        raise RuntimeError(f"No frames found for date {cfg.date} under {dataset.root / 'images'}")
    sample = read_rgb(frames[0].image_path)
    image_h, image_w = sample.shape[:2]
    calib = dataset.calibration_size(cfg.date, default=(cfg.calibration_width, cfg.calibration_height))
    cameras = dataset.load_cameras(cfg.date, image_size=(image_w, image_h), calibration_size=calib)
    stamps = _stamps(dataset, frames, cameras, cfg, behavior)
    chosen = _select_window(stamps, cfg)
    origin = _head_origin(chosen, cfg)
    views = _build_views(chosen, origin, cfg)
    if len(views) < cfg.min_views:
        raise RuntimeError(
            f"Only {len(views)} head crops survived for cow {cfg.cow_id}. "
            "The head volume has to project inside the cow box on at least "
            f"{cfg.min_views} cameras."
        )
    points = _sample_head(origin_local=np.zeros(3), size=np.asarray(cfg.head_size_m, dtype=np.float64), count=cfg.init_points, seed=cfg.seed)
    colors = _sample_colors(points, views)
    model = GaussianModel.from_points(points, colors, opacity=0.2)
    result = train_gaussians(model, views, cfg.train, background=cfg.background)
    return _write_outputs(Path(out_dir), result, views, chosen, origin, cfg)


def _stamps(dataset, frames, cameras, cfg: PipelineConfig, behavior: dict[int, int]) -> list[_Stamp]:
    grouped: dict[int, list[tuple[FrameRecord, Detection]]] = {}
    for frame in frames:
        detection = dataset.detections_for(frame, cfg.cow_id)
        if detection is None:
            continue
        grouped.setdefault(frame.timestamp, []).append((frame, detection))
    stamps: list[_Stamp] = []
    for timestamp, items in grouped.items():
        if len(items) < cfg.min_views:
            continue
        used_cameras = []
        pixels = []
        shots = []
        for frame, detection in items:
            camera = cameras[frame.camera_index]
            u, v = detection.pixel_center(camera.width, camera.height)
            used_cameras.append(camera)
            pixels.append((u, v))
            shots.append(_Shot(frame, detection, camera))
        point = localize_bbox_centers(used_cameras, pixels)
        stamps.append(_Stamp(timestamp, point, behavior.get(timestamp), shots))
    if not stamps:
        raise RuntimeError(
            f"Cow {cfg.cow_id} is not visible in {cfg.min_views} or more cameras on {cfg.date}."
        )
    return stamps


def _select_window(stamps: list[_Stamp], cfg: PipelineConfig) -> list[_Stamp]:
    allowed = set(cfg.allowed_behaviors) if cfg.allowed_behaviors else None
    pool = [stamp for stamp in stamps if allowed is None or stamp.behavior is None or stamp.behavior in allowed]
    if not pool:
        logger.warning("Behavior filter removed every frame; using every multi-view timestamp")
        pool = list(stamps)
    pool.sort(key=lambda stamp: stamp.timestamp)
    best: list[_Stamp] | None = None
    best_views = -1
    best_motion = 1e9
    for start, _stamp in enumerate(pool):
        window = [pool[start]]
        for nxt in pool[start + 1 :]:
            stacked = np.stack([item.point[:2] for item in window + [nxt]])
            if float((stacked.max(0) - stacked.min(0)).max()) > cfg.max_motion_m:
                break
            window.append(nxt)
        views = sum(len(item.shots) for item in window)
        stacked = np.stack([item.point[:2] for item in window])
        motion = float((stacked.max(0) - stacked.min(0)).max()) if len(window) else 0.0
        if views > best_views or (views == best_views and motion < best_motion):
            best = window
            best_views = views
            best_motion = motion
    assert best is not None
    if len(best) > cfg.max_frames:
        indexes = np.linspace(0, len(best) - 1, cfg.max_frames).round().astype(int)
        best = [best[int(index)] for index in indexes]
    logger.info(
        "Using %d timestamps (%d views), xy motion %.2f m, behaviors %s",
        len(best),
        sum(len(item.shots) for item in best),
        best_motion,
        sorted({BEHAVIOR_NAMES.get(item.behavior, "unlabeled") for item in best}),
    )
    return best


def _head_origin(stamps: list[_Stamp], cfg: PipelineConfig) -> np.ndarray:
    origin = np.mean([stamp.point for stamp in stamps], axis=0)
    # A full-body box center often lands on the torso. Lift very low points
    # into the head band. A head-sized box (the synthetic capture, or a tight
    # box) is already near 1.4 m and is left where the rays agree.
    if origin[2] < 1.0:
        origin = origin.copy()
        origin[2] = float(cfg.head_z_m)
    return origin


def _build_views(stamps: list[_Stamp], origin: np.ndarray, cfg: PipelineConfig) -> list[TrainView]:
    views: list[TrainView] = []
    for stamp in stamps:
        for shot in stamp.shots:
            image = read_rgb(shot.frame.image_path)
            camera = shot.camera
            if (image.shape[1], image.shape[0]) != (camera.width, camera.height):
                camera = camera.resized(image.shape[1], image.shape[0])
            cropped = _crop_head(image, camera, shot.detection, origin, np.asarray(cfg.head_size_m, dtype=np.float64))
            if cropped is None:
                continue
            crop, mask, crop_camera = cropped
            crop, mask, crop_camera = _resize_long_side(crop, mask, crop_camera, cfg.train_long_side)
            local_camera = crop_camera.translated(origin)
            views.append(
                TrainView(
                    image=crop,
                    mask=mask,
                    camera=local_camera,
                    timestamp=stamp.timestamp,
                    camera_id=shot.camera.camera_id,
                )
            )
    return views


def _crop_head(image, camera, detection: Detection, origin: np.ndarray, size: np.ndarray):
    height, width = image.shape[:2]
    box = detection.pixel_xyxy(width, height)
    x0, y0, x1, y1 = _expand_box(box, 0.08, width, height)
    corners = _box_corners(origin, size)
    u, v, z = project_points(camera, corners)
    visible = z > 0.05
    if int(visible.sum()) < 4:
        return None
    cloud = np.stack([u[visible], v[visible]], axis=1).astype(np.float32)
    hull = cv2.convexHull(cloud)
    region = np.zeros((height, width), np.uint8)
    cv2.fillConvexPoly(region, np.round(hull).astype(np.int32), 255)
    cow = np.zeros_like(region)
    cv2.rectangle(cow, (int(x0), int(y0)), (int(np.ceil(x1)), int(np.ceil(y1))), 255, thickness=-1)
    region = cv2.bitwise_and(region, cow)
    ys, xs = np.where(region > 0)
    if len(xs) < 40:
        return None
    left, right = max(0, int(xs.min()) - 2), min(width, int(xs.max()) + 3)
    top, bottom = max(0, int(ys.min()) - 2), min(height, int(ys.max()) + 3)
    if right - left < 12 or bottom - top < 12:
        return None
    crop = image[top:bottom, left:right].copy()
    mask = (region[top:bottom, left:right] > 0).astype(np.float32)
    return crop, mask, camera.cropped(left, top, right - left, bottom - top)


def _resize_long_side(image, mask, camera: Camera, long_side: int):
    height, width = image.shape[:2]
    scale = float(long_side) / float(max(height, width))
    new_w = max(8, int(round(width * scale)))
    new_h = max(8, int(round(height * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    image = cv2.resize(image, (new_w, new_h), interpolation=interp)
    mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0).astype(np.float32)
    return image, mask, camera.resized(new_w, new_h)


def _sample_head(origin_local: np.ndarray, size: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(count, 3))
    raw /= np.linalg.norm(raw, axis=1, keepdims=True)
    raw *= rng.uniform(0.0, 1.0, size=(count, 1)) ** (1.0 / 3.0)
    return (origin_local + raw * (size / 2.0)).astype(np.float32)


def _sample_colors(points: np.ndarray, views: list[TrainView]) -> np.ndarray:
    colors = np.zeros((len(points), 3), dtype=np.float64)
    counts = np.zeros(len(points), dtype=np.float64)
    for view in views:
        u, v, z = project_points(view.camera, points)
        ui = np.rint(u).astype(int)
        vi = np.rint(v).astype(int)
        height, width = view.image.shape[:2]
        inside = (z > 0.05) & (ui >= 0) & (vi >= 0) & (ui < width) & (vi < height)
        if not np.any(inside):
            continue
        covered = view.mask[vi[inside], ui[inside]] > 0.4
        chosen = np.flatnonzero(inside)[covered]
        if len(chosen) == 0:
            continue
        colors[chosen] += view.image[vi[chosen], ui[chosen]].astype(np.float64) / 255.0
        counts[chosen] += 1.0
    hit = counts > 0
    colors[hit] /= counts[hit, None]
    colors[~hit] = 0.5
    return colors.astype(np.float32)


def _write_outputs(out_dir: Path, result, views: list[TrainView], stamps: list[_Stamp], origin: np.ndarray, cfg: PipelineConfig) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_path = out_dir / "gaussians.ply"
    save_gaussians_ply(ply_path, result.model)
    preview = out_dir / "preview"
    preview.mkdir(exist_ok=True)
    pairs = _comparison_pairs(result.model, views[:3], cfg)
    if pairs:
        save_comparison(preview / "compare.png", pairs)
    radius = _orbit_radius(views)
    video_path = preview / "turntable.mp4"
    if cfg.turntable_frames > 0:
        render_turntable(
            result.model,
            video_path,
            frames=cfg.turntable_frames,
            size=cfg.turntable_size,
            radius=radius,
            background=cfg.background,
        )
    meta = {
        "source": "mmcows",
        "cow_id": int(cfg.cow_id),
        "date": str(cfg.date),
        "timestamps": [int(stamp.timestamp) for stamp in stamps],
        "behaviors": [BEHAVIOR_NAMES.get(stamp.behavior, "unlabeled") for stamp in stamps],
        "num_views": len(views),
        "num_gaussians": result.model.num_points,
        "origin_m": [float(value) for value in origin],
        "head_size_m": [float(value) for value in cfg.head_size_m],
        "train_psnr": result.psnr,
        "train_ssim": result.ssim,
        "loss_start": result.loss_start,
        "loss_end": result.loss_end,
        "ply": str(ply_path),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("cow %s psnr %.2f  gaussians %d  -> %s", cfg.cow_id, result.psnr, result.model.num_points, ply_path)
    return meta


def _comparison_pairs(model: GaussianModel, views: list[TrainView], cfg: PipelineConfig):
    if not views:
        return []
    model = model.cpu()
    bg = torch.tensor(cfg.background, dtype=torch.float32)
    pairs = []
    for view in views:
        with torch.no_grad():
            predicted = render_gaussians(model, view.camera, bg, tile=cfg.train.tile, max_per_tile=cfg.train.max_per_tile)
        pred = (predicted.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        pairs.append((view.image, pred))
    return pairs


def _orbit_radius(views: list[TrainView]) -> float:
    distances = [float(np.linalg.norm(view.camera.center())) for view in views]
    radius = float(np.median(distances)) if distances else 2.4
    return float(np.clip(radius, 1.2, 6.0))


def _expand_box(box, fraction, width, height):
    x0, y0, x1, y1 = box
    dx, dy = (x1 - x0) * fraction, (y1 - y0) * fraction
    return max(0.0, x0 - dx), max(0.0, y0 - dy), min(float(width), x1 + dx), min(float(height), y1 + dy)


def _box_corners(center: np.ndarray, size: np.ndarray) -> np.ndarray:
    half = np.asarray(size, dtype=np.float64) / 2.0
    center = np.asarray(center, dtype=np.float64)
    corners = []
    for x in (-half[0], half[0]):
        for y in (-half[1], half[1]):
            for z in (-half[2], half[2]):
                corners.append(center + np.array([x, y, z]))
    return np.stack(corners)
