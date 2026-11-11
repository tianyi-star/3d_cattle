"""PLY export compatible with 3D Gaussian Splatting viewers, plus turntable video."""

from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

from cattle3d.cameras import intrinsics, look_at
from cattle3d.gaussians import GaussianModel, _logit
from cattle3d.render import render_gaussians

SH_C0 = 0.28209479177387814

PLY_PROPERTIES = (
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)


def save_gaussians_ply(path: str | Path, model: GaussianModel) -> None:
    arrays = _ply_table(model)
    count = len(arrays)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
    header.extend(f"property float {name}" for name in PLY_PROPERTIES)
    header.append("end_header\n")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        handle.write(("\n".join(header)).encode("ascii"))
        arrays.tofile(handle)


def load_gaussians_ply(path: str | Path) -> GaussianModel:
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    end = raw.find(marker)
    if end < 0:
        raise ValueError(f"{path} is not a PLY file")
    header = raw[:end].decode("ascii", errors="replace")
    count = None
    for line in header.splitlines():
        if line.startswith("element vertex"):
            count = int(line.split()[-1])
    if count is None:
        raise ValueError(f"{path} has no vertex element")
    body = raw[end + len(marker) :]
    table = np.frombuffer(body, dtype=np.dtype([(name, "<f4") for name in PLY_PROPERTIES]), count=count)
    def column(*names: str) -> np.ndarray:
        stacked = np.stack([np.array(table[name], dtype=np.float32, copy=True) for name in names], axis=1)
        return np.ascontiguousarray(stacked)
    means = column("x", "y", "z")
    log_scales = column("scale_0", "scale_1", "scale_2")
    quats = column("rot_0", "rot_1", "rot_2", "rot_3")
    logit_opacity = np.array(table["opacity"], dtype=np.float32, copy=True).reshape(-1, 1)
    f_dc = column("f_dc_0", "f_dc_1", "f_dc_2")
    rgb = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)
    logit_color = _logit(rgb).astype(np.float32)
    return GaussianModel(means, log_scales, quats, logit_opacity, logit_color)


def render_turntable(
    model: GaussianModel,
    path: str | Path,
    frames: int = 24,
    size: int = 192,
    radius: float = 2.4,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fps: int = 12,
) -> None:
    """Orbit the origin in the xy plane. World +Z is up, which matches the barn."""
    if frames <= 0:
        return
    size = int(size) + (int(size) % 2)
    model = model.cpu()
    bg = torch.tensor(background, dtype=torch.float32)
    images = []
    focal = 1.15 * size
    K = intrinsics(size, size, focal)
    for angle in np.linspace(0.0, 2.0 * np.pi, frames, endpoint=False):
        eye = np.array([radius * np.cos(angle), radius * np.sin(angle), 0.35], dtype=np.float64)
        camera = look_at(eye, np.zeros(3), np.array([0.0, 0.0, 1.0]), size, size, K)
        with torch.no_grad():
            image = render_gaussians(model, camera, bg, tile=16, max_per_tile=max(64, model.num_points))
        frame = (image.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        images.append(frame)
    write_video(path, images, fps=fps)


def write_video(path: str | Path, frames: list[np.ndarray], fps: int = 12) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    scratch = destination.parent / f".{destination.stem}_frames"
    if scratch.exists():
        for old in scratch.glob("*.png"):
            old.unlink()
    scratch.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(scratch / f"{index:04d}.png"), bgr)
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(scratch / "%04d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(destination),
    ]
    subprocess.run(command, check=True)
    for old in scratch.glob("*.png"):
        old.unlink()
    scratch.rmdir()


def save_comparison(path: str | Path, pairs: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Save ground-truth | render strips stacked vertically."""
    strips = []
    for gt, pred in pairs:
        if gt.shape != pred.shape:
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        gap = np.full((gt.shape[0], 4, 3), 255, dtype=np.uint8)
        strips.append(np.concatenate([gt, gap, pred], axis=1))
    width = max(strip.shape[1] for strip in strips)
    padded = []
    for strip in strips:
        if strip.shape[1] < width:
            pad = np.zeros((strip.shape[0], width - strip.shape[1], 3), dtype=np.uint8)
            strip = np.concatenate([strip, pad], axis=1)
        padded.append(strip)
    sheet = np.concatenate(padded, axis=0)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def _ply_table(model: GaussianModel) -> np.ndarray:
    with torch.no_grad():
        state = model.activated()
        means = state["means"].detach().cpu().numpy()
        color = state["color"].detach().cpu().numpy()
        log_scales = model.log_scales.detach().cpu().numpy()
        opacity = model.logit_opacity.detach().cpu().numpy().reshape(-1)
        quats = state["quats"].detach().cpu().numpy()
    f_dc = (color - 0.5) / SH_C0
    table = np.zeros(len(means), dtype=[(name, "<f4") for name in PLY_PROPERTIES])
    table["x"], table["y"], table["z"] = means[:, 0], means[:, 1], means[:, 2]
    table["f_dc_0"], table["f_dc_1"], table["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    table["opacity"] = opacity
    table["scale_0"], table["scale_1"], table["scale_2"] = log_scales[:, 0], log_scales[:, 1], log_scales[:, 2]
    table["rot_0"], table["rot_1"], table["rot_2"], table["rot_3"] = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return table
