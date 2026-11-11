"""3D Gaussian parameters for a cow head.

Each Gaussian has a mean, an anisotropic scale, a quaternion rotation, an
opacity, and a view-independent RGB color. Scales and opacity are stored
pre-activation (log and logit), which is also how the exported PLY is read by
standard 3DGS viewers.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-4, 1.0 - 1e-4)
    return np.log(clipped / (1.0 - clipped))


def knn_mean_distance(points: np.ndarray, k: int = 3) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 1:
        return np.array([0.05])
    delta = pts[:, None, :] - pts[None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    dist.sort(axis=1)
    neighbor_count = min(k, len(pts) - 1)
    return dist[:, 1 : neighbor_count + 1].mean(axis=1)


class GaussianModel(nn.Module):
    def __init__(
        self,
        means: np.ndarray | torch.Tensor,
        log_scales: np.ndarray | torch.Tensor,
        quats: np.ndarray | torch.Tensor,
        logit_opacity: np.ndarray | torch.Tensor,
        logit_color: np.ndarray | torch.Tensor,
    ):
        super().__init__()
        self.means = nn.Parameter(torch.as_tensor(means, dtype=torch.float32))
        self.log_scales = nn.Parameter(torch.as_tensor(log_scales, dtype=torch.float32))
        self.quats = nn.Parameter(torch.as_tensor(quats, dtype=torch.float32))
        self.logit_opacity = nn.Parameter(torch.as_tensor(logit_opacity, dtype=torch.float32))
        self.logit_color = nn.Parameter(torch.as_tensor(logit_color, dtype=torch.float32))

    @property
    def num_points(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def from_points(
        cls,
        points: np.ndarray,
        colors: np.ndarray,
        opacity: float = 0.15,
    ) -> GaussianModel:
        pts = np.asarray(points, dtype=np.float32)
        rgb = np.asarray(colors, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        if rgb.shape != (len(pts), 3):
            raise ValueError("colors must have shape (N, 3)")
        spacing = knn_mean_distance(pts).astype(np.float32) * 0.55
        spacing = np.clip(spacing, 1e-3, 0.25)
        log_scales = np.log(spacing)[:, None].repeat(3, axis=1)
        quats = np.zeros((len(pts), 4), dtype=np.float32)
        quats[:, 0] = 1.0
        logit_opacity = np.full((len(pts), 1), _logit(np.array([opacity]))[0], dtype=np.float32)
        logit_color = _logit(rgb).astype(np.float32)
        return cls(pts, log_scales, quats, logit_opacity, logit_color)

    def activated(self) -> dict[str, torch.Tensor]:
        quats = self.quats
        quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        return {
            "means": self.means,
            "scales": self.log_scales.exp(),
            "quats": quats,
            "opacity": self.logit_opacity.sigmoid(),
            "color": self.logit_color.sigmoid(),
        }

    def clamp_parameters_(self) -> None:
        with torch.no_grad():
            self.log_scales.clamp_(-8.0, 0.5)
            self.logit_opacity.clamp_(-12.0, 12.0)
            self.quats.div_(self.quats.norm(dim=-1, keepdim=True).clamp(min=1e-8))
