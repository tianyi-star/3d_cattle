"""Multi-view body localization from bbox-center rays.

MmCows derives a 3D body point by finding the location closest to the rays that
pass through each camera and the cow's bounding-box center. The same geometric
step anchors the head volume used for Gaussian splatting.
"""

from __future__ import annotations

import numpy as np

from cattle3d.cameras import Camera, pixel_ray


def closest_point_to_rays(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    """Least-squares point closest to a set of 3D rays."""
    origins = np.asarray(origins, dtype=np.float64).reshape(-1, 3)
    directions = np.asarray(directions, dtype=np.float64).reshape(-1, 3)
    if len(origins) < 2:
        raise ValueError("at least two rays are required")
    system = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for origin, direction in zip(origins, directions):
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            continue
        direction = direction / norm
        projector = np.eye(3) - np.outer(direction, direction)
        system += projector
        rhs += projector @ origin
    return np.linalg.solve(system, rhs)


def localize_bbox_centers(cameras: list[Camera], pixels: list[tuple[float, float]]) -> np.ndarray:
    """Intersect the rays through bbox centers. ``pixels`` align with ``cameras``."""
    if len(cameras) != len(pixels):
        raise ValueError("cameras and pixels must have the same length")
    origins = []
    directions = []
    for camera, (u, v) in zip(cameras, pixels):
        origin, direction = pixel_ray(camera, u, v)
        origins.append(origin)
        directions.append(direction)
    return closest_point_to_rays(np.stack(origins), np.stack(directions))
