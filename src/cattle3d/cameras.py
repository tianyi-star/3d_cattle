"""Pinhole cameras in the OpenCV convention: x right, y down, z forward."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    """World-to-camera rigid transform and intrinsics.

    ``X_cam = R @ X_world + t``. ``R`` is a proper rotation (det +1).
    """

    R: np.ndarray
    t: np.ndarray
    K: np.ndarray
    width: int
    height: int
    camera_id: int = 0

    def __post_init__(self) -> None:
        self.R = np.asarray(self.R, dtype=np.float64).reshape(3, 3)
        self.t = np.asarray(self.t, dtype=np.float64).reshape(3)
        self.K = np.asarray(self.K, dtype=np.float64).reshape(3, 3)
        self.width = int(self.width)
        self.height = int(self.height)
        self.camera_id = int(self.camera_id)

    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    def cropped(self, x0: float, y0: float, width: int, height: int) -> Camera:
        """Shift the principal point into a crop whose top-left is ``(x0, y0)``."""
        intrinsics = self.K.copy()
        intrinsics[0, 2] -= float(x0)
        intrinsics[1, 2] -= float(y0)
        return Camera(self.R, self.t, intrinsics, int(width), int(height), self.camera_id)

    def resized(self, width: int, height: int) -> Camera:
        sx = float(width) / float(self.width)
        sy = float(height) / float(self.height)
        intrinsics = self.K.copy()
        intrinsics[0, 0] *= sx
        intrinsics[0, 2] *= sx
        intrinsics[1, 1] *= sy
        intrinsics[1, 2] *= sy
        return Camera(self.R, self.t, intrinsics, int(width), int(height), self.camera_id)

    def translated(self, origin: np.ndarray) -> Camera:
        """Express the camera in a frame whose origin is ``origin`` in the old frame.

        ``X_local = X_world - origin`` leaves image projections unchanged.
        """
        origin = np.asarray(origin, dtype=np.float64).reshape(3)
        return Camera(self.R, self.t + self.R @ origin, self.K, self.width, self.height, self.camera_id)


def intrinsics(width: int, height: int, focal: float) -> np.ndarray:
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray, width: int, height: int, K: np.ndarray, camera_id: int = 0) -> Camera:
    """Camera at ``eye`` looking toward ``target``.

    ``up`` is the world up direction. Image +u points to the camera's right and
    image +v points down, matching OpenCV.
    """
    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    up = np.asarray(up, dtype=np.float64).reshape(3)
    forward = target - eye
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        raise ValueError("look_at eye and target are the same point")
    forward = forward / norm
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-6:
        axis = np.array([1.0, 0.0, 0.0]) if abs(forward[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, axis)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)
    rotation = np.stack([right, down, forward], axis=0)
    translation = -rotation @ eye
    return Camera(rotation, translation, K, width, height, camera_id)


def project_points(camera: Camera, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to pixels. Returns ``u, v, z_cam``."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    camera_pts = pts @ camera.R.T + camera.t
    z = camera_pts[:, 2]
    z_safe = np.where(np.abs(z) < 1e-8, 1e-8, z)
    pixels = camera_pts @ camera.K.T
    u = pixels[:, 0] / z_safe
    v = pixels[:, 1] / z_safe
    return u, v, z


def pixel_ray(camera: Camera, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
    """Ray origin and unnormalized direction in world coordinates through a pixel."""
    direction_cam = np.linalg.inv(camera.K) @ np.array([u, v, 1.0], dtype=np.float64)
    direction = camera.R.T @ direction_cam
    return camera.center(), direction
