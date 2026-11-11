"""MmCows projection matrices.

MmCows stores a 3x4 world-to-pixel matrix per camera in OpenCV XML. The matrix
was calibrated in centimeters. This module turns it into an OpenCV pinhole
camera in meters: ``X_cam = R @ X_m + t``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from cattle3d.cameras import Camera


def write_projection_xml(path: str | Path, projection: np.ndarray, camera_matrix: np.ndarray) -> None:
    """Write ``projection_matrix`` and ``camera_matrix`` in the MmCows XML layout."""
    root = ET.Element("opencv_storage")
    _add_matrix(root, "projection_matrix", np.asarray(projection, dtype=np.float64))
    _add_matrix(root, "camera_matrix", np.asarray(camera_matrix, dtype=np.float64))
    tree = ET.ElementTree(root)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def read_projection_xml(path: str | Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return ``(P, K)`` where ``K`` is the stored camera matrix, if present."""
    root = ET.parse(path).getroot()
    projection = _read_named_matrix(root, "projection_matrix")
    stored_k = None
    if root.find(".//camera_matrix") is not None:
        stored_k = _read_named_matrix(root, "camera_matrix")
    return projection, stored_k


def compose_projection(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    extrinsic = np.hstack([np.asarray(R, dtype=np.float64), np.asarray(t, dtype=np.float64).reshape(3, 1)])
    return np.asarray(K, dtype=np.float64) @ extrinsic


def decompose_projection(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factor ``P = K [R | t]`` into intrinsics, rotation, and translation.

    The returned ``K`` has positive focal lengths and ``K[2, 2] == 1``. ``R`` is
    a proper rotation. ``P`` and ``K [R | t]`` match up to a global sign, which
    does not change the projected pixels.
    """
    matrix = np.asarray(projection, dtype=np.float64).reshape(3, 4)
    K, rotation, center_h, *_rest = cv2.decomposeProjectionMatrix(matrix)
    center = (center_h[:3] / center_h[3]).reshape(3)
    translation = (-rotation @ center).reshape(3)
    if K[2, 2] < 0:
        K = -K
        rotation = -rotation
        translation = -translation
    K = K / K[2, 2]
    signs = np.eye(3)
    if K[0, 0] < 0:
        signs[0, 0] = -1.0
    if K[1, 1] < 0:
        signs[1, 1] = -1.0
    K = K @ signs
    rotation = signs @ rotation
    translation = signs @ translation
    if np.linalg.det(rotation) < 0:
        rotation = -rotation
        translation = -translation
    return K, rotation, translation


def camera_from_projection(
    projection: np.ndarray,
    width: int,
    height: int,
    calibration_width: int,
    calibration_height: int,
    camera_id: int = 0,
    world_scale: float = 0.01,
) -> Camera:
    """Build a metric camera.

    ``world_scale`` converts the calibration unit into meters. MmCows projection
    matrices are in centimeters, so the default is ``0.01``. Intrinsics are then
    scaled from the calibration resolution (4480x2800 in MmCows) onto ``width``
    by ``height``.
    """
    K, rotation, translation = decompose_projection(projection)
    translation = translation * float(world_scale)
    sx = float(width) / float(calibration_width)
    sy = float(height) / float(calibration_height)
    K = K.copy()
    K[0, 0] *= sx
    K[0, 2] *= sx
    K[1, 1] *= sy
    K[1, 2] *= sy
    return Camera(rotation, translation, K, width, height, camera_id)


def _add_matrix(parent: ET.Element, name: str, matrix: np.ndarray) -> None:
    node = ET.SubElement(parent, name, {"type_id": "opencv-matrix"})
    rows, cols = matrix.shape
    ET.SubElement(node, "rows").text = str(rows)
    ET.SubElement(node, "cols").text = str(cols)
    ET.SubElement(node, "dt").text = "d"
    ET.SubElement(node, "data").text = " ".join(f"{value:.16e}" for value in matrix.reshape(-1))


def _read_named_matrix(root: ET.Element, name: str) -> np.ndarray:
    node = root.find(".//" + name)
    if node is None:
        raise ValueError(f"XML is missing {name}")
    rows = int(node.findtext("rows"))
    cols = int(node.findtext("cols"))
    data = node.findtext("data")
    if data is None:
        raise ValueError(f"XML matrix {name} has no data")
    values = np.fromstring(data, sep=" ")
    if values.size != rows * cols:
        raise ValueError(f"{name} has {values.size} values, expected {rows * cols}")
    return values.reshape(rows, cols)
