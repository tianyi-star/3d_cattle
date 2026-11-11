"""Differentiable tile rasterizer for 3D Gaussians.

The rasterizer follows the EWA projection from Kerbl et al. (3D Gaussian
Splatting, SIGGRAPH 2023): project the world covariance with the local affine
approximation of the pinhole camera, then alpha-composite front to back. It is
a pure PyTorch implementation, so the same code trains on CPU and CUDA. It is
not a copy of the Inria CUDA rasterizer.
"""

from __future__ import annotations

import math

import torch

from cattle3d.cameras import Camera
from cattle3d.gaussians import GaussianModel


def quat_to_rotmat(quats: torch.Tensor) -> torch.Tensor:
    quats = quats / quats.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = quats.unbind(-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    return torch.stack(
        (
            torch.stack((ww + xx - yy - zz, 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1),
            torch.stack((2 * (x * y + z * w), ww - xx + yy - zz, 2 * (y * z - x * w)), dim=-1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), ww - xx - yy + zz), dim=-1),
        ),
        dim=-2,
    )


def world_covariance(quats: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    rotation = quat_to_rotmat(quats)
    scaled = rotation * scales[:, None, :]
    return scaled @ scaled.transpose(-1, -2)


def render_gaussians(
    model: GaussianModel,
    camera: Camera,
    background: torch.Tensor,
    tile: int = 16,
    max_per_tile: int = 1024,
    near: float = 0.05,
) -> torch.Tensor:
    """Render ``model`` into an image of ``camera.height`` by ``camera.width``."""
    height, width = int(camera.height), int(camera.width)
    device = model.means.device
    dtype = model.means.dtype
    bg = background.to(device=device, dtype=dtype).reshape(3)
    state = model.activated()
    cov = world_covariance(state["quats"], state["scales"])
    u, v, depth, cov2d, valid = _project(state["means"], cov, camera, near)
    opacity = torch.where(valid, state["opacity"].reshape(-1), torch.zeros_like(depth))
    image = _rasterize(
        u, v, depth, cov2d, opacity, state["color"], valid, height, width, bg, tile, max_per_tile
    )
    return image


def _project(
    means: torch.Tensor,
    cov: torch.Tensor,
    camera: Camera,
    near: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device, dtype = means.device, means.dtype
    rotation = torch.as_tensor(camera.R, device=device, dtype=dtype)
    translation = torch.as_tensor(camera.t, device=device, dtype=dtype)
    intrinsics = torch.as_tensor(camera.K, device=device, dtype=dtype)
    camera_pts = means @ rotation.T + translation
    z = camera_pts[:, 2]
    in_front = z > near
    z_safe = z.clamp(min=near)
    x, y = camera_pts[:, 0], camera_pts[:, 1]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    u = fx * x / z_safe + cx
    v = fy * y / z_safe + cy

    jacobian = torch.zeros(means.shape[0], 2, 3, device=device, dtype=dtype)
    jacobian[:, 0, 0] = fx / z_safe
    jacobian[:, 0, 2] = -fx * x / z_safe**2
    jacobian[:, 1, 1] = fy / z_safe
    jacobian[:, 1, 2] = -fy * y / z_safe**2
    cov_cam = rotation @ cov @ rotation.T
    cov2d = jacobian @ cov_cam @ jacobian.transpose(-1, -2)
    eye = torch.eye(2, device=device, dtype=dtype)
    cov2d = cov2d + 0.3 * eye  # EWA low-pass, in pixels, as in 3DGS
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = a * c - b * b
    valid = in_front & torch.isfinite(u) & torch.isfinite(v) & (det > 1e-6) & torch.isfinite(det)
    return u, v, z_safe, cov2d, valid


def _rasterize(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: torch.Tensor,
    cov2d: torch.Tensor,
    opacity: torch.Tensor,
    color: torch.Tensor,
    valid: torch.Tensor,
    height: int,
    width: int,
    background: torch.Tensor,
    tile: int,
    max_per_tile: int,
) -> torch.Tensor:
    device = u.device
    a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
    det = (a * c - b * b).clamp(min=1e-6)
    inv00, inv01, inv11 = c / det, -b / det, a / det
    mid = 0.5 * (a + c)
    radius = 3.0 * torch.sqrt((mid + torch.sqrt((0.5 * (a - c)) ** 2 + b * b)).clamp(min=0.0))
    radius = torch.where(valid, radius.clamp(max=tile * 6), torch.full_like(radius, -1.0))

    bins = _bin_tiles(u.detach(), v.detach(), radius.detach(), depth.detach(), height, width, tile, max_per_tile)
    if bins is None:
        return background.view(1, 1, 3).expand(height, width, 3).clone()
    ids, ntx, nty, count = bins
    ids_t = torch.from_numpy(ids).to(device)
    known = ids_t >= 0
    safe = ids_t.clamp(min=0)

    def gather(values: torch.Tensor) -> torch.Tensor:
        flat = values[safe.reshape(-1)]
        return flat.reshape(*safe.shape, *values.shape[1:])

    u_g = gather(u)
    v_g = gather(v)
    op_g = gather(opacity)
    col_g = gather(color)
    inv00_g = gather(inv00)
    inv01_g = gather(inv01)
    inv11_g = gather(inv11)

    px, py = _tile_pixels(ntx, nty, tile, device, u.dtype)
    dx = px[:, None, :] - u_g[:, :, None]
    dy = py[:, None, :] - v_g[:, :, None]
    power = inv00_g[:, :, None] * dx * dx + 2.0 * inv01_g[:, :, None] * dx * dy + inv11_g[:, :, None] * dy * dy
    gauss = torch.exp(-0.5 * power.clamp(0.0, 40.0))
    alpha = op_g[:, :, None] * gauss
    alpha = torch.where(known[:, :, None], alpha, torch.zeros_like(alpha)).clamp(max=0.999)
    one_minus = 1.0 - alpha
    cumulative = torch.cumprod(one_minus, dim=1)
    transmittance = torch.cat((torch.ones_like(one_minus[:, :1]), cumulative[:, :-1]), dim=1)
    weight = alpha * transmittance
    tile_color = (col_g[:, :, None, :] * weight[..., None]).sum(dim=1)
    remaining = transmittance[:, -1] * one_minus[:, -1]
    tile_color = tile_color + remaining[..., None] * background
    return _stitch(tile_color, ntx, nty, tile, height, width)


def _bin_tiles(u, v, radius, depth, height, width, tile, max_per_tile):
    u = u.cpu().numpy()
    v = v.cpu().numpy()
    radius = radius.cpu().numpy()
    depth = depth.cpu().numpy()
    ntx = max(1, math.ceil(width / tile))
    nty = max(1, math.ceil(height / tile))
    buckets: list[list[int]] = [[] for _ in range(ntx * nty)]
    for index in range(len(u)):
        if radius[index] < 0 or not math.isfinite(radius[index]):
            continue
        x0 = int(math.floor((u[index] - radius[index]) / tile))
        x1 = int(math.floor((u[index] + radius[index]) / tile))
        y0 = int(math.floor((v[index] - radius[index]) / tile))
        y1 = int(math.floor((v[index] + radius[index]) / tile))
        if x1 < 0 or y1 < 0 or x0 >= ntx or y0 >= nty:
            continue
        x0, x1 = max(0, x0), min(ntx - 1, x1)
        y0, y1 = max(0, y0), min(nty - 1, y1)
        for ty in range(y0, y1 + 1):
            base = ty * ntx
            for tx in range(x0, x1 + 1):
                buckets[base + tx].append(index)
    if not any(buckets):
        return None
    count = min(max_per_tile, max(len(bucket) for bucket in buckets))
    ids = -np_full((ntx * nty, count))
    for tile_index, bucket in enumerate(buckets):
        if not bucket:
            continue
        order = sorted(bucket, key=lambda item: depth[item])[:count]
        ids[tile_index, : len(order)] = order
    return ids, ntx, nty, count


def _tile_pixels(ntx, nty, tile, device, dtype):
    import numpy as np

    xs, ys = [], []
    for ty in range(nty):
        for tx in range(ntx):
            x = np.arange(tile) + tx * tile + 0.5
            y = np.arange(tile) + ty * tile + 0.5
            yy, xx = np.meshgrid(y, x, indexing="ij")
            xs.append(xx.reshape(-1))
            ys.append(yy.reshape(-1))
    px = torch.as_tensor(np.stack(xs), device=device, dtype=dtype)
    py = torch.as_tensor(np.stack(ys), device=device, dtype=dtype)
    return px, py


def _stitch(tile_color, ntx, nty, tile, height, width):
    tiles = tile_color.reshape(nty, ntx, tile, tile, 3)
    rows = []
    for ty in range(nty):
        hh = min(tile, height - ty * tile)
        cols = []
        for tx in range(ntx):
            ww = min(tile, width - tx * tile)
            cols.append(tiles[ty, tx, :hh, :ww])
        rows.append(torch.cat(cols, dim=1))
    return torch.cat(rows, dim=0)


def np_full(shape):
    import numpy as np

    return np.full(shape, -1, dtype=np.int64)
