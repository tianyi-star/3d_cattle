import numpy as np
import torch

from cattle3d.cameras import Camera, intrinsics, project_points
from cattle3d.gaussians import GaussianModel
from cattle3d.render import render_gaussians


def _camera():
    return Camera(np.eye(3), np.zeros(3), intrinsics(64, 64, 80), 64, 64)


def _model(means, colors, scales, opacity=0.95):
    points = np.asarray(means, dtype=np.float32)
    rgb = np.asarray(colors, dtype=np.float32)
    if rgb.ndim == 1:
        rgb = np.repeat(rgb[None], len(points), axis=0)
    model = GaussianModel.from_points(points, rgb, opacity=opacity)
    with torch.no_grad():
        model.log_scales.fill_(float(np.log(scales)))
    return model


def test_gaussian_peaks_on_its_projected_mean():
    camera = _camera()
    means = np.array([[0.05, -0.08, 2.0]], dtype=np.float32)
    model = _model(means, [0.9, 0.05, 0.05], scales=0.18)
    image = render_gaussians(model, camera, torch.zeros(3), max_per_tile=8).detach()
    peak = torch.stack(torch.where(image[:, :, 0] == image[:, :, 0].max())).float().mean(1)
    u, v, _z = project_points(camera, means)
    # peak row is v, peak col is u
    assert abs(float(peak[1]) - u[0]) < 1.6
    assert abs(float(peak[0]) - v[0]) < 1.6
    assert float(image[32, 32, 0]) > 0.2
    assert float(image[2, 2].max()) < 0.05


def test_near_gaussian_occludes_far_one():
    camera = _camera()
    means = np.array([[0.0, 0.0, 1.6], [0.0, 0.0, 3.0]], dtype=np.float32)
    colors = np.array([[0.05, 0.85, 0.05], [0.9, 0.05, 0.05]], dtype=np.float32)
    model = _model(means, colors, scales=0.25)
    image = render_gaussians(model, camera, torch.zeros(3), max_per_tile=8).detach()
    center = image[32, 32]
    assert float(center[1]) > 0.5
    assert float(center[0]) < 0.25


def test_pixel_loss_reaches_the_mean():
    camera = _camera()
    model = _model([[0.0, 0.0, 2.0]], [0.8, 0.1, 0.1], scales=0.2, opacity=0.8)
    image = render_gaussians(model, camera, torch.zeros(3), max_per_tile=8)
    image[30:34, 30:34, 0].mean().backward()
    assert model.means.grad is not None
    assert float(model.means.grad.abs().sum()) > 0
