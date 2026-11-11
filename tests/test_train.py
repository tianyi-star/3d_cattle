import numpy as np
import torch

from cattle3d.cameras import intrinsics, look_at
from cattle3d.gaussians import GaussianModel
from cattle3d.render import render_gaussians
from cattle3d.synthetic import stylized_head
from cattle3d.train import TrainConfig, TrainView, train_gaussians


def test_training_reduces_loss_on_a_head():
    rng = np.random.default_rng(1)
    truth = stylized_head(np.zeros(3), rng, count=40)
    views = []
    background = torch.zeros(3)
    for index, degrees in enumerate((20.0, 90.0, 170.0)):
        angle = np.deg2rad(degrees)
        eye = np.array([1.6 * np.cos(angle), 1.6 * np.sin(angle), 0.4])
        camera = look_at(eye, np.zeros(3), np.array([0.0, 0.0, 1.0]), 48, 48, intrinsics(48, 48, 70), camera_id=index + 1)
        with torch.no_grad():
            image = render_gaussians(truth, camera, background, tile=16, max_per_tile=48)
        rgb = (image.clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)
        mask = np.ones(rgb.shape[:2], np.float32)
        views.append(TrainView(rgb, mask, camera))
    noisy = truth.means.detach().cpu().numpy() + rng.normal(scale=0.02, size=(truth.num_points, 3))
    colors = truth.activated()["color"].detach().cpu().numpy()
    start = GaussianModel.from_points(noisy.astype(np.float32), colors.astype(np.float32), opacity=0.5)
    cfg = TrainConfig(iterations=12, max_per_tile=48, lambda_ssim=0.0, seed=1, lr_means=0.01)
    result = train_gaussians(start, views, cfg)
    assert result.loss_end < result.loss_start * 0.7
    assert result.psnr > 12.0
