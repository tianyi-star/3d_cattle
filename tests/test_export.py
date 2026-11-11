import numpy as np
import torch

from cattle3d.export import load_gaussians_ply, save_gaussians_ply
from cattle3d.gaussians import GaussianModel


def test_ply_roundtrip(tmp_path):
    points = np.array([[0.0, 0.1, 0.2], [0.3, -0.2, 0.4]], dtype=np.float32)
    colors = np.array([[0.2, 0.4, 0.6], [0.8, 0.1, 0.3]], dtype=np.float32)
    model = GaussianModel.from_points(points, colors, opacity=0.4)
    path = tmp_path / "gaussians.ply"
    save_gaussians_ply(path, model)
    loaded = load_gaussians_ply(path)
    assert loaded.num_points == 2
    assert torch.allclose(loaded.means, model.means, atol=1e-5)
    assert torch.allclose(loaded.logit_opacity, model.logit_opacity, atol=1e-4)
    assert torch.allclose(loaded.activated()["color"], model.activated()["color"], atol=1e-4)
