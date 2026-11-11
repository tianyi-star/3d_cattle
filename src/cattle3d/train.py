"""Photometric training of a cow-head Gaussian model."""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from cattle3d.cameras import Camera
from cattle3d.gaussians import GaussianModel
from cattle3d.loss import masked_psnr, photometric_loss, ssim
from cattle3d.render import render_gaussians


@dataclass
class TrainView:
    image: np.ndarray  # uint8 RGB, already at training resolution
    mask: np.ndarray  # float32 HxW
    camera: Camera
    timestamp: int = 0
    camera_id: int = 0


@dataclass
class TrainConfig:
    iterations: int = 200
    lr_means: float = 0.004
    lr_scales: float = 0.008
    lr_quats: float = 0.002
    lr_opacity: float = 0.05
    lr_color: float = 0.01
    lambda_ssim: float = 0.2
    max_per_tile: int = 1024
    tile: int = 16
    near: float = 0.02
    seed: int = 0
    log_every: int = 25


@dataclass
class TrainResult:
    model: GaussianModel
    psnr: float
    ssim: float
    loss_start: float
    loss_end: float


def train_gaussians(
    model: GaussianModel,
    views: list[TrainView],
    cfg: TrainConfig,
    background: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> TrainResult:
    if not views:
        raise ValueError("training needs at least one view")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    bg = torch.tensor(background, dtype=torch.float32, device=device)
    packed = [_pack_view(view, device) for view in views]
    optimizer = torch.optim.Adam(
        [
            {"params": [model.means], "lr": cfg.lr_means},
            {"params": [model.log_scales], "lr": cfg.lr_scales},
            {"params": [model.quats], "lr": cfg.lr_quats},
            {"params": [model.logit_opacity], "lr": cfg.lr_opacity},
            {"params": [model.logit_color], "lr": cfg.lr_color},
        ],
        eps=1e-8,
    )
    loss_start = None
    loss_end = 0.0
    steps = range(cfg.iterations)
    if cfg.iterations >= 10:
        steps = tqdm(steps, desc="train", leave=False)
    for step in steps:
        decay = 0.15 ** (step / max(cfg.iterations - 1, 1))
        optimizer.param_groups[0]["lr"] = cfg.lr_means * decay
        view = packed[step % len(packed)]
        predicted = render_gaussians(
            model,
            view["camera"],
            bg,
            tile=cfg.tile,
            max_per_tile=cfg.max_per_tile,
            near=cfg.near,
        )
        loss = photometric_loss(predicted, view["image"], view["mask"], cfg.lambda_ssim, bg)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        if loss_start is None:
            loss_start = float(loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.clamp_parameters_()
        loss_end = float(loss.detach())
    psnr, score = _evaluate(model, packed, bg, cfg)
    return TrainResult(model=model, psnr=psnr, ssim=score, loss_start=float(loss_start), loss_end=loss_end)


def _pack_view(view: TrainView, device: torch.device) -> dict:
    image = torch.as_tensor(view.image, device=device).float() / 255.0
    mask = torch.as_tensor(view.mask, device=device).float()
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("view image must be HxWx3")
    if mask.shape != image.shape[:2]:
        raise ValueError("view mask must match the image")
    return {"image": image, "mask": mask, "camera": view.camera}


@torch.no_grad()
def _evaluate(model, packed, background, cfg) -> tuple[float, float]:
    psnr_values = []
    ssim_values = []
    for view in packed:
        predicted = render_gaussians(
            model,
            view["camera"],
            background,
            tile=cfg.tile,
            max_per_tile=cfg.max_per_tile,
            near=cfg.near,
        ).clamp(0.0, 1.0)
        psnr_values.append(float(masked_psnr(predicted, view["image"], view["mask"])))
        ssim_values.append(float(ssim(predicted, view["image"])))
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))
