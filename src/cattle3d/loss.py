"""Masked photometric loss for a cow-head crop."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def gaussian_window(window: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(window, device=device, dtype=dtype) - window // 2
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel_2d = kernel[:, None] * kernel[None, :]
    return kernel_2d.expand(channels, 1, window, window).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window: int = 11) -> torch.Tensor:
    """Mean SSIM of two HxWx3 images in ``[0, 1]``."""
    if min(pred.shape[0], pred.shape[1]) < window:
        return pred.new_zeros(())
    x = pred.permute(2, 0, 1).unsqueeze(0)
    y = target.permute(2, 0, 1).unsqueeze(0)
    kernel = gaussian_window(window, 1.5, x.shape[1], pred.device, pred.dtype)
    pad = window // 2
    mu_x = F.conv2d(x, kernel, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y, kernel, padding=pad, groups=x.shape[1])
    sigma_x = F.conv2d(x * x, kernel, padding=pad, groups=x.shape[1]) - mu_x * mu_x
    sigma_y = F.conv2d(y * y, kernel, padding=pad, groups=x.shape[1]) - mu_y * mu_y
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=x.shape[1]) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).mean()


def masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    residual = (pred - target).abs()
    if mask is None:
        return residual.mean()
    weight = mask.unsqueeze(-1)
    return residual.mul(weight).sum() / weight.sum().clamp(min=1.0) / 3.0


def photometric_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None,
    lambda_ssim: float,
    background: torch.Tensor,
) -> torch.Tensor:
    l1 = masked_l1(pred, target, mask)
    if lambda_ssim <= 0 or min(pred.shape[0], pred.shape[1]) < 11:
        return l1
    if mask is None:
        score = ssim(pred, target)
    else:
        weight = mask.unsqueeze(-1)
        bg = background.to(device=pred.device, dtype=pred.dtype).view(1, 1, 3)
        pred_c = pred * weight + bg * (1.0 - weight)
        target_c = target * weight + bg * (1.0 - weight)
        score = ssim(pred_c, target_c)
    return (1.0 - lambda_ssim) * l1 + lambda_ssim * (1.0 - score)


def masked_psnr(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    error = (pred - target) ** 2
    if mask is None:
        mse = error.mean()
    else:
        weight = mask.unsqueeze(-1)
        mse = error.mul(weight).sum() / weight.sum().clamp(min=1.0) / 3.0
    return -10.0 * torch.log10(mse.clamp(min=1e-10))
