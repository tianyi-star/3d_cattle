"""YAML configuration for a cow-head reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path

import yaml

from cattle3d.mmcows import HEAD_BEHAVIORS
from cattle3d.train import TrainConfig


@dataclass
class PipelineConfig:
    date: str = "0725"
    cow_id: int = 1
    label_set: str = "combined"
    calibration_width: int = 4480
    calibration_height: int = 2800
    min_views: int = 2
    max_frames: int = 6
    max_motion_m: float = 0.45
    allowed_behaviors: tuple[int, ...] | None = HEAD_BEHAVIORS
    head_size_m: tuple[float, float, float] = (0.9, 0.9, 0.7)
    head_z_m: float = 1.45
    train_long_side: int = 160
    init_points: int = 700
    background: tuple[float, float, float] = (0.0, 0.0, 0.0)
    turntable_frames: int = 24
    turntable_size: int = 192
    seed: int = 0
    train: TrainConfig = field(default_factory=TrainConfig)

    @classmethod
    def for_demo(cls) -> PipelineConfig:
        cfg = cls(date="0725", cow_id=3, max_frames=4, train_long_side=72, init_points=160, turntable_frames=8, turntable_size=96)
        cfg.head_size_m = (0.7, 0.7, 0.62)
        cfg.train.iterations = 25
        cfg.train.max_per_tile = 1024
        cfg.train.log_every = 5
        cfg.train.seed = 0
        return cfg


def load_config(path: str | Path, base: PipelineConfig | None = None) -> PipelineConfig:
    cfg = base or PipelineConfig()
    payload = yaml.safe_load(Path(path).read_text()) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    _merge(cfg, payload)
    return cfg


def _merge(obj, payload: dict) -> None:
    for key, value in payload.items():
        if not hasattr(obj, key):
            raise ValueError(f"unknown config field {key}")
        current = getattr(obj, key)
        if is_dataclass(current) and isinstance(value, dict):
            _merge(current, value)
            continue
        if key in {"head_size_m", "background", "allowed_behaviors"} and isinstance(value, list):
            value = tuple(value)
        setattr(obj, key, value)
