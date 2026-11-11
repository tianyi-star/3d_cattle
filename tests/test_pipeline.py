import json

import numpy as np

from cattle3d.config import PipelineConfig
from cattle3d.pipeline import reconstruct
from cattle3d.synthetic import HEAD_CENTER, write_dataset


def test_pipeline_reconstructs_synthetic_mmcows_head(tmp_path):
    root = write_dataset(tmp_path / "visual_data", seed=2)
    cfg = PipelineConfig.for_demo()
    cfg.turntable_frames = 4
    cfg.turntable_size = 64
    cfg.train.iterations = 8
    cfg.train.lambda_ssim = 0.0
    cfg.init_points = 80
    cfg.train_long_side = 56
    out = tmp_path / "run"
    meta = reconstruct(root, out, cfg)
    assert meta["cow_id"] == 3
    assert meta["num_views"] == 4
    assert (out / "gaussians.ply").is_file()
    assert (out / "preview" / "compare.png").is_file()
    assert (out / "preview" / "turntable.mp4").is_file()
    assert meta["loss_end"] < meta["loss_start"]
    origin = np.asarray(meta["origin_m"])
    assert np.linalg.norm(origin - HEAD_CENTER) < 0.1
    saved = json.loads((out / "meta.json").read_text())
    assert saved["source"] == "mmcows"
