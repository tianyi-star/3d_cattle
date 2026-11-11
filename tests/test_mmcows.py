import json

import numpy as np

from cattle3d.localize import localize_bbox_centers
from cattle3d.mmcows import BEHAVIOR_NAMES, MmCowsVision, read_behavior_csv
from cattle3d.synthetic import DATE, HEAD_CENTER, write_dataset


def test_behavior_names_match_mmcows_definitions():
    assert BEHAVIOR_NAMES[2] == "standing"
    assert BEHAVIOR_NAMES[3] == "feeding_head_up"
    assert BEHAVIOR_NAMES[4] == "feeding_head_down"
    assert BEHAVIOR_NAMES[7] == "lying"


def test_synthetic_dataset_roundtrip_localizes_the_head(tmp_path):
    root = write_dataset(tmp_path / "visual_data")
    dataset = MmCowsVision(root)
    width, height = dataset.calibration_size(DATE)
    assert (width, height) == (192, 120)
    cameras = dataset.load_cameras(DATE)
    assert len(cameras) == 4
    frames = dataset.iter_frames(DATE)
    assert len(frames) == 4
    pixels = []
    used = []
    for frame in frames:
        detection = dataset.detections_for(frame, 3)
        assert detection is not None
        camera = cameras[frame.camera_index]
        pixels.append(detection.pixel_center(camera.width, camera.height))
        used.append(camera)
    found = localize_bbox_centers(used, pixels)
    assert np.linalg.norm(found - HEAD_CENTER) < 0.08
    behavior = read_behavior_csv(root / "behavior_labels" / "individual" / "C03_0725.csv")
    assert behavior[json.loads((root / "source.json").read_text())["timestamp"]] == 2
    loaded = dataset.behavior_table(DATE, 3)
    assert loaded == behavior
