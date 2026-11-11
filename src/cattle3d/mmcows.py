"""Reader for the MmCows visual layout.

Expected tree, matching the Neis Lab release (Vu et al., NeurIPS 2024):

```
visual_data/
  images/{MMDD}/cam_{1..4}/{unix}_{HH-MM-SS}.jpg
  labels/combined/{MMDD}/cam_{1..4}/{unix}_{HH-MM-SS}.txt
  proj_mat/{MMDD}/proj_mat_cam{1..4}.xml
  behavior_labels/individual/C{id}_{MMDD}.csv   # optional
```

Each label row is ``cow_id cx cy w h`` with the box center and size normalized
to the image. Projection XML holds a 3x4 ``projection_matrix`` in centimeters
and the derived ``camera_matrix``.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cattle3d.cameras import Camera
from cattle3d.projection import camera_from_projection, read_projection_xml

BEHAVIOR_NAMES = {
    0: "unknown",
    1: "walking",
    2: "standing",
    3: "feeding_head_up",
    4: "feeding_head_down",
    5: "licking",
    6: "drinking",
    7: "lying",
}

# Head stays near one place in these isometric views. Walking moves between
# the 15-second samples, and lying hides the head in the stall.
HEAD_BEHAVIORS = (2, 3, 4, 5, 6)

CAMERA_NAMES = ("cam_1", "cam_2", "cam_3", "cam_4")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


@dataclass(frozen=True)
class Detection:
    cow_id: int
    cx: float
    cy: float
    w: float
    h: float
    behavior: int | None = None

    def pixel_center(self, width: int, height: int) -> tuple[float, float]:
        return self.cx * width, self.cy * height

    def pixel_xyxy(self, width: int, height: int) -> tuple[float, float, float, float]:
        cx, cy = self.pixel_center(width, height)
        pw, ph = self.w * width, self.h * height
        return cx - pw / 2.0, cy - ph / 2.0, cx + pw / 2.0, cy + ph / 2.0


@dataclass(frozen=True)
class FrameRecord:
    timestamp: int
    camera_index: int
    image_path: Path
    label_path: Path | None
    stem: str


def parse_timestamp(stem: str) -> int | None:
    head = stem.split("_", 1)[0]
    if not head.isdigit():
        return None
    return int(head)


def read_detections(path: str | Path) -> list[Detection]:
    text = Path(path).read_text().strip()
    if not text:
        return []
    rows: list[Detection] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        behavior = int(float(parts[5])) if len(parts) >= 6 else None
        rows.append(
            Detection(
                cow_id=int(float(parts[0])),
                cx=float(parts[1]),
                cy=float(parts[2]),
                w=float(parts[3]),
                h=float(parts[4]),
                behavior=behavior,
            )
        )
    return rows


def write_detections(path: str | Path, detections: list[Detection]) -> None:
    lines = []
    for det in detections:
        line = f"{det.cow_id} {det.cx:.6f} {det.cy:.6f} {det.w:.6f} {det.h:.6f}"
        if det.behavior is not None:
            line += f" {det.behavior}"
        lines.append(line)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(lines) + ("\n" if lines else ""))


def read_behavior_csv(path: str | Path) -> dict[int, int]:
    """Map unix timestamp to behavior id. Accepts the MmCows individual CSV."""
    table: dict[int, int] = {}
    with Path(path).open(newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        has_header = "timestamp" in sample.lower() or "behavior" in sample.lower() or "behaviour" in sample.lower()
        reader = csv.DictReader(handle) if has_header else None
        if reader is not None and reader.fieldnames:
            fields = {name.lower(): name for name in reader.fieldnames}
            time_key = fields.get("timestamp")
            beh_key = fields.get("behavior") or fields.get("behaviour")
            if time_key is None or beh_key is None:
                raise ValueError(f"behavior CSV {path} needs timestamp and behavior columns")
            for row in reader:
                table[int(float(row[time_key]))] = int(float(row[beh_key]))
            return table
    with Path(path).open(newline="") as handle:
        for row in csv.reader(handle):
            nums = []
            for cell in row:
                try:
                    nums.append(float(cell))
                except ValueError:
                    continue
            if len(nums) >= 2:
                table[int(nums[-2])] = int(nums[-1])
    return table


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


class MmCowsVision:
    """Index of one unpacked ``visual_data`` directory."""

    def __init__(self, root: str | Path, label_set: str = "combined"):
        self.root = Path(root)
        self.label_set = label_set
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"MmCows visual directory not found: {self.root}. "
                "Download visual_data.zip from https://huggingface.co/datasets/neis-lab/mmcows "
                "and unzip it so images/, labels/, and proj_mat/ sit in this folder."
            )

    def calibration_size(self, date: str, default: tuple[int, int] = (4480, 2800)) -> tuple[int, int]:
        path = self.root / "proj_mat" / _date(date) / "calibration.json"
        if not path.is_file():
            return default
        payload = json.loads(path.read_text())
        return int(payload["width"]), int(payload["height"])

    def projection_path(self, date: str, camera_index: int) -> Path:
        return self.root / "proj_mat" / _date(date) / f"proj_mat_cam{camera_index + 1}.xml"

    def load_cameras(
        self,
        date: str,
        image_size: tuple[int, int] | None = None,
        calibration_size: tuple[int, int] | None = None,
    ) -> list[Camera]:
        """Load the four barn cameras in meters.

        ``image_size`` is the resolution of the frames actually read. Intrinsics
        are scaled from the calibration resolution onto that size.
        """
        date = _date(date)
        calib_w, calib_h = calibration_size or self.calibration_size(date)
        width, height = image_size or (calib_w, calib_h)
        cameras: list[Camera] = []
        missing = []
        for index in range(4):
            path = self.projection_path(date, index)
            if not path.is_file():
                missing.append(str(path))
                continue
            projection, _stored_k = read_projection_xml(path)
            cameras.append(
                camera_from_projection(
                    projection,
                    width=width,
                    height=height,
                    calibration_width=calib_w,
                    calibration_height=calib_h,
                    camera_id=index + 1,
                )
            )
        if missing:
            raise FileNotFoundError(
                "Missing MmCows projection matrices:\n  " + "\n  ".join(missing)
            )
        return cameras

    def label_root(self, date: str) -> Path:
        date = _date(date)
        candidates = [
            self.root / "labels" / self.label_set / date,
            self.root / "labels" / date,
            self.root / "labels" / "combined" / date,
            self.root / "labels" / "nonlying" / date,
        ]
        for candidate in candidates:
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"No label folder for {date} under {self.root / 'labels'}. "
            "Expected labels/combined/{date}/cam_1/*.txt from visual_data.zip."
        )

    def iter_frames(self, date: str) -> list[FrameRecord]:
        date = _date(date)
        image_root = self.root / "images" / date
        if not image_root.is_dir():
            raise FileNotFoundError(f"No images for {date} at {image_root}")
        label_root = self.label_root(date)
        frames: list[FrameRecord] = []
        for index, name in enumerate(CAMERA_NAMES):
            image_dir = image_root / name
            if not image_dir.is_dir():
                continue
            label_dir = label_root / name
            for image_path in sorted(image_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                timestamp = parse_timestamp(image_path.stem)
                if timestamp is None:
                    continue
                label_path = label_dir / f"{image_path.stem}.txt"
                frames.append(
                    FrameRecord(
                        timestamp=timestamp,
                        camera_index=index,
                        image_path=image_path,
                        label_path=label_path if label_path.is_file() else None,
                        stem=image_path.stem,
                    )
                )
        return frames

    def detections_for(self, frame: FrameRecord, cow_id: int) -> Detection | None:
        if frame.label_path is None:
            return None
        for detection in read_detections(frame.label_path):
            if detection.cow_id == int(cow_id):
                return detection
        return None

    def behavior_table(self, date: str, cow_id: int, extra_roots: list[Path] | None = None) -> dict[int, int]:
        date = _date(date)
        name = f"C{int(cow_id):02d}_{date}.csv"
        roots = [self.root, *(extra_roots or [])]
        relative = [
            Path("behavior_labels") / "individual" / name,
            Path("behavior_labels") / "individual" / f"C{int(cow_id):02d}" / name,
        ]
        for root in roots:
            for rel in relative:
                path = root / rel
                if path.is_file():
                    return read_behavior_csv(path)
        return {}


def _date(date: str) -> str:
    text = str(date).strip()
    if text.isdigit():
        return text.zfill(4)
    return text
