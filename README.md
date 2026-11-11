# 3d_cattle

Rebuild a dairy cow's head from [MmCows](https://github.com/neis-lab/mmcows) multi-view frames with 3D Gaussian Splatting.

MmCows (Vu et al., NeurIPS 2024) is a barn dataset: four calibrated isometric cameras, cow ids, bounding boxes, and behavior labels for 16 Holstein cows. This project reads that layout, picks a moment when one cow is standing or feeding and nearly still, places a head-sized volume on the multi-view body point, and fits a 3D Gaussian splat to the masked crops.

The rasterizer is a pure PyTorch tile renderer, so it runs on CPU. The exported `gaussians.ply` uses the standard 3D Gaussian Splatting attribute layout (log-scale, logit opacity, spherical-harmonic DC color).

Our code is Apache-2.0. The MmCows **data** is CC BY-NC-SA 4.0, is not included here, and is not licensed for commercial use.

## Layout this reader expects

Unzip `visual_data.zip` (and optionally the behavior CSVs from `sensor_data.zip`) so the tree matches the MmCows release:

```
visual_data/
  images/0725/cam_1/1721902638_14-57-18.jpg
  images/0725/cam_2/...
  labels/combined/0725/cam_1/1721902638_14-57-18.txt
  proj_mat/0725/proj_mat_cam1.xml
  behavior_labels/individual/C01_0725.csv    # optional
```

Label rows are `cow_id cx cy w h` with the box normalized to the image. Projection XML is the OpenCV 3x4 matrix in centimeters at 4480x2800. Behavior ids follow the MmCows definitions: 0 unknown, 1 walking, 2 standing, 3 feeding head up, 4 feeding head down, 5 licking, 6 drinking, 7 lying.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
```

With a CUDA build of PyTorch the same code uses the GPU.

## Demo without the dataset

```bash
cattle3d demo --out runs/demo
```

This writes a small MmCows-format capture of a stylized head (`runs/demo/synthetic_mmcows`) and a reconstruction (`runs/demo/reconstruction`).

## Reconstruct one cow

```bash
cattle3d reconstruct \
  --visual-data /path/to/visual_data \
  --sensor-data /path/to/sensor_data \
  --cow 1 \
  --date 0725 \
  --config configs/mmcows_face.yaml \
  --out runs/cow01
```

`--sensor-data` is only used to read `behavior_labels/individual/C01_0725.csv`. If that file is already under `visual_data`, the flag can be omitted. Frames labeled walking or lying are skipped. Among the remaining timestamps the run keeps a short window whose body position moves less than `max_motion_m` (default 0.45 m).

```bash
cattle3d render --ply runs/cow01/gaussians.ply --out runs/cow01/turntable.mp4
```

## Outputs

`runs/cow01/`

| File | Contents |
| --- | --- |
| `gaussians.ply` | Head splat in meters. The origin is the estimated head center and +Z is barn up. |
| `meta.json` | Cow id, timestamps, behaviors, barn-frame origin, PSNR, loss. |
| `preview/compare.png` | Cropped camera view beside the trained render. |
| `preview/turntable.mp4` | Orbit of the head. |

`meta.json` field `origin_m` maps a local point back to the barn: `X_barn = X_local + origin_m`.

## Method

1. Factor each MmCows projection matrix into `K` and `[R|t]`, convert centimeters to meters, and scale `K` if the frames are not 4480x2800.
2. For each timestamp, shoot a ray through every camera's bbox center and take the point closest to those rays. That is the same construction MmCows uses for visual localization.
3. Put a head-sized box on that point (lifting it into the head band when the rays meet below 1 m, which is typical of a full-body box).
4. Crop each view to the projection of the box, intersected with the cow's bounding box, and train 3D Gaussians with a masked L1 + SSIM loss. The floor, the stalls, and the other cows stay outside the mask.

## Limits

The barn cameras look down into the pen. They do not see a close-up face, and a 15-second sample is only rigid when the cow stays put, which is why walking and lying are excluded. The head box is a geometric prior, not a landmark detector, so the splat covers the head and whatever shoulders fall inside that volume. A phone video orbiting a haltered head is outside this dataset and is not what this pipeline consumes.

## Tests

```bash
pytest
```

## Citation

Hien Vu, Omkar Prabhune, Unmesh Raskar, Dimuth Panditharatne, Hanwook Chung, Christopher Choi, and Younghyun Kim. "MmCows: A Multimodal Dataset for Dairy Cattle Monitoring." NeurIPS Datasets and Benchmarks, 2024. https://github.com/neis-lab/mmcows
