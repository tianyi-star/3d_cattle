"""Command line for MmCows cow-head reconstruction."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from cattle3d.config import PipelineConfig, load_config
from cattle3d.export import load_gaussians_ply, render_turntable
from cattle3d.pipeline import reconstruct
from cattle3d.synthetic import write_dataset


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="cattle3d", description="Rebuild a cow head from MmCows multi-view frames with 3D Gaussian Splatting.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="Reconstruct a synthetic MmCows head. Does not download the dataset.")
    demo.add_argument("--out", required=True, help="Directory for the synthetic capture and the reconstruction")
    demo.add_argument("--config", default=None, help="Optional YAML merged on top of the demo settings")

    rec = sub.add_parser("reconstruct", help="Reconstruct one cow from an unpacked MmCows visual_data directory")
    rec.add_argument("--visual-data", required=True)
    rec.add_argument("--out", required=True)
    rec.add_argument("--cow", type=int, required=True, help="Cow id, 1-16")
    rec.add_argument("--date", default="0725", help="MMDD folder, default 0725 (the annotated day)")
    rec.add_argument("--config", default=None)
    rec.add_argument("--sensor-data", default=None, help="Optional sensor_data root, used for behavior CSVs")

    show = sub.add_parser("render", help="Render an orbit video from gaussians.ply")
    show.add_argument("--ply", required=True)
    show.add_argument("--out", required=True)
    show.add_argument("--frames", type=int, default=24)
    show.add_argument("--size", type=int, default=192)
    show.add_argument("--radius", type=float, default=2.4)

    args = parser.parse_args(argv)
    if args.command == "demo":
        _demo(args)
    elif args.command == "reconstruct":
        _reconstruct(args)
    elif args.command == "render":
        model = load_gaussians_ply(args.ply)
        render_turntable(model, args.out, frames=args.frames, size=args.size, radius=args.radius)
        print(args.out)
    else:
        parser.error(f"unknown command {args.command}")


def _demo(args) -> None:
    out = Path(args.out)
    cfg = PipelineConfig.for_demo()
    if args.config:
        cfg = load_config(args.config, base=cfg)
    root = write_dataset(out / "synthetic_mmcows", seed=cfg.seed)
    meta = reconstruct(root, out / "reconstruction", cfg)
    _print_meta(meta)


def _reconstruct(args) -> None:
    cfg = load_config(args.config) if args.config else PipelineConfig()
    cfg.cow_id = int(args.cow)
    cfg.date = str(args.date)
    meta = reconstruct(args.visual_data, args.out, cfg, sensor_data=args.sensor_data)
    _print_meta(meta)


def _print_meta(meta: dict) -> None:
    print(
        f"cow {meta['cow_id']}  views {meta['num_views']}  "
        f"gaussians {meta['num_gaussians']}  psnr {meta['train_psnr']:.2f}"
    )
    print(f"ply {meta['ply']}")


if __name__ == "__main__":
    main()
