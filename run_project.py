from __future__ import annotations

import argparse
from pathlib import Path

from evaluate import evaluate_folder
from sea_detector import DetectorConfig

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def build_parser() -> argparse.ArgumentParser:
    default_cfg = DetectorConfig()
    p = argparse.ArgumentParser(description="Sea-surface small target detection pipeline.")
    p.add_argument("--dataset", type=str, default="data/raw/boat2", help="dataset folder path")
    p.add_argument("--output", type=str, default="outputs/runs/full_pipeline", help="output folder")
    p.add_argument("--iou", type=float, default=0.1, help="IoU threshold for TP")

    p.add_argument("--roi-start-y", type=int, default=default_cfg.roi_start_y)
    p.add_argument("--peak-kernel", type=int, default=default_cfg.peak_kernel)
    p.add_argument("--l-factor", type=float, default=default_cfg.l_factor)
    p.add_argument("--min-area", type=int, default=default_cfg.min_area)
    p.add_argument("--max-area", type=int, default=default_cfg.max_area)
    p.add_argument("--opening-kernel", type=int, default=default_cfg.opening_kernel)
    p.add_argument("--merge-margin", type=int, default=default_cfg.merge_margin)
    p.add_argument("--horizon-band-height", type=int, default=default_cfg.horizon_band_height)
    p.add_argument("--horizon-offset", type=int, default=default_cfg.horizon_offset)
    p.add_argument("--max-boxes", type=int, default=default_cfg.max_boxes)
    p.add_argument("--disable-auto-horizon", action="store_true", help="use a fixed ROI instead of horizon search")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = DetectorConfig(
        roi_start_y=args.roi_start_y,
        peak_kernel=args.peak_kernel,
        l_factor=args.l_factor,
        min_area=args.min_area,
        max_area=args.max_area,
        opening_kernel=args.opening_kernel,
        merge_margin=args.merge_margin,
        horizon_band_height=args.horizon_band_height,
        horizon_offset=args.horizon_offset,
        max_boxes=args.max_boxes,
        auto_horizon=not args.disable_auto_horizon,
    )

    dataset_dir = resolve_path(args.dataset)
    output_dir = resolve_path(args.output)
    result = evaluate_folder(dataset_dir=dataset_dir, output_dir=output_dir, cfg=cfg, iou_thr=args.iou)
    metrics = result["metrics"]
    artifacts = result["artifacts"]

    print("Evaluation complete")
    print(f"Dataset: {dataset_dir}")
    print(f"Output:  {output_dir}")
    print(
        "Metrics: "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}, "
        f"tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']}"
    )
    print(f"Summary report: {artifacts['summary']}")
    print(f"Overview image: {artifacts['overview']}")
    print(f"Best hits:      {artifacts['best_hits']}")
    print(f"Misses:         {artifacts['misses']}")
    print(f"False alarms:   {artifacts['false_alarms']}")
    print(f"All frames:     {artifacts['visualizations']}")


if __name__ == "__main__":
    main()
