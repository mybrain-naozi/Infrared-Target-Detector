from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable

from dataset_io import list_images
from detect_others import detect_folder
from evaluate import evaluate_folder
from sea_detector import DetectorConfig

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def has_images(folder: Path) -> bool:
    return folder.is_dir() and bool(list_images(folder))


def discover_dataset_dirs(root: Path) -> list[Path]:
    """Return image folders to process.

    A direct image folder is treated as one dataset. If the given path is a
    parent folder such as data/raw, all immediate child folders containing
    images are processed.
    """
    if has_images(root):
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"Dataset path not found: {root}")
    datasets = [p for p in sorted(root.iterdir(), key=lambda x: x.name) if p.is_dir() and has_images(p)]
    if not datasets:
        raise RuntimeError(f"No image datasets found under {root}")
    return datasets


def build_config(args: argparse.Namespace) -> DetectorConfig:
    return DetectorConfig(
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


def build_parser() -> argparse.ArgumentParser:
    default_cfg = DetectorConfig()
    p = argparse.ArgumentParser(description="Sea-surface infrared target detection pipeline.")
    p.add_argument(
        "--dataset",
        type=str,
        default="data/raw",
        help="dataset folder or parent folder containing multiple dataset folders",
    )
    p.add_argument("--output", type=str, default="outputs/runs/all_datasets", help="output folder")
    p.add_argument("--iou", type=float, default=0.1, help="IoU threshold for labeled evaluation")
    p.add_argument("--save-steps", action="store_true", help="save step-by-step sheets for labeled datasets")
    p.add_argument("--step-limit", type=int, default=12, help="number of labeled frames for step sheets; 0 means all")
    p.add_argument("--paper-step-count", type=int, default=0, help="export step images for top hit labeled frames")

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


def reset_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for entry in folder.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def dataset_output_dir(output_root: Path, datasets: list[Path], dataset_dir: Path) -> Path:
    return output_root / dataset_dir.name if len(datasets) > 1 else output_root


def run_one_dataset(
    dataset_dir: Path,
    output_dir: Path,
    cfg: DetectorConfig,
    args: argparse.Namespace,
) -> dict[str, object]:
    if (dataset_dir / "groundtruth.txt").exists():
        result = evaluate_folder(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            cfg=cfg,
            iou_thr=args.iou,
            save_steps=args.save_steps,
            step_limit=args.step_limit,
            paper_step_count=args.paper_step_count,
        )
        result["mode"] = "evaluate"
        return result
    return detect_folder(input_dir=dataset_dir, output_dir=output_dir, cfg=cfg)


def write_aggregate_summary(output_root: Path, results: Iterable[dict[str, object]]) -> Path:
    rows = []
    lines = [
        "# 全部数据集运行汇总",
        "",
        "| 数据集 | 类型 | 帧数/图片数 | Precision | Recall | F1 | 输出目录 |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for result in results:
        dataset_name = Path(str(result["dataset_dir"])).name
        mode = str(result.get("mode", ""))
        artifacts = result.get("artifacts", {})
        if mode == "evaluate":
            metrics = result["metrics"]  # type: ignore[index]
            row = (
                dataset_name,
                "有标注评测",
                int(metrics["frames"]),  # type: ignore[index]
                float(metrics["precision"]),  # type: ignore[index]
                float(metrics["recall"]),  # type: ignore[index]
                float(metrics["f1"]),  # type: ignore[index]
                str(artifacts.get("summary", "")),  # type: ignore[union-attr]
            )
            lines.append(
                f"| {row[0]} | {row[1]} | {row[2]} | {row[3]:.4f} | {row[4]:.4f} | {row[5]:.4f} | `{row[6]}` |"
            )
        else:
            row = (
                dataset_name,
                "无标注检测",
                int(result["processed"]),
                "",
                "",
                "",
                str(artifacts.get("summary", "")),  # type: ignore[union-attr]
            )
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} |  |  |  | `{row[6]}` |")
        rows.append(row)

    summary_path = output_root / "all_datasets_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def print_result(result: dict[str, object]) -> None:
    dataset_dir = Path(str(result["dataset_dir"]))
    artifacts = result.get("artifacts", {})
    print(f"\nDataset: {dataset_dir}")
    if result.get("mode") == "evaluate":
        metrics = result["metrics"]  # type: ignore[index]
        print("Mode:    labeled evaluation")
        print(
            "Metrics: "
            f"precision={metrics['precision']:.4f}, "  # type: ignore[index]
            f"recall={metrics['recall']:.4f}, "  # type: ignore[index]
            f"f1={metrics['f1']:.4f}, "  # type: ignore[index]
            f"tp={metrics['tp']}, fp={metrics['fp']}, fn={metrics['fn']}"  # type: ignore[index]
        )
        print(f"Summary: {artifacts['summary']}")  # type: ignore[index]
        print(f"Overview:{artifacts['overview']}")  # type: ignore[index]
    else:
        print("Mode:    unlabeled detection")
        print(f"Images:  {result['processed']}/{result['images']} processed")
        print(f"Failed:  {result['failed']}")
        print(f"Summary: {artifacts['summary']}")  # type: ignore[index]
        print(f"Preview: {artifacts['preview']}")  # type: ignore[index]


def main() -> None:
    args = build_parser().parse_args()
    cfg = build_config(args)
    dataset_root = resolve_path(args.dataset)
    output_root = resolve_path(args.output)
    datasets = discover_dataset_dirs(dataset_root)

    if len(datasets) > 1:
        reset_dir(output_root)

    results = []
    for dataset_dir in datasets:
        out_dir = dataset_output_dir(output_root, datasets, dataset_dir)
        result = run_one_dataset(dataset_dir, out_dir, cfg, args)
        results.append(result)
        print_result(result)

    if len(results) > 1:
        summary_path = write_aggregate_summary(output_root, results)
        print(f"\nAll datasets complete: {summary_path}")
    else:
        print("\nPipeline complete")


if __name__ == "__main__":
    main()
