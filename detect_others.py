from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2

from image_io import read_image, write_image
from report_utils import build_contact_sheet
from sea_detector import DetectorConfig, SeaSurfaceDetector

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def reset_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for entry in folder.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def draw_boxes(gray_img, boxes: List[Tuple[int, int, int, int]], horizon_y: int, roi_start_y: int):
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.line(vis, (0, horizon_y), (vis.shape[1] - 1, horizon_y), (255, 128, 0), 1)
    cv2.line(vis, (0, roi_start_y), (vis.shape[1] - 1, roi_start_y), (0, 255, 255), 1)
    cv2.putText(
        vis,
        f"preds={len(boxes)}",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return vis


def detect_folder(input_dir: Path, output_dir: Path, cfg: DetectorConfig) -> dict[str, object]:
    """Run detection on an unlabeled image folder and save visual results."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    reset_dir(output_dir)
    images = list_images(input_dir)
    if not images:
        raise RuntimeError(f"No images found in {input_dir}")

    detector = SeaSurfaceDetector(cfg)
    processed = 0
    failed: List[str] = []
    preview_items: List[tuple[Path, str]] = []
    pred_counts: List[int] = []

    for idx, img_path in enumerate(images):
        img = read_image(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            failed.append(img_path.name)
            continue

        norm = detector.initialize_image(img)
        mask, meta = detector.detect(norm)
        boxes = detector.mask_to_boxes(mask)
        vis = draw_boxes(norm, boxes, meta["horizon_y"], meta["roi_start_y"])

        out_name = f"result_{idx:04d}_{img_path.name}"
        out_path = output_dir / out_name
        write_image(out_path, vis)
        if len(preview_items) < 12:
            preview_items.append((out_path, f"{img_path.name[:24]}  preds {len(boxes)}"))
        pred_counts.append(len(boxes))
        processed += 1

    preview_path = output_dir / "preview.png"
    build_contact_sheet(preview_items, preview_path, f"{input_dir.name} preview", cols=3)
    avg_pred = (sum(pred_counts) / len(pred_counts)) if pred_counts else 0.0

    summary_lines = [
        "# 检测摘要",
        "",
        f"- 输入目录：`{input_dir.name}`，共 `{len(images)}` 张，成功处理 `{processed}` 张。",
        f"- 检测结果：读取失败 `{len(failed)}` 张，平均每张输出 `{avg_pred:.2f}` 个检测框。",
        "- 输出内容：`preview.png` 为预览图，`result_*.png` 为单张检测结果图。",
        (
            f"- 参数设置：自动海天线 `{cfg.auto_horizon}`，搜索带高度 "
            f"`{cfg.horizon_band_height}` px，最多保留 `{cfg.max_boxes}` 个检测框。"
        ),
    ]
    if failed:
        summary_lines.extend(["", "## 读取失败文件", ""])
        summary_lines.extend([f"- `{name}`" for name in failed[:30]])

    summary_path = output_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    return {
        "dataset_dir": str(input_dir),
        "mode": "detect",
        "images": len(images),
        "processed": processed,
        "failed": len(failed),
        "avg_pred_per_frame": avg_pred,
        "artifacts": {
            "summary": str(summary_path),
            "preview": str(preview_path),
            "visualizations": str(output_dir),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    default_cfg = DetectorConfig()
    p = argparse.ArgumentParser(description="Detect all images in an unlabeled image folder.")
    p.add_argument("--input", type=str, default="data/raw/others", help="input image folder")
    p.add_argument("--output", type=str, default="outputs/detect/others_results", help="output folder")
    p.add_argument(
        "--clear-output",
        action="store_true",
        help="deprecated: output is now always cleared before run",
    )
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
    input_dir = resolve_path(args.input)
    output_dir = resolve_path(args.output)

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
    result = detect_folder(input_dir, output_dir, cfg)
    artifacts = result["artifacts"]

    print("Detection complete")
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Images:  {result['processed']}/{result['images']} processed")
    print(f"Failed:  {result['failed']}")
    print(f"Preview: {artifacts['preview']}")
    print(f"Report:  {artifacts['summary']}")


if __name__ == "__main__":
    main()
