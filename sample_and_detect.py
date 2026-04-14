from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List

import cv2

from image_io import read_image, write_image
from sea_detector import DetectorConfig, SeaSurfaceDetector


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def collect_images(folder: Path) -> List[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])


def reset_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for entry in folder.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def draw_boxes(gray_img, boxes, horizon_y: int, roi_start_y: int):
    vis = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    for x1, y1, x2, y2 in boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 1)
    cv2.line(vis, (0, horizon_y), (vis.shape[1] - 1, horizon_y), (255, 128, 0), 1)
    cv2.line(vis, (0, roi_start_y), (vis.shape[1] - 1, roi_start_y), (0, 255, 255), 1)
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample images into dataset, then run detection.")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["data/raw/others", "data/raw/boat2", "data/raw/data1"],
        help="Source folders for random sampling.",
    )
    parser.add_argument("--dataset-dir", default="data/sampled/dataset", help="Output sampled dataset folder.")
    parser.add_argument("--detect-dir", default="outputs/detect/sampled_results", help="Output detection result folder.")
    parser.add_argument("--total-samples", type=int, default=60, help="Total number of samples.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    dataset_dir = resolve_path(args.dataset_dir)
    detect_dir = resolve_path(args.detect_dir)
    reset_dir(dataset_dir)
    reset_dir(detect_dir)

    all_images: List[Path] = []
    for src in args.sources:
        src_dir = resolve_path(src)
        imgs = collect_images(src_dir)
        all_images.extend(imgs)
        print(f"[INFO] source={src_dir} images={len(imgs)}")

    if not all_images:
        raise RuntimeError("No images found in given source folders.")

    random.seed(args.seed)
    sample_n = min(args.total_samples, len(all_images))
    selected = random.sample(all_images, sample_n)

    copied_paths: List[Path] = []
    for idx, src in enumerate(selected):
        dst_name = f"{idx:04d}_{src.parent.name}_{src.name}"
        dst = dataset_dir / dst_name
        shutil.copy2(src, dst)
        copied_paths.append(dst)
    print(f"[INFO] copied {len(copied_paths)} images -> {dataset_dir}")

    detector = SeaSurfaceDetector(DetectorConfig())
    for img_path in copied_paths:
        img = read_image(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        norm = detector.initialize_image(img)
        mask, meta = detector.detect(norm)
        boxes = detector.mask_to_boxes(mask)
        vis = draw_boxes(norm, boxes, meta["horizon_y"], meta["roi_start_y"])
        out_path = detect_dir / f"result_{img_path.name}"
        write_image(out_path, vis)
    print(f"[INFO] detection finished -> {detect_dir}")


if __name__ == "__main__":
    main()
