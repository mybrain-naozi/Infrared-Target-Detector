from __future__ import annotations

import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from dataset_io import align_lengths, list_images, read_attributes, read_groundtruth
from image_io import read_image, write_image
from report_utils import build_contact_sheet
from sea_detector import DetectorConfig, SeaSurfaceDetector

Box = Tuple[int, int, int, int]


@dataclass
class EvalMetrics:
    frames: int
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    mean_iou_tp: float
    avg_pred_per_frame: float
    frames_with_no_pred: int


def iou(a: Box, b: Box) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    iw = max(0, x2 - x1 + 1)
    ih = max(0, y2 - y1 + 1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    area_b = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    return inter / float(area_a + area_b - inter)


def best_iou(preds: Sequence[Box], gt: Box) -> float:
    if not preds:
        return 0.0
    return max(iou(p, gt) for p in preds)


def reset_dir(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for entry in folder.iterdir():
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _status_for_frame(hit: bool, pred_count: int) -> str:
    if hit:
        return "hit"
    if pred_count > 0:
        return "false_alarm"
    return "miss"


def _as_display_image(image: np.ndarray, normalize: bool = False) -> np.ndarray:
    if image.ndim == 2:
        if normalize and image.max() > image.min():
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return image.copy()


def _draw_guides(panel: np.ndarray, meta: Dict[str, int]) -> np.ndarray:
    out = panel.copy()
    horizon_y = int(meta.get("horizon_y", -1))
    roi_start_y = int(meta.get("roi_start_y", -1))
    if 0 <= horizon_y < out.shape[0]:
        cv2.line(out, (0, horizon_y), (out.shape[1] - 1, horizon_y), (255, 128, 0), 1)
    if 0 <= roi_start_y < out.shape[0]:
        cv2.line(out, (0, roi_start_y), (out.shape[1] - 1, roi_start_y), (0, 255, 255), 1)
    return out


def _make_step_panel(
    image: np.ndarray,
    caption: str,
    meta: Dict[str, int],
    normalize: bool = False,
    thumb_size: tuple[int, int] = (320, 256),
) -> np.ndarray:
    panel = _as_display_image(image, normalize=normalize)
    panel = _draw_guides(panel, meta)
    panel = cv2.resize(panel, thumb_size, interpolation=cv2.INTER_AREA)
    canvas = np.full((thumb_size[1] + 34, thumb_size[0], 3), 245, dtype=np.uint8)
    canvas[34:, :] = panel
    cv2.putText(canvas, caption, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
    return canvas


def _write_step_sheet(
    output_path: Path,
    steps: Dict[str, np.ndarray],
    final_vis: np.ndarray,
    meta: Dict[str, int],
) -> None:
    labels = [
        ("01_original", "1 Original", False),
        ("02_row_brightness_adjusted", "2 Row adjusted", False),
        ("03_search_band", "3 Search band", False),
        ("04_local_peak_mask", "4 Local peak mask", False),
        ("05_tophat_response", "5 Top-hat response", True),
        ("06_tophat_mask", "6 Top-hat mask", False),
        ("07_fused_candidate_mask", "7 Fused mask", False),
        ("08_cleaned_candidate_mask", "8 Cleaned mask", False),
    ]
    panels = [
        _make_step_panel(steps[key], caption, meta, normalize=normalize)
        for key, caption, normalize in labels
        if key in steps
    ]
    panels.append(_make_step_panel(final_vis, "9 Final boxes", meta))

    cols = 3
    rows = int(np.ceil(len(panels) / cols))
    panel_h, panel_w = panels[0].shape[:2]
    canvas = np.full((rows * panel_h, cols * panel_w, 3), 235, dtype=np.uint8)
    for idx, panel in enumerate(panels):
        r = idx // cols
        c = idx % cols
        canvas[r * panel_h : (r + 1) * panel_h, c * panel_w : (c + 1) * panel_w] = panel
    write_image(output_path, canvas)


def _write_individual_step_images(
    output_dir: Path,
    steps: Dict[str, np.ndarray],
    final_vis: np.ndarray,
    meta: Dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = [
        ("01_original.png", "01_original", False, False),
        ("02_row_brightness_adjusted.png", "02_row_brightness_adjusted", False, False),
        ("03_horizon_and_roi.png", "01_original", False, True),
        ("04_search_band.png", "03_search_band", False, False),
        ("05_local_peak_mask.png", "04_local_peak_mask", False, False),
        ("06_tophat_response.png", "05_tophat_response", True, False),
        ("07_tophat_mask.png", "06_tophat_mask", False, False),
        ("08_fused_candidate_mask.png", "07_fused_candidate_mask", False, False),
        ("09_cleaned_candidate_mask.png", "08_cleaned_candidate_mask", False, False),
    ]
    for filename, key, normalize, draw_guides in exports:
        if key not in steps:
            continue
        image = _as_display_image(steps[key], normalize=normalize)
        if draw_guides:
            image = _draw_guides(image, meta)
        write_image(output_dir / filename, image)
    write_image(output_dir / "10_final_detection.png", final_vis)
    _write_step_sheet(output_dir / "00_step_sheet.png", steps, final_vis, meta)


def _write_best_step_exports(
    output_dir: Path,
    selected_records: Sequence[Dict[str, object]],
    images: Sequence[Path],
    gts: Sequence[Box],
    detector: SeaSurfaceDetector,
) -> str:
    paper_dir = output_dir / "paper_step_images"
    paper_dir.mkdir(parents=True, exist_ok=True)
    summary_lines = [
        "# Best step images",
        "",
        "| rank | frame | image | best_iou | folder |",
        "| --- | ---: | --- | ---: | --- |",
    ]

    for rank, record in enumerate(selected_records, start=1):
        idx = int(record["frame"])
        img_path = images[idx]
        img = read_image(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        norm = detector.initialize_image(img)
        mask, meta, steps = detector.detect_steps(norm)
        preds = detector.mask_to_boxes(mask)
        gt = gts[idx]
        biou = best_iou(preds, gt)

        final_vis = read_image(Path(str(record["vis_path"])), cv2.IMREAD_COLOR)
        if final_vis is None:
            final_vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
            for p in preds:
                cv2.rectangle(final_vis, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 1)
            cv2.rectangle(final_vis, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 1)
            final_vis = _draw_guides(final_vis, meta)

        safe_iou = f"{biou:.3f}".replace(".", "p")
        frame_dir = paper_dir / f"rank{rank}_frame{idx:04d}_{img_path.stem}_iou{safe_iou}"
        _write_individual_step_images(frame_dir, steps, final_vis, meta)
        summary_lines.append(
            f"| {rank} | {idx} | {img_path.name} | {biou:.3f} | `{frame_dir.name}` |"
        )

    (paper_dir / "selected_frames.md").write_text("\n".join(summary_lines), encoding="utf-8")
    return str(paper_dir)


def _draw_metric_bar(
    canvas: np.ndarray,
    label: str,
    value: float,
    y: int,
    color: tuple[int, int, int],
    label_x: int = 30,
    bar_x: int = 210,
) -> None:
    bar_w = 420
    cv2.putText(canvas, label, (label_x, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (40, 40, 40), 2)
    cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_w, y + 28), (225, 225, 225), -1)
    cv2.rectangle(
        canvas,
        (bar_x, y),
        (bar_x + int(bar_w * max(0.0, min(1.0, value))), y + 28),
        color,
        -1,
    )
    cv2.rectangle(canvas, (bar_x, y), (bar_x + bar_w, y + 28), (180, 180, 180), 1)
    cv2.putText(
        canvas,
        _pct(value),
        (bar_x + bar_w + 20, y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 40, 40),
        2,
    )


def _write_summary_markdown(
    output_path: Path,
    dataset_dir: Path,
    cfg: DetectorConfig,
    metrics: EvalMetrics,
    attr_summary: Dict[str, Dict[str, float]],
    artifacts: Dict[str, str],
) -> None:
    attr_text = "无"
    if attr_summary:
        attr_text = "，".join(
            [f"{key} {_pct(item['hit_rate'])}" for key, item in attr_summary.items()]
        )

    lines = [
        "# 检测简报",
        "",
        f"- 数据集：`{dataset_dir.name}`，共 `{metrics.frames}` 帧",
        f"- 指标：准确率 `{_pct(metrics.precision)}`，召回率 `{_pct(metrics.recall)}`，F1 `{_pct(metrics.f1)}`",
        f"- 统计：TP `{metrics.tp}`，FP `{metrics.fp}`，FN `{metrics.fn}`，命中帧平均 IoU `{metrics.mean_iou_tp:.3f}`",
        f"- 输出：总览图 `overview.png`，命中案例 `best_hits.png`，漏检案例 `misses.png`，误检案例 `false_alarms.png`",
        f"- 补充：属性命中率 {attr_text}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _build_overview_image(
    output_path: Path,
    dataset_dir: Path,
    metrics: EvalMetrics,
    attr_summary: Dict[str, Dict[str, float]],
    cfg: DetectorConfig,
) -> None:
    canvas = np.full((760, 1180, 3), 245, dtype=np.uint8)
    cv2.putText(canvas, "Sea Target Detection Overview", (28, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (35, 35, 35), 2)
    cv2.putText(
        canvas,
        f"Dataset: {dataset_dir.name}    Frames: {metrics.frames}",
        (30, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (70, 70, 70),
        2,
    )

    _draw_metric_bar(canvas, "Precision", metrics.precision, 120, (70, 130, 240))
    _draw_metric_bar(canvas, "Recall", metrics.recall, 170, (70, 200, 120))
    _draw_metric_bar(canvas, "F1", metrics.f1, 220, (240, 170, 70))

    cv2.putText(canvas, "Counts", (30, 305), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (40, 40, 40), 2)
    count_lines = [
        f"TP: {metrics.tp}",
        f"FP: {metrics.fp}",
        f"FN: {metrics.fn}",
        f"Mean IoU on hits: {metrics.mean_iou_tp:.3f}",
        f"Avg preds/frame: {metrics.avg_pred_per_frame:.2f}",
        f"Frames with no prediction: {metrics.frames_with_no_pred}",
    ]
    for idx, line in enumerate(count_lines):
        cv2.putText(
            canvas,
            line,
            (34, 340 + idx * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (60, 60, 60),
            2,
        )

    cv2.putText(canvas, "Attribute hit rate", (620, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (40, 40, 40), 2)
    if attr_summary:
        start_y = 150
        colors = [(80, 160, 240), (90, 190, 120), (240, 170, 80), (180, 120, 240), (100, 200, 200)]
        for idx, (key, item) in enumerate(attr_summary.items()):
            y = start_y + idx * 52
            _draw_metric_bar(
                canvas,
                key,
                float(item["hit_rate"]),
                y,
                colors[idx % len(colors)],
                label_x=620,
                bar_x=800,
            )
    else:
        cv2.putText(canvas, "No attribute labels found.", (620, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 2)

    notes = [
        "Notes",
        f"- Search starts near the detected horizon (+{cfg.horizon_offset}px).",
        f"- Search band height: {cfg.horizon_band_height}px.",
        f"- At most {cfg.max_boxes} boxes are kept per frame.",
        "- Most false alarms come from bright horizon clutter.",
    ]
    base_y = 500
    for idx, line in enumerate(notes):
        cv2.putText(
            canvas,
            line,
            (620, base_y + idx * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68 if idx else 0.82,
            (45, 45, 45),
            2,
        )

    write_image(output_path, canvas)


def evaluate_folder(
    dataset_dir: Path,
    output_dir: Path,
    cfg: DetectorConfig,
    iou_thr: float = 0.1,
    save_steps: bool = False,
    step_limit: int = 12,
    paper_step_count: int = 0,
) -> Dict[str, object]:
    reset_dir(output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    step_dir = output_dir / "step_visualizations"
    if save_steps:
        step_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(dataset_dir)
    gts = read_groundtruth(dataset_dir / "groundtruth.txt")
    attrs = read_attributes(dataset_dir)
    n = align_lengths(images, gts)
    if n == 0:
        raise RuntimeError("No aligned frames between images and groundtruth.")

    detector = SeaSurfaceDetector(cfg)
    tp = fp = fn = 0
    ious_tp: List[float] = []
    frame_records: List[Dict[str, object]] = []
    saved_step_count = 0

    for idx in range(n):
        img_path = images[idx]
        img = read_image(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        norm = detector.initialize_image(img)
        should_save_steps = save_steps and (step_limit <= 0 or saved_step_count < step_limit)
        if should_save_steps:
            mask, meta, steps = detector.detect_steps(norm)
        else:
            mask, meta = detector.detect(norm)
            steps = {}
        preds = detector.mask_to_boxes(mask)
        gt = gts[idx]
        biou = best_iou(preds, gt)
        hit = biou >= iou_thr

        if hit:
            tp += 1
            ious_tp.append(biou)
            fp += max(0, len(preds) - 1)
        else:
            fn += 1
            fp += len(preds)

        vis = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
        for p in preds:
            cv2.rectangle(vis, (p[0], p[1]), (p[2], p[3]), (0, 0, 255), 1)
        cv2.rectangle(vis, (gt[0], gt[1]), (gt[2], gt[3]), (0, 255, 0), 1)
        cv2.line(vis, (0, meta["horizon_y"]), (vis.shape[1] - 1, meta["horizon_y"]), (255, 128, 0), 1)
        cv2.line(vis, (0, meta["roi_start_y"]), (vis.shape[1] - 1, meta["roi_start_y"]), (0, 255, 255), 1)
        cv2.putText(
            vis,
            f"iou={biou:.3f} preds={len(preds)}",
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )

        vis_path = vis_dir / f"{idx:04d}_{img_path.name}"
        write_image(vis_path, vis)
        step_path = None
        if should_save_steps:
            step_path = step_dir / f"{idx:04d}_{img_path.stem}_steps.png"
            _write_step_sheet(step_path, steps, vis, meta)
            saved_step_count += 1

        frame_records.append(
            {
                "frame": idx,
                "image": img_path.name,
                "gt": gt,
                "pred_count": len(preds),
                "best_iou": biou,
                "hit": hit,
                "status": _status_for_frame(hit, len(preds)),
                "vis_path": str(vis_path),
                "step_path": str(step_path) if step_path else "",
                "horizon_y": meta["horizon_y"],
                "roi_start_y": meta["roi_start_y"],
            }
        )

    avg_pred_per_frame = sum(int(r["pred_count"]) for r in frame_records) / max(1, len(frame_records))
    frames_with_no_pred = sum(1 for r in frame_records if int(r["pred_count"]) == 0)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    metrics = EvalMetrics(
        frames=n,
        tp=tp,
        fp=fp,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=f1,
        mean_iou_tp=(sum(ious_tp) / len(ious_tp)) if ious_tp else 0.0,
        avg_pred_per_frame=avg_pred_per_frame,
        frames_with_no_pred=frames_with_no_pred,
    )

    attr_summary: Dict[str, Dict[str, float]] = {}
    for key, seq in attrs.items():
        m = min(n, len(seq))
        pos = [frame_records[i]["hit"] for i in range(m) if seq[i] == 1]
        if pos:
            attr_summary[key] = {
                "frames": float(len(pos)),
                "hit_rate": float(sum(1 for x in pos if x) / len(pos)),
            }

    best_hits = sorted(
        [r for r in frame_records if bool(r["hit"])],
        key=lambda r: float(r["best_iou"]),
        reverse=True,
    )[:9]
    misses = sorted(
        [r for r in frame_records if r["status"] == "miss"],
        key=lambda r: int(r["frame"]),
    )[:9]
    false_alarms = sorted(
        [r for r in frame_records if r["status"] == "false_alarm"],
        key=lambda r: (int(r["pred_count"]), -float(r["best_iou"])),
        reverse=True,
    )[:9]
    paper_step_dir = ""
    if paper_step_count > 0:
        paper_records = best_hits[:paper_step_count]
        if not paper_records:
            paper_records = sorted(
                frame_records,
                key=lambda r: float(r["best_iou"]),
                reverse=True,
            )[:paper_step_count]
        paper_step_dir = _write_best_step_exports(output_dir, paper_records, images, gts, detector)

    best_hits_path = output_dir / "best_hits.png"
    misses_path = output_dir / "misses.png"
    false_alarms_path = output_dir / "false_alarms.png"
    overview_path = output_dir / "overview.png"
    summary_path = output_dir / "summary.md"

    build_contact_sheet(
        [(Path(r["vis_path"]), f"frame {int(r['frame']):04d}  iou {float(r['best_iou']):.2f}") for r in best_hits],
        best_hits_path,
        "Best hits",
    )
    build_contact_sheet(
        [(Path(r["vis_path"]), f"frame {int(r['frame']):04d}  preds {int(r['pred_count'])}") for r in misses],
        misses_path,
        "Misses",
    )
    build_contact_sheet(
        [(Path(r["vis_path"]), f"frame {int(r['frame']):04d}  preds {int(r['pred_count'])}") for r in false_alarms],
        false_alarms_path,
        "False alarms",
    )
    _build_overview_image(overview_path, dataset_dir, metrics, attr_summary, cfg)

    artifacts = {
        "summary": str(summary_path),
        "overview": str(overview_path),
        "visualizations": str(vis_dir),
        "step_visualizations": str(step_dir) if save_steps else "",
        "paper_step_images": paper_step_dir,
        "best_hits": str(best_hits_path),
        "misses": str(misses_path),
        "false_alarms": str(false_alarms_path),
    }
    _write_summary_markdown(summary_path, dataset_dir, cfg, metrics, attr_summary, artifacts)

    return {
        "dataset_dir": str(dataset_dir),
        "config": asdict(cfg),
        "iou_threshold": iou_thr,
        "metrics": asdict(metrics),
        "attributes": attr_summary,
        "frames": frame_records,
        "artifacts": artifacts,
    }
