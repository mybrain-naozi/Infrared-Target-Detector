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
) -> Dict[str, object]:
    reset_dir(output_dir)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

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

    for idx in range(n):
        img_path = images[idx]
        img = read_image(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        norm = detector.initialize_image(img)
        mask, meta = detector.detect(norm)
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
