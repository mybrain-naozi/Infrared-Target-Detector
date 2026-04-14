from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np


@dataclass
class DetectorConfig:
    target_width: int = 640
    target_height: int = 512
    roi_start_y: int = 40
    peak_kernel: int = 19
    l_factor: float = 2.1
    min_area: int = 6
    max_area: int = 800
    opening_kernel: int = 1
    merge_margin: int = 2
    horizon_band_height: int = 200
    min_width: int = 3
    min_height: int = 2
    max_aspect_ratio: float = 8.0
    nms_iou_threshold: float = 0.35
    tophat_kernel: int = 11
    tophat_percentile: float = 99.7
    min_local_contrast: float = 5.0
    max_boxes: int = 2
    auto_horizon: bool = True
    horizon_search_top: int = 5
    horizon_search_bottom: int = 200
    horizon_offset: int = 10


class SeaSurfaceDetector:
    """Sea-surface small target detector with auto horizon anchoring."""

    def __init__(self, cfg: DetectorConfig | None = None):
        self.cfg = cfg or DetectorConfig()

    def initialize_image(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("Input image is None.")
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img.shape != (self.cfg.target_height, self.cfg.target_width):
            img = cv2.resize(
                img,
                (self.cfg.target_width, self.cfg.target_height),
                interpolation=cv2.INTER_CUBIC,
            )
        cropped = img[3:509, 3:637]
        norm = cv2.resize(
            cropped,
            (self.cfg.target_width, self.cfg.target_height),
            interpolation=cv2.INTER_CUBIC,
        )
        return norm

    @staticmethod
    def row_average_adjust(img: np.ndarray) -> np.ndarray:
        row_means = np.mean(img, axis=1)
        target = float(np.max(row_means))
        delta = target - row_means
        return np.clip(img.astype(np.float32) + delta[:, None], 0, 255).astype(np.uint8)

    def estimate_horizon(self, gray: np.ndarray) -> int:
        row_signal = gray.mean(axis=1).astype(np.float32).reshape(-1, 1)
        row_signal = cv2.GaussianBlur(row_signal, (1, 31), 0).reshape(-1)
        top = int(np.clip(self.cfg.horizon_search_top, 0, gray.shape[0] - 2))
        bottom = int(np.clip(self.cfg.horizon_search_bottom, top + 1, gray.shape[0] - 1))
        grad = np.diff(row_signal)
        window = grad[top:bottom]
        if window.size == 0:
            return int(np.clip(self.cfg.roi_start_y - self.cfg.horizon_offset, 0, gray.shape[0] - 1))
        return int(np.argmin(window) + top)

    def resolve_roi_start(self, gray: np.ndarray, horizon_y: int | None = None) -> int:
        if self.cfg.auto_horizon:
            if horizon_y is None:
                horizon_y = self.estimate_horizon(gray)
            roi_start = horizon_y + int(self.cfg.horizon_offset)
        else:
            roi_start = int(self.cfg.roi_start_y)
        return int(np.clip(roi_start, 0, gray.shape[0] - 1))

    def _peak_mask(self, roi: np.ndarray) -> np.ndarray:
        k = max(3, int(self.cfg.peak_kernel))
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), np.uint8)
        local_max = cv2.dilate(roi, kernel)
        bright = roi > np.mean(roi)
        peak = (roi == local_max) & bright
        values = roi[peak]
        if values.size == 0:
            return np.zeros_like(roi, dtype=np.uint8)
        threshold = min(float(np.mean(values) + self.cfg.l_factor * np.std(values)), 255.0)
        return (peak & (roi >= threshold)).astype(np.uint8) * 255

    def _tophat_mask(self, roi: np.ndarray) -> np.ndarray:
        k = max(3, int(self.cfg.tophat_kernel))
        if k % 2 == 0:
            k += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        th = cv2.morphologyEx(roi, cv2.MORPH_TOPHAT, kernel).astype(np.float32)
        thr = float(np.percentile(th, self.cfg.tophat_percentile))
        return (th >= thr).astype(np.uint8) * 255

    def _clean_mask(self, mask: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
        if self.cfg.opening_kernel > 1:
            kernel = np.ones((self.cfg.opening_kernel, self.cfg.opening_kernel), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = np.zeros_like(mask)
        h_img, w_img = ref_img.shape
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            if w <= 0 or h <= 0:
                continue
            aspect = max(w / h, h / w)
            x1 = x
            y1 = y
            x2 = min(w_img, x + w)
            y2 = min(h_img, y + h)
            pad = 4
            rx1 = max(0, x1 - pad)
            ry1 = max(0, y1 - pad)
            rx2 = min(w_img, x2 + pad)
            ry2 = min(h_img, y2 + pad)
            inner = ref_img[y1:y2, x1:x2]
            ring = ref_img[ry1:ry2, rx1:rx2].copy()
            ring[(y1 - ry1) : (y2 - ry1), (x1 - rx1) : (x2 - rx1)] = 0
            inner_mean = float(np.mean(inner)) if inner.size else 0.0
            ring_vals = ring[ring > 0]
            ring_mean = float(np.mean(ring_vals)) if ring_vals.size else inner_mean
            contrast = inner_mean - ring_mean
            if (
                self.cfg.min_area <= area <= self.cfg.max_area
                and w >= self.cfg.min_width
                and h >= self.cfg.min_height
                and aspect <= self.cfg.max_aspect_ratio
                and contrast >= self.cfg.min_local_contrast
            ):
                out[labels == i] = 255
        return out

    def detect(self, gray: np.ndarray) -> tuple[np.ndarray, Dict[str, int]]:
        horizon_y = self.estimate_horizon(gray)
        roi_start = self.resolve_roi_start(gray, horizon_y)
        adj = self.row_average_adjust(gray)
        roi = adj[roi_start:, :]
        if roi.size == 0:
            return np.zeros_like(adj, dtype=np.uint8), {
                "horizon_y": horizon_y,
                "roi_start_y": roi_start,
                "band_height": 0,
            }

        band_h = int(np.clip(self.cfg.horizon_band_height, 1, roi.shape[0]))
        band = roi[:band_h, :]
        peak_mask = self._peak_mask(band)
        th_mask = self._tophat_mask(band)
        mask = cv2.bitwise_and(peak_mask, th_mask)
        mask = self._clean_mask(mask, band)

        padded = np.zeros_like(roi, dtype=np.uint8)
        padded[:band_h, :] = mask
        out = np.zeros_like(adj, dtype=np.uint8)
        out[roi_start:, :] = padded
        return out, {
            "horizon_y": int(horizon_y),
            "roi_start_y": int(roi_start),
            "band_height": int(band_h),
        }

    def detect_mask(self, gray: np.ndarray) -> np.ndarray:
        mask, _ = self.detect(gray)
        return mask

    def mask_to_boxes(self, mask: np.ndarray, expand: int | None = None) -> List[Tuple[int, int, int, int]]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        h, w = mask.shape
        margin = self.cfg.merge_margin if expand is None else int(expand)
        boxes: List[Tuple[int, int, int, int]] = []
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(w - 1, x + bw + margin)
            y2 = min(h - 1, y + bh + margin)
            boxes.append((x1, y1, x2, y2))
        boxes = self._nms_boxes(boxes, iou_thr=self.cfg.nms_iou_threshold)
        return boxes[: max(1, int(self.cfg.max_boxes))]

    @staticmethod
    def _nms_boxes(
        boxes: List[Tuple[int, int, int, int]], iou_thr: float = 0.35
    ) -> List[Tuple[int, int, int, int]]:
        if not boxes:
            return []
        areas = [max(1, (b[2] - b[0] + 1) * (b[3] - b[1] + 1)) for b in boxes]
        order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
        keep: List[Tuple[int, int, int, int]] = []
        for idx in order:
            cand = boxes[idx]
            suppress = False
            for kept in keep:
                xx1 = max(cand[0], kept[0])
                yy1 = max(cand[1], kept[1])
                xx2 = min(cand[2], kept[2])
                yy2 = min(cand[3], kept[3])
                iw = max(0, xx2 - xx1 + 1)
                ih = max(0, yy2 - yy1 + 1)
                inter = iw * ih
                if inter == 0:
                    continue
                cand_area = max(1, (cand[2] - cand[0] + 1) * (cand[3] - cand[1] + 1))
                kept_area = max(1, (kept[2] - kept[0] + 1) * (kept[3] - kept[1] + 1))
                iou = inter / float(cand_area + kept_area - inter)
                if iou >= iou_thr:
                    suppress = True
                    break
            if not suppress:
                keep.append(cand)
        return keep
