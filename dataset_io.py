from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def list_images(folder: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])


def gt_line_to_box(line: str) -> Tuple[int, int, int, int]:
    vals = [float(x.strip()) for x in line.split(",") if x.strip()]
    if len(vals) < 8:
        raise ValueError(f"groundtruth line format invalid: {line}")
    xs = vals[0::2]
    ys = vals[1::2]
    x1, x2 = int(min(xs)), int(max(xs))
    y1, y2 = int(min(ys)), int(max(ys))
    return x1, y1, x2, y2


def read_groundtruth(path: Path) -> List[Tuple[int, int, int, int]]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [gt_line_to_box(ln) for ln in lines]


def read_binary_label(path: Path) -> List[int]:
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [int(float(x)) for x in lines]


def read_attributes(folder: Path) -> Dict[str, List[int]]:
    names = [
        "camera_motion.label",
        "motion_change.label",
        "size_change.label",
        "occlusion.label",
        "dynamics_change.label",
    ]
    out: Dict[str, List[int]] = {}
    for n in names:
        p = folder / n
        if p.exists():
            out[p.stem] = read_binary_label(p)
    return out


def align_lengths(*seqs: Sequence) -> int:
    if not seqs:
        return 0
    return min(len(s) for s in seqs)
