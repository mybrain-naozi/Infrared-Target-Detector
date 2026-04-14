from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_image(path: str | Path, flags: int = cv2.IMREAD_COLOR):
    raw_path = Path(path)
    try:
        data = np.fromfile(str(raw_path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def write_image(path: str | Path, image: np.ndarray, params: list[int] | None = None) -> bool:
    raw_path = Path(path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = raw_path.suffix.lower() or ".png"
    ok, buffer = cv2.imencode(suffix, image, params or [])
    if not ok:
        return False
    buffer.tofile(str(raw_path))
    return True
