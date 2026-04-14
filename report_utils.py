from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from image_io import read_image, write_image


def build_contact_sheet(
    items: Sequence[tuple[Path, str]],
    output_path: Path,
    title: str,
    thumb_size: tuple[int, int] = (320, 256),
    cols: int = 3,
) -> None:
    thumb_w, thumb_h = thumb_size
    padding = 16
    title_h = 60
    caption_h = 28

    if not items:
        canvas = np.full((180, 720, 3), 245, dtype=np.uint8)
        cv2.putText(canvas, title, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 40, 40), 2)
        cv2.putText(
            canvas,
            "No images available for this panel.",
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (90, 90, 90),
            2,
        )
        write_image(output_path, canvas)
        return

    rows = math.ceil(len(items) / cols)
    sheet_w = padding + cols * (thumb_w + padding)
    sheet_h = title_h + padding + rows * (thumb_h + caption_h + padding)
    canvas = np.full((sheet_h, sheet_w, 3), 245, dtype=np.uint8)

    cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.95, (40, 40, 40), 2)

    for idx, (img_path, caption) in enumerate(items):
        row = idx // cols
        col = idx % cols
        x = padding + col * (thumb_w + padding)
        y = title_h + padding + row * (thumb_h + caption_h + padding)

        image = read_image(img_path)
        if image is None:
            tile = np.full((thumb_h, thumb_w, 3), 230, dtype=np.uint8)
            cv2.putText(
                tile,
                "Image missing",
                (40, thumb_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (80, 80, 80),
                2,
            )
        else:
            if image.ndim == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            tile = cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

        canvas[y : y + thumb_h, x : x + thumb_w] = tile
        cv2.rectangle(canvas, (x, y), (x + thumb_w, y + thumb_h), (180, 180, 180), 1)
        cv2.putText(
            canvas,
            caption[:42],
            (x, y + thumb_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )

    write_image(output_path, canvas)
