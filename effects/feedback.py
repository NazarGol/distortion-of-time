"""
Hyper-imposition — a stateful streaming post stage holding ONE accumulator:

    acc = combine(frame, transform(acc) * persistence)
    emit acc

`transform` is the parallax half of the effect: a small zoom / rotation /
translation applied to the accumulator on every pass, so echoes recede into a
tunnel instead of merely fading. With transform at zero it is plain trails.

Memory is one accumulator — flat regardless of render length.
"""
from __future__ import annotations

from typing import Iterable, Iterator

import cv2
import numpy as np

MODES = ("mean", "max", "add")


def stage(src: Iterable[np.ndarray], *, persistence: float = 0.0,
          mode: str = "mean", zoom: float = 0.0, rotate: float = 0.0,
          tx: float = 0.0, ty: float = 0.0) -> Iterator[np.ndarray]:
    p = float(persistence)
    if p <= 0.0:
        yield from src
        return

    acc = None
    m = None
    for frame in src:
        f = frame.astype(np.float32)
        if acc is None:
            acc = f
            yield np.clip(acc, 0, 255).astype(np.uint8)
            continue

        prev = acc
        if zoom or rotate or tx or ty:
            if m is None:
                h, w = frame.shape[:2]
                m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(rotate),
                                            1.0 + float(zoom))
                m[0, 2] += float(tx)
                m[1, 2] += float(ty)
            prev = cv2.warpAffine(prev, m, (frame.shape[1], frame.shape[0]),
                                  flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
        prev = prev * p

        if mode == "max":
            acc = np.maximum(f, prev)
        elif mode == "add":
            acc = np.minimum(f + prev, 255.0)
        else:                                     # mean / exponential moving average
            acc = f * (1.0 - p) + prev
        yield np.clip(acc, 0, 255).astype(np.uint8)
