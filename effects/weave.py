"""
Weave generator — the strip time-displacement weave, as a sibling effect.

This is a thin adapter ONLY. The maths live in the frozen, unit-verified
`pool_weave.iter_frames` (which in turn matches `weave.py` bit-for-bit); nothing
here touches the strip/time/source mapping. Keeping it a pure delegation is what
guarantees a neutral weave render stays byte-identical to previous output.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

import pool_weave

NAME = "weave"


def generate(selection: list[dict], cache, *,
             strip_width: int = 2, frame_step: float = 1.0,
             orientation: str = "vertical", source_mode: str = "rotate",
             num_frames: int | None = None, **_ignored) -> Iterator[np.ndarray]:
    return pool_weave.iter_frames(
        selection, cache,
        strip_width=strip_width, frame_step=frame_step,
        orientation=orientation, source_mode=source_mode,
        num_frames=num_frames,
    )


def output_len(selection: list[dict], source_mode: str = "rotate",
               num_frames: int | None = None) -> int:
    return pool_weave._output_len(selection, source_mode, num_frames)
