"""
Passthrough generator — plain frames straight out of the pool, no structural
effect. Exists so the blur suite and trails can be used on unmodified footage
(no weave, no superimposition involved).

Walks the selection in order, playing each selected clip end to end.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

NAME = "frames"


def output_len(selection: list[dict], num_frames: int | None = None) -> int:
    if num_frames:
        return max(1, int(num_frames))
    return max(1, sum(max(1, int(v["n_frames"])) for v in selection))


def generate(selection: list[dict], cache, *,
             num_frames: int | None = None, **_ignored) -> Iterator[np.ndarray]:
    if not selection:
        raise ValueError("passthrough needs at least one clip")
    T = output_len(selection, num_frames)
    emitted = 0
    while emitted < T:
        for v in selection:
            L = max(1, int(v["n_frames"]))
            for i in range(L):
                if emitted >= T:
                    return
                yield cache.get(v["video_id"], i)
                emitted += 1
