"""
Passthrough generator — the stills the sequencer selected, unmodified.

No structural effect: it simply walks the sequence, so the blur suite and trails
can be used on plain footage, and so the sequencer's rate/order/jumps can be
auditioned on their own (this is the clearest way to see a traversal).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

NAME = "frames"


def output_len(selection: list[dict], num_frames: int | None = None) -> int:
    if num_frames:
        return max(1, int(num_frames))
    return max(1, sum(max(1, int(v["n_frames"])) for v in selection))


def generate(selection: list[dict], cache, *, num_frames: int | None = None,
             sequence=None, **_ignored) -> Iterator[np.ndarray]:
    if not selection:
        raise ValueError("passthrough needs at least one clip")
    if sequence is None:
        T = output_len(selection, num_frames)
        for t in range(T):
            v = selection[0]
            yield cache.get(v["video_id"], t % max(1, int(v["n_frames"])))
        return
    for t in range(sequence.length):
        ref = sequence.at(t)
        yield cache.get(ref.video_id, ref.frame_no)
