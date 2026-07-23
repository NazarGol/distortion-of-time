"""
Weave generator — the strip time-displacement weave, as a sibling effect.

Two paths, and the distinction matters:

  • NEUTRAL sequence (linear order, one still per output frame, no jumps) →
    delegates straight into the frozen, unit-verified `pool_weave.iter_frames`.
    Pure delegation is what guarantees a neutral render stays byte-identical to
    previous output.

  • NON-NEUTRAL sequence → the same strip displacement formula, but the time axis
    is resolved through the sequencer, so strips inherit the traversal's rate,
    order and jumps:

        strip i of output frame t  ←  sequence.ref(step_index(t) + round(i*frame_step))

    The mapping (which strip, which displacement) is unchanged; only what a
    "time index" resolves to changes — which is the whole point of the sequencer.
    Note that in this path the sequence carries video identity per step, so the
    traversal (e.g. `crosscut`) governs source rotation instead of `i mod N`.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

import config
import pool_weave

NAME = "weave"


def generate(selection: list[dict], cache, *,
             strip_width: int = 2, frame_step: float = 1.0,
             orientation: str = "vertical", source_mode: str = "rotate",
             num_frames: int | None = None, sequence=None,
             **_ignored) -> Iterator[np.ndarray]:
    if sequence is None or sequence.neutral:
        return pool_weave.iter_frames(
            selection, cache,
            strip_width=strip_width, frame_step=frame_step,
            orientation=orientation, source_mode=source_mode,
            num_frames=num_frames,
        )
    return _sequenced(selection, cache, strip_width, frame_step,
                      orientation, sequence)


def _sequenced(selection, cache, strip_width, frame_step, orientation, seq
               ) -> Iterator[np.ndarray]:
    W, H = config.WORKING_WIDTH, config.WORKING_HEIGHT
    sw = max(1, int(strip_width))
    vertical = orientation != "horizontal"
    axis_len = W if vertical else H
    n_strips = (axis_len + sw - 1) // sw
    # same per-strip plan as the frozen weave: band + integer displacement
    bands = [(i * sw, min(i * sw + sw, axis_len), int(round(i * frame_step)))
             for i in range(n_strips)]

    for t in range(seq.length):
        pos = seq.step_index(t)
        out = np.empty((H, W, 3), np.uint8)
        for a, b, disp in bands:
            ref = seq.ref(pos + disp)
            src = cache.get(ref.video_id, ref.frame_no)
            if vertical:
                out[:, a:b] = src[:, a:b]
            else:
                out[a:b, :] = src[a:b, :]
        yield out


def output_len(selection: list[dict], source_mode: str = "rotate",
               num_frames: int | None = None) -> int:
    return pool_weave._output_len(selection, source_mode, num_frames)
