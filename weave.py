"""
The core effect: vertical-strip time-displacement weave.

A slit-scan across multiple videos — NOT a blend, mean, or overlay. The output
frame is divided into thin strips (columns by default). Walking the strips
left → right, each successive strip advances one step in time and rotates to the
next source video:

    out[t], columns of strip i  =  the same columns taken from
        video[i mod N]  at  frame (t + i * frame_step) mod len(video)

So column band i shows video (i mod N) at a *later moment* than band i-1: a
single output image holds many consecutive moments woven across the N sources —
time runs across the width of the frame. Because the sample time is (t + …),
the whole weave flows forward as the output plays. Clips shorter than needed
loop (modulo their own length).

Vectorisation: for each source video we gather all of its strips for all output
frames in ONE numpy fancy-index — no per-pixel loops, no per-frame Python loop.

    source_mode = "rotate"  strip i ← video[i mod N]        (default)
                  "single"  every strip ← video[0]          (pure slit-scan)
"""
from __future__ import annotations

import numpy as np


def weave(
    clips: list[np.ndarray],
    strip_width: int = 2,
    frame_step: int = 1,
    orientation: str = "vertical",
    source_mode: str = "rotate",
    num_frames: int | None = None,
) -> np.ndarray:
    """Weave N clips into one frame stack.

    clips        : list of (T_v, H, W, 3) uint8 RGB stacks, same H and W
                   (lengths T_v may differ — sampling wraps per clip).
    strip_width  : strip thickness in px (1–3 px = fine lenticular shimmer).
    frame_step   : frames advanced per strip (may be negative: time runs the
                   other way across the frame).
    orientation  : "vertical" (strips are columns) or "horizontal" (rows).
    num_frames   : output length; default = longest clip.

    Returns (T, H, W, 3) uint8.
    """
    if not clips:
        raise ValueError("weave() needs at least one clip")
    n = len(clips)
    h, w = clips[0].shape[1:3]
    for c in clips:
        if c.shape[1:3] != (h, w):
            raise ValueError("all clips must share the same working resolution")

    t_out = int(num_frames) if num_frames else max(c.shape[0] for c in clips)
    sw = max(1, int(strip_width))
    step = int(frame_step)
    vertical = orientation != "horizontal"
    axis_len = w if vertical else h

    # Per-column (or per-row) maps: which strip, hence which source and which
    # temporal displacement, feeds each pixel column of the output.
    pos = np.arange(axis_len)
    strip = pos // sw                                     # strip index i
    src_of = strip % n if source_mode == "rotate" else np.zeros_like(strip)
    t_off = strip * step                                  # i * frame_step

    t_arr = np.arange(t_out)
    out = np.empty((t_out, h, w, 3), dtype=np.uint8)

    for v in range(n):
        sel = np.flatnonzero(src_of == v)                 # this video's columns
        if sel.size == 0:
            continue
        t_v = clips[v].shape[0]
        # (T, K) source-frame index for every output frame × column of video v
        f = (t_arr[:, None] + t_off[None, sel]) % t_v
        if vertical:
            # advanced indices at axes (0, 2) with a slice between → (T, K, H, 3)
            g = clips[v][f, :, sel]
            out[:, :, sel] = np.transpose(g, (0, 2, 1, 3))
        else:
            # advanced indices at adjacent axes (0, 1) → (T, K, W, 3)
            out[:, sel] = clips[v][f, sel]
    return out
