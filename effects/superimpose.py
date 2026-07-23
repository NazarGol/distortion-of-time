"""
Superimposition generator — a sibling of the weave, NOT a stage inside it.

For each output frame it gathers `layers` frames from the pool and blends them
into one image. No strips are involved.

Layer sourcing follows the filmstrip selection: layer j comes from
`selection[j % N]` at time `t + j*spread`. Two controls give three gestures:

    many clips + spread 0   → N different videos at the same instant
    one clip  + spread > 0  → long-exposure stack of one scene's moments
    many clips + spread > 0 → both at once

`parallax` spatially displaces each layer before blending (see parallax.py), so
the stack can read as depth instead of a flat ghost.

Memory is bounded by `layers` (≤32 frames), never by render length.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from effects.parallax import Parallax

NAME = "superimpose"
BLENDS = ("mean", "screen", "max", "difference", "multiply")


def output_len(selection: list[dict], num_frames: int | None = None) -> int:
    if num_frames:
        return max(1, int(num_frames))
    return max(max(1, int(v["n_frames"])) for v in selection)


def _blend(stack: list[np.ndarray], mode: str) -> np.ndarray:
    """Blend layer images (uint8) → one uint8 image. Vectorised whole-array ops."""
    if len(stack) == 1:
        return stack[0]
    if mode == "mean":
        acc = np.zeros(stack[0].shape, np.float32)
        for img in stack:
            acc += img
        acc /= len(stack)
        return np.clip(acc, 0, 255).astype(np.uint8)

    acc = stack[0].astype(np.float32) / 255.0
    for img in stack[1:]:
        b = img.astype(np.float32) / 255.0
        if mode == "screen":
            acc = 1.0 - (1.0 - acc) * (1.0 - b)
        elif mode == "max":
            acc = np.maximum(acc, b)
        elif mode == "difference":
            acc = np.abs(acc - b)
        elif mode == "multiply":
            acc = acc * b
        else:
            acc = b
    return np.clip(acc * 255.0, 0, 255).astype(np.uint8)


def generate(selection: list[dict], cache, *,
             layers: int = 1, spread: float = 0.0, blend: str = "mean",
             fps: float = 24.0, parallax: str = "none", amount: float = 0.0,
             px_zoom: bool = True, px_pan: bool = False, px_rotate: bool = False,
             num_frames: int | None = None, **_ignored) -> Iterator[np.ndarray]:
    if not selection:
        raise ValueError("superimpose needs at least one clip")
    n_layers = max(1, min(32, int(layers)))
    step_frames = float(spread) * float(fps)
    n_sel = len(selection)
    T = output_len(selection, num_frames)

    px = Parallax(parallax, amount, zoom=px_zoom, pan=px_pan,
                  rotate=px_rotate, fps=fps)
    base = selection[0]
    base_len = max(1, int(base["n_frames"]))

    for t in range(T):
        # per-frame precompute (flow is computed once, not once per layer)
        if px.mode == "flow" and px.amount:
            b0 = cache.get(base["video_id"], t % base_len)
            b1 = cache.get(base["video_id"], (t + 1) % base_len)
            px.begin_frame(b0, b1)

        stack = []
        for j in range(n_layers):
            v = selection[j % n_sel]
            L = max(1, int(v["n_frames"]))
            fn = int(round(t + j * step_frames)) % L
            img = cache.get(v["video_id"], fn)
            stack.append(px.apply(img, j, video_id=v["video_id"], frame_no=fn))
        yield _blend(stack, blend)
