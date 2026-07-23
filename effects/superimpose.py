"""
Superimposition generator — a sibling of the weave, NOT a stage inside it.

For each output frame it gathers `layers` stills and blends them into one image.
No strips are involved.

Layers are **traversal neighbours**, not clock-time offsets: layer j comes from

    sequence[pos + j*stride]           (forward)
    sequence[pos + (j - (L-1)/2)*stride]   (centred)

where `pos` is the sequence step for this output frame. Because the sequence may
jump, a "neighbour" can be a genuinely unrelated moment — or a different video.
That discontinuity is the effect, deliberately preserved.

`stride` is in sequence steps (fractional-friendly), which is why it replaced the
old `spread` in seconds: once traversal rate and order are compositional, clock
time is the wrong unit for the stack.

`parallax` spatially displaces each layer before blending (see parallax.py).
Memory is bounded by `layers` (≤32), never by render length.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np

from effects.parallax import Parallax

NAME = "superimpose"
BLENDS = ("mean", "screen", "max", "difference", "multiply")
FLOW_SOURCES = ("source", "output")


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
        return np.clip(acc / len(stack), 0, 255).astype(np.uint8)

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
             layers: int = 1, stride: float = 1.0, blend: str = "mean",
             centred: bool = False, fps: float = 24.0,
             parallax: str = "none", amount: float = 0.0,
             px_zoom: bool = True, px_pan: bool = False, px_rotate: bool = False,
             flow_source: str = "source",
             num_frames: int | None = None, sequence=None,
             **_ignored) -> Iterator[np.ndarray]:
    if not selection:
        raise ValueError("superimpose needs at least one clip")
    if sequence is None:
        raise ValueError("superimpose needs a sequence (see sequencer.build_sequence)")

    n_layers = max(1, min(32, int(layers)))
    stride = float(stride)
    px = Parallax(parallax, amount, zoom=px_zoom, pan=px_pan,
                  rotate=px_rotate, fps=fps)
    prev_out = None

    for t in range(sequence.length):
        pos = sequence.step_index(t)

        # per-frame flow precompute (once, not once per layer)
        if px.mode == "flow" and px.amount:
            r0 = sequence.ref(pos)
            base_img = cache.get(r0.video_id, r0.frame_no)
            if flow_source == "output" and prev_out is not None:
                px.begin_frame(prev_out, base_img)
            else:
                r1 = sequence.ref(pos + 1)
                px.begin_frame(base_img, cache.get(r1.video_id, r1.frame_no))

        stack = []
        for j in range(n_layers):
            off = (j - (n_layers - 1) / 2.0) if centred else float(j)
            ref = sequence.ref(pos + int(round(off * stride)))
            img = cache.get(ref.video_id, ref.frame_no)
            stack.append(px.apply(img, j, video_id=ref.video_id, frame_no=ref.frame_no))

        out = _blend(stack, blend)
        prev_out = out
        yield out
