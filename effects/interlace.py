"""
Interlacing / Lenticular — the signature effect.

Weave 2+ layers into fine strips within each output frame. Because each layer is
sampled at its own temporal offset (set on the layer, upstream), a single output
frame interleaves *different moments*: the venetian/lenticular time-smear.

  • strip_width  — px per strip. 1–3 px = fine lenticular shimmer; wider = bands.
  • orientation  — vertical (default) or horizontal strips.
  • phase_speed  — px/frame the strip pattern scrolls, giving parallax-in-time.
  • feather      — soften strip seams (px) via a directional blur across the weave.

Fully vectorised: strip→layer assignment is computed as arrays over (time, axis);
selection is a handful of masked np.where calls (one per layer, L is 2–4). No
per-pixel Python loops.
"""
from __future__ import annotations

import cv2
import numpy as np

from core import RenderContext
from effects.base import Effect, register, stack_layers, to_u8


@register
class InterlaceEffect(Effect):
    name = "interlace"
    label = "Interlace / Lenticular"
    reduces = True
    PARAMS = [
        {"name": "strip_width", "label": "Strip width (px)", "type": "int",
         "min": 1, "max": 64, "step": 1, "default": 2},
        {"name": "orientation", "label": "Orientation", "type": "choice",
         "choices": ["vertical", "horizontal"], "default": "vertical"},
        {"name": "phase_speed", "label": "Scroll speed (px/frame)", "type": "float",
         "min": -8.0, "max": 8.0, "step": 0.25, "default": 1.0},
        {"name": "feather", "label": "Feather seams (px)", "type": "int",
         "min": 0, "max": 16, "step": 1, "default": 0},
    ]

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        if len(layers) == 1:
            return layers  # nothing to weave

        S = stack_layers(layers)                 # (L, T, H, W, 3) float32
        L, T, H, W, _ = S.shape

        strip_w = max(1, int(self.params["strip_width"]))
        orient = self.params["orientation"]
        phase_speed = float(self.params["phase_speed"])
        feather = int(self.params["feather"])

        # Work along the strip axis: columns (W) for vertical, rows (H) for horizontal.
        axis_len = W if orient == "vertical" else H
        coord = np.arange(axis_len)                          # (axis_len,)
        t = np.arange(T)
        phase = np.round(phase_speed * t).astype(np.int64)   # (T,) animated scroll

        # layer index per (time, coord): which layer owns this strip at this frame
        pos = coord[None, :] + phase[:, None]                # (T, axis_len)
        strip = np.floor_divide(pos, strip_w)
        layer_idx = np.mod(strip, L).astype(np.int64)        # (T, axis_len) in [0,L)

        out = np.zeros((T, H, W, 3), np.float32)
        for l in range(L):
            sel = (layer_idx == l)                           # (T, axis_len) bool
            if orient == "vertical":
                mask = sel[:, None, :, None]                 # broadcast over H, C
            else:
                mask = sel[:, :, None, None]                 # broadcast over W, C
            np.copyto(out, S[l], where=mask)

        if feather > 0:
            out = _feather_seams(out, orient, feather)

        return [to_u8(out)]


def _feather_seams(frames: np.ndarray, orient: str, feather: int) -> np.ndarray:
    """Soften strip seams with a 1-D Gaussian blur across the strip axis only.

    Vectorised per frame via OpenCV; no pixel loops. Kernel is odd and spans the
    strip axis (horizontal for vertical strips, vertical for horizontal strips).
    """
    k = 2 * feather + 1
    if orient == "vertical":
        ksize = (k, 1)
    else:
        ksize = (1, k)
    out = np.empty_like(frames)
    for i in range(frames.shape[0]):
        out[i] = cv2.GaussianBlur(frames[i], ksize, 0)
    return out
