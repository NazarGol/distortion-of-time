"""
Superimposition & Hyper-imposition (Phase 3).

  • SuperimposeEffect — a *combiner*: blend N layers into one with per-layer
    opacity and a blend mode (normal / add / screen / multiply / difference).
    Each layer already carries its own temporal offset (set upstream), so
    superimposing several offset copies of one clip is a long-exposure of many
    moments.

  • FeedbackEffect — *hyper-imposition*: blend each new frame into a decaying
    accumulator of prior output, leaving echo/trails. This is a causal scan over
    time; it loops over frames (T ≈ 150) but every step is a vectorised array op
    over the whole (H, W, 3) plane — no per-pixel loops.

Blend math runs in normalised float [0,1] then returns uint8.
"""
from __future__ import annotations

import numpy as np

from core import RenderContext
from effects.base import Effect, register, to_u8


# ── blend primitives (operate on float arrays in [0,1]) ──────────────────────────
def _blend(a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    if mode == "normal":
        return b
    if mode == "add":
        return np.clip(a + b, 0.0, 1.0)
    if mode == "screen":
        return 1.0 - (1.0 - a) * (1.0 - b)
    if mode == "multiply":
        return a * b
    if mode == "difference":
        return np.abs(a - b)
    return b


@register
class SuperimposeEffect(Effect):
    name = "superimpose"
    label = "Superimpose"
    reduces = True
    PARAMS = [
        {"name": "blend", "label": "Blend mode", "type": "choice",
         "choices": ["normal", "add", "screen", "multiply", "difference"],
         "default": "screen"},
        {"name": "mix", "label": "Mix (upper layers)", "type": "float",
         "min": 0.0, "max": 1.0, "step": 0.05, "default": 0.6},
    ]

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        if len(layers) == 1:
            return layers

        opac = ctx.layer_opacities or [1.0] * len(layers)
        # guard against a stale/short opacity list
        if len(opac) < len(layers):
            opac = list(opac) + [1.0] * (len(layers) - len(opac))

        mode = self.params["blend"]
        mix = float(self.params["mix"])

        acc = layers[0].astype(np.float32) / 255.0 * float(opac[0])   # base, faded from black
        for i in range(1, len(layers)):
            b = layers[i].astype(np.float32) / 255.0
            blended = _blend(acc, b, mode)
            alpha = float(opac[i]) * mix
            acc = acc * (1.0 - alpha) + blended * alpha

        return [to_u8(acc * 255.0)]


@register
class FeedbackEffect(Effect):
    name = "feedback"
    label = "Feedback trails (hyper-impose)"
    reduces = False
    PARAMS = [
        {"name": "decay", "label": "Persistence", "type": "float",
         "min": 0.0, "max": 0.98, "step": 0.02, "default": 0.8},
        {"name": "mode", "label": "Trail mode", "type": "choice",
         "choices": ["max", "mean", "add"], "default": "max"},
    ]

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        decay = float(self.params["decay"])
        mode = self.params["mode"]

        out_layers = []
        for layer in layers:
            f = layer.astype(np.float32)
            acc = np.empty_like(f)
            acc[0] = f[0]
            for t in range(1, f.shape[0]):
                prev = acc[t - 1] * decay
                if mode == "max":
                    acc[t] = np.maximum(f[t], prev)
                elif mode == "add":
                    acc[t] = np.minimum(f[t] + prev, 255.0)
                else:  # mean / exponential moving average
                    acc[t] = f[t] * (1.0 - decay) + acc[t - 1] * decay
            out_layers.append(to_u8(acc))
        return out_layers
