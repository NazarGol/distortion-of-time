"""
Feedback trails (hyper-imposition) — optional post-processing on the woven
output. Blends each new frame into a decaying accumulator of prior output,
leaving echo/trails. A causal scan over time: loops over frames (T ≈ 150) but
every step is a vectorised array op over the whole (H, W, 3) plane.
"""
from __future__ import annotations

import numpy as np

from core import RenderContext
from effects.base import Effect, register, to_u8


@register
class FeedbackEffect(Effect):
    name = "feedback"
    label = "Feedback trails"
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
