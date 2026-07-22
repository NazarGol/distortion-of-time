"""
Composition — the layered compositor.

A Composition is N source *layers* + an ordered *effect chain*, rendered to a
frame stack.

  layer = { clip, offset, scale }
    clip   : library clip name (or absolute path)
    offset : temporal offset in frames — the layer shows moment (t - offset),
             wrapping around the clip. This is how one clip becomes "many moments":
             add it several times with different offsets.
    scale  : per-layer zoom about centre (1.0 = fit working frame)

  effect = { name, params }   applied in order; combiners (interlace) collapse
                              the layer list to one.

Everything is normalised to a common working (width, height, fps) so layers
compose cleanly. Preview renders a short, downscaled segment; full render uses
the composition's own settings.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import config
import effects
from core import RenderContext, get_device
from frames import clip as clipmod


def resolve_clip(name_or_path: str) -> str:
    """Turn a stored clip reference into an absolute file path."""
    if os.path.isabs(name_or_path) and os.path.exists(name_or_path):
        return name_or_path
    cand = os.path.join(config.LIBRARY_DIR, name_or_path)
    if os.path.exists(cand):
        return cand
    return name_or_path  # let the caller surface the missing-file error


def _even(x: int) -> int:
    return max(2, x - (x % 2))


@dataclass
class Composition:
    layers: list[dict[str, Any]] = field(default_factory=list)
    chain: list[dict[str, Any]] = field(default_factory=list)
    width: int = config.DEFAULT_WIDTH
    height: int = config.DEFAULT_HEIGHT
    fps: int = config.DEFAULT_FPS
    frames: int | None = None            # output length; None = auto (longest layer)
    name: str = "untitled"

    # ── layer construction ──────────────────────────────────────────────────────
    def _working_dims(self, is_preview: bool) -> tuple[int, int]:
        w, h = int(self.width), int(self.height)
        if is_preview and w > config.PREVIEW_MAX_WIDTH:
            scale = config.PREVIEW_MAX_WIDTH / float(w)
            w = config.PREVIEW_MAX_WIDTH
            h = int(round(h * scale))
        return _even(w), _even(h)

    def build_layers(self, is_preview: bool = False) -> tuple[list[np.ndarray], RenderContext]:
        if not self.layers:
            raise ValueError("Composition has no layers")

        w, h = self._working_dims(is_preview)

        # decode each layer's source at working resolution/fps
        sources: list[np.ndarray] = []
        for layer in self.layers:
            path = resolve_clip(layer["clip"])
            src = clipmod.extract(path, width=w, height=h, fps=self.fps)
            src = _apply_scale(src, float(layer.get("scale", 1.0)))
            sources.append(src)

        # output length
        if self.frames:
            T = int(self.frames)
        else:
            T = max(len(s) for s in sources)
        if is_preview:
            T = min(T, config.PREVIEW_MAX_FRAMES)

        # apply temporal offset per layer, wrapping around the clip
        layer_seqs: list[np.ndarray] = []
        for layer, src in zip(self.layers, sources):
            ns = len(src)
            offset = int(layer.get("offset", 0))
            idx = np.mod(np.arange(T) - offset, ns)
            layer_seqs.append(src[idx])

        ctx = RenderContext(
            fps=self.fps, width=w, height=h,
            is_preview=is_preview, device=get_device(),
            layer_opacities=[float(l.get("opacity", 1.0)) for l in self.layers],
        )
        return layer_seqs, ctx

    # ── render ───────────────────────────────────────────────────────────────────
    def render(self, is_preview: bool = False) -> tuple[np.ndarray, RenderContext]:
        layers, ctx = self.build_layers(is_preview)

        for spec in self.chain:
            eff = effects.build(spec["name"], spec.get("params", {}))
            layers = eff.apply(layers, ctx)

        frames = _flatten(layers)
        return frames, ctx

    # ── persistence ──────────────────────────────────────────────────────────────
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": self.frames,
            "layers": self.layers,
            "chain": self.chain,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Composition":
        return cls(
            layers=d.get("layers", []),
            chain=d.get("chain", []),
            width=d.get("width", config.DEFAULT_WIDTH),
            height=d.get("height", config.DEFAULT_HEIGHT),
            fps=d.get("fps", config.DEFAULT_FPS),
            frames=d.get("frames"),
            name=d.get("name", "untitled"),
        )

    def save(self, path: str | None = None) -> str:
        if path is None:
            safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.name)
            path = os.path.join(config.COMPOSITIONS_DIR, f"{safe or 'composition'}.json")
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def load(cls, path: str) -> "Composition":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ── helpers ──────────────────────────────────────────────────────────────────────
def _apply_scale(src: np.ndarray, scale: float) -> np.ndarray:
    """Zoom each frame about its centre by `scale`, keeping frame size (crop/pad).

    Vectorised via an affine warp per frame (OpenCV); no pixel loops.
    """
    if abs(scale - 1.0) < 1e-3:
        return src
    import cv2

    n, h, w = src.shape[:3]
    m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0.0, scale)
    out = np.empty_like(src)
    for i in range(n):
        out[i] = cv2.warpAffine(src[i], m, (w, h), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT)
    return out


def _flatten(layers: list[np.ndarray]) -> np.ndarray:
    """Collapse any remaining layers to one frame stack (mean blend as a safety
    net when no combiner ran)."""
    if len(layers) == 1:
        return layers[0]
    acc = np.mean(np.stack([l.astype(np.float32) for l in layers], 0), axis=0)
    return np.clip(acc, 0, 255).astype(np.uint8)


def new_render_path(prefix: str = "render") -> str:
    return os.path.join(config.OUTPUT_DIR, f"{prefix}_{int(time.time())}.mp4")
