"""
Effect framework.

An Effect transforms a *list of layers*. Each layer is an (T, H, W, 3) uint8 RGB
array; all layers in the list share the same T, H, W (the compositor guarantees
this). Working in whole sequences (not per-pixel, not per-frame-in-Python) keeps
everything vectorised — the hard rule for this project.

Two broad kinds of effect:
  • combiners  (interlace, superimpose)  reduce many layers -> one layer.
  • filters    (blur, feedback)          transform layer(s) in place.

Effects declare a PARAMS schema so the Gradio UI can build sliders generically —
that's the extension point: add an Effect subclass + register it, and it shows up
in the UI with no UI code changes.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from core import RenderContext


class Effect:
    name: str = "base"
    label: str = "Base"
    reduces: bool = False          # True -> collapses N layers into 1
    PARAMS: list[dict[str, Any]] = []   # UI/param schema (see interlace.py for example)

    def __init__(self, **params: Any):
        merged = {p["name"]: p.get("default") for p in self.PARAMS}
        merged.update({k: v for k, v in params.items() if k in merged})
        self.params = merged

    def apply(self, layers: list[np.ndarray], ctx: RenderContext) -> list[np.ndarray]:
        """Override me. Return a new list of layer arrays."""
        return layers

    # convenience -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "params": dict(self.params)}


# ── Registry ────────────────────────────────────────────────────────────────────
_REGISTRY: dict[str, type[Effect]] = {}


def register(cls: type[Effect]) -> type[Effect]:
    _REGISTRY[cls.name] = cls
    return cls


def available() -> list[type[Effect]]:
    return list(_REGISTRY.values())


def get(name: str) -> type[Effect]:
    return _REGISTRY[name]


def build(name: str, params: dict[str, Any] | None = None) -> Effect:
    return _REGISTRY[name](**(params or {}))


# ── Small shared numeric helpers ─────────────────────────────────────────────────
def stack_layers(layers: list[np.ndarray]) -> np.ndarray:
    """(L,) list of (T,H,W,3) -> (L, T, H, W, 3) float32."""
    return np.stack([l.astype(np.float32) for l in layers], axis=0)


def to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)
