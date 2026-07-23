"""
Optional monocular depth — powers `parallax = depth` and (once present) a
depth-based blur.

Everything here is lazy and optional. torch + transformers are NOT in the base
install; if they're missing, `available()` is False, the UI disables the depth
option with a note, and the rest of the app is unaffected. Install with:

    pip install -r requirements-depth.txt

Depth maps are computed once per pool frame and cached on disk beside the frames
(`pool/<video_id>/depth/000123.png`, 16-bit), so a second render is free.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

import config

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"

_pipe = None
_checked = False
_reason = "not checked"


def available() -> tuple[bool, str]:
    """(usable?, human-readable reason). Never raises, never imports torch unless
    the dependency is actually present."""
    global _checked, _reason
    if _checked:
        return (_pipe is not None or _reason == "ready"), _reason
    _checked = True
    import importlib.util as u
    if not u.find_spec("torch") or not u.find_spec("transformers"):
        _reason = "depth needs torch + transformers (pip install -r requirements-depth.txt)"
        return False, _reason
    _reason = "ready"
    return True, _reason


def _get_pipe():
    global _pipe, _reason
    if _pipe is not None:
        return _pipe
    ok, _ = available()
    if not ok:
        return None
    try:
        import torch
        from transformers import pipeline
        dev = 0 if torch.cuda.is_available() else -1
        _pipe = pipeline("depth-estimation", model=MODEL_ID, device=dev)
    except Exception as e:                      # model download/load failure
        _reason = f"depth model unavailable: {e}"
        _pipe = None
    return _pipe


def _cache_path(video_id: str, frame_no: int) -> str:
    d = os.path.join(config.POOL_DIR, video_id, "depth")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{frame_no:06d}.png")


def depth_for(video_id: str | None, frame_no: int | None,
              img: np.ndarray) -> np.ndarray | None:
    """Normalised (H, W) float32 depth in [0,1] (1 = nearest), or None if the
    optional dependency isn't installed. Cached on disk per pool frame."""
    ok, _ = available()
    if not ok:
        return None

    path = _cache_path(video_id, frame_no) if video_id is not None and frame_no is not None else None
    if path and os.path.isfile(path):
        cached = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if cached is not None:
            return cached.astype(np.float32) / 65535.0

    pipe = _get_pipe()
    if pipe is None:
        return None
    try:
        from PIL import Image
        out = pipe(Image.fromarray(img))["depth"]
        d = np.asarray(out, dtype=np.float32)
    except Exception:
        return None
    if d.shape[:2] != img.shape[:2]:
        d = cv2.resize(d, (img.shape[1], img.shape[0]))
    lo, hi = float(d.min()), float(d.max())
    d = (d - lo) / (hi - lo) if hi > lo else np.zeros_like(d)
    if path:
        cv2.imwrite(path, (d * 65535.0).astype(np.uint16))
    return d
