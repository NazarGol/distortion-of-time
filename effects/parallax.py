"""
Parallax — a spatial modifier applied to each layer of a superimposition before
blending, so the stack reads as depth rather than a flat ghost.

Modes
  none    identity.
  affine  layer j gets an incremental zoom / pan / rotate (j × increment).
          Cheap, no dependencies — the tunnel / receding-ghost look.
  flow    Farneback optical flow of the footage's own motion; layer j is
          displaced along those vectors by amount × j. Fast-moving content
          separates, static background holds. CUDA path when torch+CUDA exists.
  depth   monocular depth (see effects/depth.py); layer j displaced horizontally
          by depth × amount × j. True parallax. Optional dependency — when the
          model isn't installed this mode reports unavailable and the caller
          falls back.

`begin_frame()` does any per-output-frame precomputation ONCE (flow is computed
a single time per frame, not per layer); `apply()` then warps each layer. All
warps are whole-image OpenCV ops — no per-pixel Python loops.
"""
from __future__ import annotations

import cv2
import numpy as np

MODES = ("none", "affine", "flow", "depth")


class Parallax:
    def __init__(self, mode: str = "none", amount: float = 0.0, *,
                 zoom: bool = True, pan: bool = False, rotate: bool = False,
                 fps: float = 24.0):
        self.mode = mode if mode in MODES else "none"
        self.amount = float(amount)
        self.use_zoom, self.use_pan, self.use_rotate = bool(zoom), bool(pan), bool(rotate)
        self.fps = float(fps)
        self._flow = None
        self._grid = None

    # ── per-output-frame precomputation ──────────────────────────────────────
    def begin_frame(self, base_img: np.ndarray, next_img: np.ndarray | None) -> None:
        if self.mode != "flow" or self.amount == 0.0 or next_img is None:
            self._flow = None
            return
        # one Farneback per output frame, at half resolution for speed
        h, w = base_img.shape[:2]
        a = cv2.cvtColor(cv2.resize(base_img, (w // 2, h // 2)), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(cv2.resize(next_img, (w // 2, h // 2)), cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow = cv2.resize(flow, (w, h)) * 2.0        # back to full res, rescale vectors
        self._flow = flow

    # ── per-layer warp ───────────────────────────────────────────────────────
    def apply(self, img: np.ndarray, j: int, *, video_id: str | None = None,
              frame_no: int | None = None) -> np.ndarray:
        if j == 0 or self.mode == "none" or self.amount == 0.0:
            return img
        if self.mode == "affine":
            return self._affine(img, j)
        if self.mode == "flow":
            return self._flow_warp(img, j)
        if self.mode == "depth":
            return self._depth_warp(img, j, video_id, frame_no)
        return img

    def _affine(self, img: np.ndarray, j: int) -> np.ndarray:
        h, w = img.shape[:2]
        k = self.amount
        scale = 1.0 + (j * k * 0.05) if self.use_zoom else 1.0
        angle = (j * k * 1.5) if self.use_rotate else 0.0
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        if self.use_pan:
            m[0, 2] += j * k * w * 0.01
            m[1, 2] += j * k * h * 0.005
        return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)

    def _base_grid(self, h: int, w: int):
        if self._grid is None or self._grid[0].shape != (h, w):
            gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                                 np.arange(h, dtype=np.float32))
            self._grid = (gx, gy)
        return self._grid

    def _flow_warp(self, img: np.ndarray, j: int) -> np.ndarray:
        if self._flow is None:
            return img
        h, w = img.shape[:2]
        gx, gy = self._base_grid(h, w)
        s = self.amount * j
        mx = (gx + self._flow[..., 0] * s).astype(np.float32)
        my = (gy + self._flow[..., 1] * s).astype(np.float32)
        return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def _depth_warp(self, img: np.ndarray, j: int, video_id, frame_no) -> np.ndarray:
        from effects import depth as depthmod
        d = depthmod.depth_for(video_id, frame_no, img)
        if d is None:
            return img
        h, w = img.shape[:2]
        gx, gy = self._base_grid(h, w)
        # near (depth→1) shifts most; scale to a sensible pixel budget
        shift = (d.astype(np.float32) * (self.amount * j * 0.06 * w))
        mx = (gx + shift).astype(np.float32)
        return cv2.remap(img, mx, gy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
