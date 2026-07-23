"""
Preprocessing applied to every video as it enters the frame pool.

Two problems in raw (esp. Telegram) footage produce the black bands and harsh
strip noise the weave must avoid — both are fixed here, at decompose time, so the
pool holds only clean, coherent material:

  1. Letterbox / pillarbox bars. Padded footage has pure-black edges; a strip
     sampled inside a bar is a black strip. `detect_bars` finds the lit content
     rectangle (per video) and we crop to it.
  2. Framing / exposure mismatch across sources. After bar-crop we centre-crop to
     the common working aspect (never squash) and normalise each video's
     brightness/contrast toward a shared target so adjacent strips from different
     videos don't alternate harshly.

All operations are whole-frame numpy / OpenCV — no per-pixel Python loops.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class VideoPrep:
    crop: tuple[int, int, int, int]      # (x0, y0, x1, y1) content rect in source px
    exp_gain: float                       # exposure: out = in * gain + bias
    exp_bias: float
    fps: float
    n_frames: int
    src_width: int
    src_height: int


# ── bar detection ────────────────────────────────────────────────────────────────
def detect_bars(luma_max: np.ndarray) -> tuple[int, int, int, int]:
    """Given the per-pixel MAX luminance over sampled frames, return the content
    rectangle (x0, y0, x1, y1). A row/col that never rises above the threshold in
    any sampled frame is a bar and is trimmed from the edges."""
    h, w = luma_max.shape[:2]
    thr = config.BAR_LUMA_THRESH
    row_lit = luma_max.max(axis=1) > thr     # (h,) — row has content somewhere
    col_lit = luma_max.max(axis=0) > thr     # (w,)

    def _bounds(lit: np.ndarray, n: int) -> tuple[int, int]:
        idx = np.flatnonzero(lit)
        if idx.size == 0:
            return 0, n                        # all dark → don't crop this axis
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        # only trust the crop if meaningful content remains (guards a dark clip)
        if (hi - lo) < 0.25 * n:
            return 0, n
        return lo, hi

    y0, y1 = _bounds(row_lit, h)
    x0, x1 = _bounds(col_lit, w)
    return x0, y0, x1, y1


# ── analysis pass ────────────────────────────────────────────────────────────────
def analyze(path: str) -> VideoPrep:
    """Sample a video: probe fps/size/length, detect bars, measure exposure."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not fps or fps <= 0:
        fps = float(config.FPS_FALLBACK)
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    n_cap = min(n_src, int(fps * config.MAX_CLIP_SECONDS)) if n_src else 0
    sample_at = (np.linspace(0, max(n_cap - 1, 0), config.BAR_SAMPLE_FRAMES).astype(int)
                 if n_cap else np.arange(config.BAR_SAMPLE_FRAMES))

    luma_max = None
    lumas = []
    for fno in sample_at:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        luma_max = gray if luma_max is None else np.maximum(luma_max, gray)
        lumas.append(gray)
    cap.release()

    if luma_max is None:                       # unreadable — safe no-op prep
        return VideoPrep((0, 0, max(w, 1), max(h, 1)), 1.0, 0.0, fps, n_cap, w, h)

    x0, y0, x1, y1 = detect_bars(luma_max)

    # exposure stats measured on the cropped content region only
    crop_lumas = np.stack([g[y0:y1, x0:x1] for g in lumas]).astype(np.float32)
    mean = float(crop_lumas.mean())
    std = float(crop_lumas.std())
    gain = config.EXPOSURE_TARGET_STD / max(std, 1e-3)
    # keep gain sane so a flat/near-black clip isn't blown up into noise
    gain = float(np.clip(gain, 0.4, 2.5))
    bias = config.EXPOSURE_TARGET_MEAN - mean * gain
    return VideoPrep((x0, y0, x1, y1), gain, bias, fps, n_cap, w, h)


# ── per-frame processing (decode → pool frame) ───────────────────────────────────
def _aspect_crop(img: np.ndarray, tw: int, th: int) -> np.ndarray:
    """Centre-crop to the target aspect (tw:th) without squashing."""
    h, w = img.shape[:2]
    target = tw / th
    cur = w / h
    if cur > target:                            # too wide → trim sides
        nw = max(1, int(round(h * target)))
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    if cur < target:                            # too tall → trim top/bottom
        nh = max(1, int(round(w / target)))
        y0 = (h - nh) // 2
        return img[y0:y0 + nh, :]
    return img


def process_frame(frame_bgr: np.ndarray, prep: VideoPrep, exposure: bool) -> np.ndarray:
    """Source BGR frame → clean (WORKING_H, WORKING_W, 3) uint8 RGB pool frame:
    bar-crop → aspect centre-crop → resize → optional exposure match."""
    x0, y0, x1, y1 = prep.crop
    img = frame_bgr[y0:y1, x0:x1]
    img = _aspect_crop(img, config.WORKING_WIDTH, config.WORKING_HEIGHT)
    img = cv2.resize(img, (config.WORKING_WIDTH, config.WORKING_HEIGHT),
                     interpolation=cv2.INTER_AREA)
    if exposure:
        img = np.clip(img.astype(np.float32) * prep.exp_gain + prep.exp_bias, 0, 255)
        img = img.astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
