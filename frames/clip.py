"""
Clip: decode a video file into a normalised RGB frame stack, with caching.

Frames are always returned as a single numpy array of shape (N, H, W, 3),
dtype uint8, RGB. Everything downstream (effects, blending, writing) speaks that
one format so there are no surprises about channel order.

Decoding is vectorised at the array level and cached both in memory (per process)
and on disk (compressed .npy) keyed by source mtime + working size + fps, so
repeated previews/renders don't re-decode.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import cv2
import numpy as np

import config


# ── Metadata probe ──────────────────────────────────────────────────────────────
@dataclass
class ClipInfo:
    path: str
    name: str
    width: int
    height: int
    fps: float
    n_frames: int
    duration: float


def probe(path: str) -> ClipInfo:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    dur = (n / fps) if fps else 0.0
    return ClipInfo(
        path=path, name=os.path.basename(path),
        width=w, height=h, fps=fps, n_frames=n, duration=dur,
    )


# ── In-memory cache ─────────────────────────────────────────────────────────────
_MEM_CACHE: dict[str, np.ndarray] = {}
_MEM_CACHE_ORDER: list[str] = []
_MEM_CACHE_MAX = 8  # keep a handful of decoded stacks resident


def _cache_key(path: str, fps: int, size: tuple[int, int]) -> str:
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0
    raw = (f"{os.path.abspath(path)}|{mtime}|{fps}|{size[0]}x{size[1]}"
           f"|cap{config.MAX_CLIP_SECONDS}")
    return hashlib.md5(raw.encode()).hexdigest()


def _mem_get(key: str) -> np.ndarray | None:
    return _MEM_CACHE.get(key)


def _mem_put(key: str, arr: np.ndarray) -> None:
    if key in _MEM_CACHE:
        return
    _MEM_CACHE[key] = arr
    _MEM_CACHE_ORDER.append(key)
    while len(_MEM_CACHE_ORDER) > _MEM_CACHE_MAX:
        old = _MEM_CACHE_ORDER.pop(0)
        _MEM_CACHE.pop(old, None)


# ── Core extraction ─────────────────────────────────────────────────────────────
def _resample_indices(n_src: int, src_fps: float, dst_fps: int, dst_len: int | None) -> np.ndarray:
    """Map destination frame indices back to source indices for fps resampling.

    Nearest-frame resample (vectorised) — cheap and fine for degraded footage.
    """
    if n_src <= 0:
        return np.zeros(0, dtype=np.int64)
    if not src_fps or src_fps <= 0:
        src_fps = float(dst_fps)
    if dst_len is None:
        duration = n_src / src_fps
        dst_len = max(1, int(round(duration * dst_fps)))
    t = np.arange(dst_len) / float(dst_fps)          # dst timestamps (s)
    src_idx = np.round(t * src_fps).astype(np.int64)
    return np.clip(src_idx, 0, n_src - 1)


def extract(
    path: str,
    width: int = config.DEFAULT_WIDTH,
    height: int = config.DEFAULT_HEIGHT,
    fps: int = config.DEFAULT_FPS,
    use_disk_cache: bool = True,
) -> np.ndarray:
    """Decode `path` to a normalised (N, H, W, 3) uint8 RGB stack at (width,
    height, fps). Cached in memory and (optionally) on disk."""
    size = (int(width), int(height))
    key = _cache_key(path, fps, size)

    cached = _mem_get(key)
    if cached is not None:
        return cached

    disk_path = os.path.join(config.CACHE_DIR, f"{key}.npy")
    if use_disk_cache and os.path.exists(disk_path):
        try:
            arr = np.load(disk_path)
            _mem_put(key, arr)
            return arr
        except Exception:
            pass  # corrupt cache -> re-decode

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    n_src = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Read source frames once (sequential decode is fastest), resize on the way.
    # Cap at MAX_CLIP_SECONDS so a long scraped video can't eat gigabytes of RAM.
    max_src_frames = int((src_fps or fps) * config.MAX_CLIP_SECONDS)
    src_frames: list[np.ndarray] = []
    while len(src_frames) < max_src_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        src_frames.append(frame)
    cap.release()

    if not src_frames:
        raise IOError(f"No frames decoded from: {path}")

    src = np.stack(src_frames, axis=0)               # (n_src, H, W, 3)
    n_src = src.shape[0]

    idx = _resample_indices(n_src, src_fps, fps, dst_len=None)
    arr = np.ascontiguousarray(src[idx])             # fps-resampled stack

    _mem_put(key, arr)
    if use_disk_cache:
        try:
            np.save(disk_path, arr)
        except Exception:
            pass
    return arr


# ── Thumbnails (for the Library tab) ─────────────────────────────────────────────
def thumbnail(path: str, max_w: int = 240) -> str:
    """Return a path to a cached JPEG thumbnail of the clip's first frame."""
    key = _cache_key(path, 0, (max_w, 0))
    thumb_path = os.path.join(config.CACHE_DIR, f"thumb_{key}.jpg")
    if os.path.exists(thumb_path):
        return thumb_path
    cap = cv2.VideoCapture(path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return path
    h, w = frame.shape[:2]
    scale = max_w / float(w) if w else 1.0
    frame = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
    cv2.imwrite(thumb_path, frame)
    return thumb_path
