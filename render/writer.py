"""
Write an (N, H, W, 3) uint8 RGB frame stack out to an mp4.

Uses imageio's bundled ffmpeg (imageio-ffmpeg) so there's no dependency on a
system ffmpeg install — a collaborator only needs `pip install -r requirements`.
"""
from __future__ import annotations

import os
import time

import imageio.v2 as imageio
import numpy as np

import config


def _even(x: int) -> int:
    return x - (x % 2)


def write_mp4(
    frames: np.ndarray,
    out_path: str | None = None,
    fps: int = config.DEFAULT_FPS,
    quality: int = 8,
) -> str:
    """Encode `frames` (N, H, W, 3 uint8 RGB) to H.264 mp4. Returns the path."""
    if frames is None or len(frames) == 0:
        raise ValueError("write_mp4 got an empty frame stack")

    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    # yuv420p (needed for broad playback) requires even dimensions.
    n, h, w = frames.shape[:3]
    eh, ew = _even(h), _even(w)
    if (eh, ew) != (h, w):
        frames = frames[:, :eh, :ew, :]

    if out_path is None:
        out_path = os.path.join(config.OUTPUT_DIR, f"render_{int(time.time())}.mp4")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    writer = imageio.get_writer(
        out_path,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=quality,
        pixelformat="yuv420p",
        macro_block_size=1,       # we already made dims even; don't auto-pad
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()
    return out_path
