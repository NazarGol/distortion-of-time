"""Prove the streaming pool weave (pool_weave.iter_frames) reproduces the
verified in-RAM weave.weave() bit-for-bit — using an in-memory lossless cache so
neither JPEG nor H.264 encoding is in the comparison path."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from weave import weave
from pool_weave import iter_frames


class MemCache:
    """video_id is the clip's index into `clips`; returns the raw array frame."""
    def __init__(self, clips): self.clips = clips
    def get(self, video_id, frame_number):
        return self.clips[int(video_id)][frame_number]


def run():
    rng = np.random.default_rng(7)
    H, W = 6, 17
    lens = [7, 5, 9]
    clips = []
    for v, L in enumerate(lens):
        c = np.empty((L, H, W, 3), np.uint8)
        for f in range(L):
            c[f] = rng.integers(0, 256, (H, W, 3))   # random → catches any misindex
        clips.append(c)
    videos = [{"video_id": str(v), "n_frames": lens[v]} for v in range(len(lens))]

    cases = [
        dict(strip_width=2, frame_step=1,  orientation="vertical",   source_mode="rotate", num_frames=11),
        dict(strip_width=1, frame_step=1,  orientation="vertical",   source_mode="rotate", num_frames=9),
        dict(strip_width=5, frame_step=3,  orientation="vertical",   source_mode="rotate", num_frames=12),
        dict(strip_width=2, frame_step=-2, orientation="vertical",   source_mode="rotate", num_frames=8),
        dict(strip_width=2, frame_step=1,  orientation="horizontal", source_mode="rotate", num_frames=10),
        dict(strip_width=3, frame_step=2,  orientation="vertical",   source_mode="single", num_frames=10),
        dict(strip_width=2, frame_step=0,  orientation="vertical",   source_mode="rotate", num_frames=6),
        dict(strip_width=4, frame_step=1,  orientation="horizontal", source_mode="single", num_frames=9),
    ]
    cache = MemCache(clips)
    for kw in cases:
        ref = weave(clips, strip_width=kw["strip_width"], frame_step=kw["frame_step"],
                    orientation=kw["orientation"], source_mode=kw["source_mode"],
                    num_frames=kw["num_frames"])
        got = np.stack(list(iter_frames(videos, cache, width=W, height=H, **kw)), axis=0)
        assert got.shape == ref.shape and (got == ref).all(), f"MISMATCH {kw}"
        print("bit-exact", kw)
    print("POOL_WEAVE == WEAVE.PY  ✓")


if __name__ == "__main__":
    run()
