"""
Project-wide paths and defaults for the Distortion of Time video tool.

Everything here is credential-free. The Telegram scraper reads its own secrets
from .env; the compositor never touches those.
"""
from __future__ import annotations

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

LIBRARY_DIR = os.path.join(ROOT, "library")   # source clips land here (import/scrape)
POOL_DIR    = os.path.join(ROOT, "pool")       # decomposed frames on disk — the material
CACHE_DIR   = os.path.join(ROOT, "cache")      # thumbnails
OUTPUT_DIR  = os.path.join(ROOT, "output")     # rendered mp4s

INDEX_DB = os.path.join(POOL_DIR, "index.db")  # SQLite index over the frame pool
WEB_DIR  = os.path.join(ROOT, "web")           # static frontend

for _d in (LIBRARY_DIR, POOL_DIR, CACHE_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

# ── Working space ──────────────────────────────────────────────────────────────
# Every video is decomposed to this common resolution at pool-ingest time, so all
# pool frames share H×W and the weave can stitch strips from any of them. This is
# fixed corpus-wide (not a per-render knob) — 16:9, sized for degraded footage.
WORKING_WIDTH  = 640
WORKING_HEIGHT = 360

# Working fps = each source clip's own fps (probed at decompose). Fall back to
# this when a container doesn't report one. Never silently downsample — each strip
# advances one *frame*, so fps is the effect's temporal resolution.
FPS_FALLBACK = 24

# Long sources are truncated at decompose so one video can't dominate the pool.
MAX_CLIP_SECONDS = 90

# ── Pool storage / streaming render ────────────────────────────────────────────
JPEG_QUALITY     = 85    # pool frames are JPEG (degraded footage — lossy is fine)
FRAME_CACHE_SIZE = 512   # decoded pool frames kept resident during a render (LRU)
THUMB_WIDTH      = 160    # filmstrip thumbnail width

# ── Preprocessing ──────────────────────────────────────────────────────────────
BAR_LUMA_THRESH  = 18    # a row/col staying below this (0–255) across sampled
                          # frames is treated as a letterbox/pillarbox bar
BAR_SAMPLE_FRAMES = 12   # frames sampled per video for bar + exposure analysis
EXPOSURE_TARGET_MEAN = 112.0   # normalise each video's luminance toward these so
EXPOSURE_TARGET_STD  = 52.0    # adjacent strips from different sources don't clash

# ── Preview ────────────────────────────────────────────────────────────────────
PREVIEW_MAX_FRAMES = 48   # a Preview render is capped to this many output frames
