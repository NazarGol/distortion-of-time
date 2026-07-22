"""
Project-wide paths and defaults for the Distortion of Time video tool.

Everything here is credential-free. The Telegram scraper reads its own secrets
from .env; the compositor never touches those.
"""
from __future__ import annotations

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

LIBRARY_DIR      = os.path.join(ROOT, "library")       # clips live here
CACHE_DIR        = os.path.join(ROOT, "cache")         # decoded-frame + thumb cache
OUTPUT_DIR       = os.path.join(ROOT, "output")        # rendered mp4s
COMPOSITIONS_DIR = os.path.join(ROOT, "compositions")  # saved composition presets (JSON)

for _d in (LIBRARY_DIR, CACHE_DIR, OUTPUT_DIR, COMPOSITIONS_DIR):
    os.makedirs(_d, exist_ok=True)

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

# ── Working-space defaults ─────────────────────────────────────────────────────
# The compositor normalises every layer to a common working resolution + fps.
# Degraded conflict footage is low-res anyway, so modest defaults keep the look
# and stay fast.
DEFAULT_WIDTH  = 640
DEFAULT_HEIGHT = 360
DEFAULT_FPS    = 12

# Preview runs on a short, downscaled segment so the UI stays responsive.
PREVIEW_MAX_FRAMES = 36
PREVIEW_MAX_WIDTH  = 384
