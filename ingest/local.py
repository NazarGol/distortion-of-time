"""
Local ingest: register video files into the library.

This is the credential-free path. Upload files or point at a folder; clips are
copied into library/ and become available to the compositor. No Telegram, no
network, no secrets.
"""
from __future__ import annotations

import os
import shutil

import config


def _safe_name(name: str) -> str:
    base = os.path.basename(name)
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in base)


def list_library() -> list[str]:
    """Return absolute paths of all video clips currently in the library."""
    out = []
    for f in sorted(os.listdir(config.LIBRARY_DIR)):
        if f.lower().endswith(config.VIDEO_EXTS):
            out.append(os.path.join(config.LIBRARY_DIR, f))
    return out


def library_names() -> list[str]:
    return [os.path.basename(p) for p in list_library()]


def import_files(paths: list[str]) -> list[str]:
    """Copy uploaded files into the library. Returns names of imported clips."""
    imported = []
    for p in paths or []:
        if not p or not os.path.isfile(p):
            continue
        if not p.lower().endswith(config.VIDEO_EXTS):
            continue
        dest = os.path.join(config.LIBRARY_DIR, _safe_name(p))
        # avoid clobbering an existing clip of the same name
        stem, ext = os.path.splitext(dest)
        i = 1
        while os.path.exists(dest):
            dest = f"{stem}_{i}{ext}"
            i += 1
        shutil.copy2(p, dest)
        imported.append(os.path.basename(dest))
    return imported


def import_folder(folder: str, move: bool = False) -> list[str]:
    """Register every video in `folder` into the library (copy by default)."""
    if not folder or not os.path.isdir(folder):
        return []
    imported = []
    for f in sorted(os.listdir(folder)):
        src = os.path.join(folder, f)
        if not os.path.isfile(src) or not f.lower().endswith(config.VIDEO_EXTS):
            continue
        dest = os.path.join(config.LIBRARY_DIR, _safe_name(f))
        stem, ext = os.path.splitext(dest)
        i = 1
        while os.path.exists(dest):
            dest = f"{stem}_{i}{ext}"
            i += 1
        if move:
            shutil.move(src, dest)
        else:
            shutil.copy2(src, dest)
        imported.append(os.path.basename(dest))
    return imported
