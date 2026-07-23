"""
The frame pool — the material the weave stitches from.

Every video that enters the library is decomposed ONCE into individual, cleaned,
working-resolution JPEG frames on disk:

    pool/<video_id>/000123.jpg

A SQLite index records every frame (frame_id, video_id, frame_number, timestamp,
path) and every video's metadata. Decomposition is idempotent — a video already
in the index is never redone.

The weave never loads whole videos into RAM: it asks the pool for exactly the
frames a given output frame needs, through a small bounded LRU cache
(`FrameCache`). Memory stays flat regardless of how large the corpus grows.
"""
from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from collections import OrderedDict
from typing import Callable

import cv2
import numpy as np

import config
import preprocess


# ── index ────────────────────────────────────────────────────────────────────────
def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(config.INDEX_DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS videos(
            video_id  TEXT PRIMARY KEY,
            name      TEXT,
            source_path TEXT,
            fps       REAL,
            n_frames  INTEGER,
            width     INTEGER,
            height    INTEGER,
            crop      TEXT,
            exposure  INTEGER,
            added_at  REAL
        );
        CREATE TABLE IF NOT EXISTS frames(
            frame_id     INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id     TEXT,
            frame_number INTEGER,
            timestamp    REAL,
            path         TEXT
        );
        CREATE INDEX IF NOT EXISTS ix_frames_vid ON frames(video_id, frame_number);
        """
    )
    return con


def video_id_for(path: str) -> str:
    """Stable id from source identity (path + size + mtime) — decompose keys on it
    so re-importing the same file is a no-op."""
    ap = os.path.abspath(path)
    try:
        st = os.stat(ap)
        sig = f"{ap}|{st.st_size}|{int(st.st_mtime)}"
    except OSError:
        sig = ap
    return "v" + hashlib.sha1(sig.encode()).hexdigest()[:12]


def frame_path(video_id: str, frame_number: int) -> str:
    return os.path.join(config.POOL_DIR, video_id, f"{frame_number:06d}.jpg")


# ── decompose ────────────────────────────────────────────────────────────────────
def is_indexed(video_id: str) -> bool:
    con = _connect()
    try:
        row = con.execute("SELECT n_frames FROM videos WHERE video_id=?",
                           (video_id,)).fetchone()
    finally:
        con.close()
    if not row:
        return False
    n = row[0] or 0
    # sanity: the last expected frame file exists
    return n > 0 and os.path.isfile(frame_path(video_id, n - 1))


def decompose(path: str, exposure: bool = True,
              progress: Callable[[int, int], None] | None = None) -> dict:
    """Decompose one video into the pool (idempotent). Returns its video record."""
    vid = video_id_for(path)
    name = os.path.basename(path)
    if is_indexed(vid):
        return get_video(vid)

    prep = preprocess.analyze(path)
    out_dir = os.path.join(config.POOL_DIR, vid)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Could not open video: {path}")

    max_frames = int(prep.fps * config.MAX_CLIP_SECONDS) if prep.fps else prep.n_frames
    enc = [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY]

    frame_rows = []
    written = 0
    while written < (max_frames or 10 ** 9):
        ok, frame = cap.read()
        if not ok:
            break
        clean = preprocess.process_frame(frame, prep, exposure)          # RGB
        fp = frame_path(vid, written)
        cv2.imwrite(fp, cv2.cvtColor(clean, cv2.COLOR_RGB2BGR), enc)
        frame_rows.append((vid, written, written / prep.fps if prep.fps else 0.0,
                           os.path.relpath(fp, config.ROOT)))
        written += 1
        if progress and written % 25 == 0:
            progress(written, max_frames or 0)
    cap.release()

    con = _connect()
    try:
        con.execute(
            "INSERT OR REPLACE INTO videos"
            "(video_id,name,source_path,fps,n_frames,width,height,crop,exposure,added_at)"
            " VALUES(?,?,?,?,?,?,?,?,?,?)",
            (vid, name, os.path.abspath(path), prep.fps, written,
             config.WORKING_WIDTH, config.WORKING_HEIGHT,
             ",".join(map(str, prep.crop)), int(exposure), time.time()),
        )
        con.execute("DELETE FROM frames WHERE video_id=?", (vid,))
        con.executemany(
            "INSERT INTO frames(video_id,frame_number,timestamp,path) VALUES(?,?,?,?)",
            frame_rows,
        )
        con.commit()
    finally:
        con.close()
    return get_video(vid)


def decompose_library(exposure: bool = True,
                      progress: Callable[[str, int, int], None] | None = None) -> int:
    """Decompose every not-yet-indexed clip in the library. Returns count added."""
    from ingest import local as ingest_local
    added = 0
    for path in ingest_local.list_library():
        vid = video_id_for(path)
        if is_indexed(vid):
            continue
        decompose(path, exposure=exposure,
                  progress=(lambda w, n, p=path: progress(os.path.basename(p), w, n))
                  if progress else None)
        added += 1
    return added


# ── queries ──────────────────────────────────────────────────────────────────────
def _row_to_video(r: sqlite3.Row) -> dict:
    return {"video_id": r["video_id"], "name": r["name"], "fps": r["fps"],
            "n_frames": r["n_frames"], "width": r["width"], "height": r["height"],
            "added_at": r["added_at"]}


def list_videos() -> list[dict]:
    con = _connect()
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT * FROM videos ORDER BY added_at, video_id").fetchall()
    finally:
        con.close()
    return [_row_to_video(r) for r in rows]


def get_video(video_id: str) -> dict:
    con = _connect()
    con.row_factory = sqlite3.Row
    try:
        r = con.execute("SELECT * FROM videos WHERE video_id=?",
                        (video_id,)).fetchone()
    finally:
        con.close()
    return _row_to_video(r) if r else {}


def stats() -> dict:
    con = _connect()
    try:
        nv = con.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
        nf = con.execute("SELECT COALESCE(SUM(n_frames),0) FROM videos").fetchone()[0]
    finally:
        con.close()
    return {"clips": int(nv), "frames": int(nf)}


# ── removal / reset ──────────────────────────────────────────────────────────────
def remove_video(video_id: str, delete_source: bool = False) -> dict:
    """Remove one video from the pool: delete its frames from disk, its index
    rows, and its thumbnail. Optionally delete the source file too (only when it
    lives in library/ — never touch files elsewhere)."""
    import shutil
    rec = get_video(video_id)
    con = _connect()
    try:
        row = con.execute("SELECT source_path FROM videos WHERE video_id=?",
                          (video_id,)).fetchone()
        src = row[0] if row else None
        con.execute("DELETE FROM frames WHERE video_id=?", (video_id,))
        con.execute("DELETE FROM videos WHERE video_id=?", (video_id,))
        con.commit()
    finally:
        con.close()
    shutil.rmtree(os.path.join(config.POOL_DIR, video_id), ignore_errors=True)
    thumb = os.path.join(config.CACHE_DIR, f"thumb_{video_id}.jpg")
    if os.path.isfile(thumb):
        os.remove(thumb)
    if (delete_source and src and os.path.isfile(src)
            and os.path.dirname(os.path.abspath(src)) == config.LIBRARY_DIR):
        os.remove(src)
    return rec


def wipe(delete_library: bool = False) -> dict:
    """Reset the pool back to empty. With delete_library, also remove the source
    clips from library/ (a clean slate to start from your own footage)."""
    import shutil
    for v in list_videos():
        remove_video(v["video_id"], delete_source=delete_library)
    for name in os.listdir(config.POOL_DIR):        # sweep any stray frame dirs
        p = os.path.join(config.POOL_DIR, name)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    con = _connect()
    try:
        con.executescript("DELETE FROM frames; DELETE FROM videos;")
        con.commit()
    finally:
        con.close()
    return stats()


def thumbnail(video_id: str) -> str:
    """Path to a small JPEG thumbnail (first pool frame), generated on demand."""
    thumb = os.path.join(config.CACHE_DIR, f"thumb_{video_id}.jpg")
    if os.path.isfile(thumb):
        return thumb
    src = frame_path(video_id, 0)
    if not os.path.isfile(src):
        return src
    img = cv2.imread(src)
    if img is None:
        return src
    h, w = img.shape[:2]
    tw = config.THUMB_WIDTH
    img = cv2.resize(img, (tw, max(1, int(h * tw / w))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(thumb, img, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return thumb


# ── bounded LRU frame cache (the whole point: flat memory) ───────────────────────
class FrameCache:
    """Decode-on-demand LRU over pool frames. Holds at most `capacity` decoded
    (H, W, 3) RGB frames; memory is bounded no matter how big the corpus is."""

    def __init__(self, capacity: int = config.FRAME_CACHE_SIZE):
        self.capacity = int(capacity)
        self._d: "OrderedDict[tuple[str,int], np.ndarray]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, video_id: str, frame_number: int) -> np.ndarray:
        key = (video_id, frame_number)
        img = self._d.get(key)
        if img is not None:
            self._d.move_to_end(key)
            self.hits += 1
            return img
        self.misses += 1
        bgr = cv2.imread(frame_path(video_id, frame_number))
        if bgr is None:
            img = np.zeros((config.WORKING_HEIGHT, config.WORKING_WIDTH, 3), np.uint8)
        else:
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._d[key] = img
        if len(self._d) > self.capacity:
            self._d.popitem(last=False)
        return img
