"""
FastAPI backend — serves the static instrument frontend (web/index.html) and a
small JSON API over the frame pool and the streaming weave renderer.

No Gradio. The Python engine (pool, preprocess, pool_weave) is unchanged UI-side;
this only exposes it. Endpoints:

  GET  /                     the single-page frontend
  GET  /api/stats            {clips, frames, fps, width, height}
  GET  /api/corpus           [{video_id, name, n_frames, fps}] + thumbnails
  GET  /thumb/{video_id}     filmstrip thumbnail (jpeg)
  POST /api/import           multipart upload → library → decompose into pool
  POST /api/decompose        decompose any library clips not yet in the pool
  POST /api/scrape           optional Telegram scrape → library → decompose
  POST /api/render           run the weave; returns {url, ...metrics}
  GET  /out/{name}           serve a rendered mp4
"""
from __future__ import annotations

import os
import statistics
import time

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import pool
import pool_weave
from ingest import local as ingest_local
from ingest import telegram as ingest_tg

app = FastAPI(title="Distortion of Time")


def _working_fps(videos: list[dict]) -> float:
    """Corpus working fps = the most common source fps (each strip advances one
    frame, so we never silently downsample). Falls back to config."""
    fpss = [round(v["fps"], 3) for v in videos if v.get("fps")]
    if not fpss:
        return float(config.FPS_FALLBACK)
    try:
        return float(statistics.mode(fpss))
    except statistics.StatisticsError:
        return float(fpss[0])


# ── read endpoints ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
def index():
    with open(os.path.join(config.WEB_DIR, "index.html")) as f:
        return HTMLResponse(f.read())


@app.get("/api/stats")
def api_stats():
    s = pool.stats()
    vids = pool.list_videos()
    tg_ok, tg_msg = ingest_tg.availability()
    s.update({"fps": _working_fps(vids),
              "width": config.WORKING_WIDTH, "height": config.WORKING_HEIGHT,
              "telegram": {"ok": bool(tg_ok), "msg": tg_msg}})
    return s


@app.get("/api/corpus")
def api_corpus():
    vids = pool.list_videos()
    return {"videos": [{"video_id": v["video_id"], "name": v["name"],
                        "n_frames": v["n_frames"], "fps": v["fps"]} for v in vids],
            **pool.stats()}


@app.get("/thumb/{video_id}")
def api_thumb(video_id: str):
    path = pool.thumbnail(video_id)
    if not os.path.isfile(path):
        return JSONResponse({"error": "no thumbnail"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


@app.get("/out/{name}")
def api_out(name: str):
    path = os.path.join(config.OUTPUT_DIR, os.path.basename(name))
    if not os.path.isfile(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


# ── ingest endpoints ─────────────────────────────────────────────────────────────
@app.post("/api/import")
async def api_import(files: list[UploadFile] = File(...),
                     exposure: bool = Form(True)):
    saved = []
    for uf in files:
        dest = os.path.join(config.LIBRARY_DIR, ingest_local._safe_name(uf.filename))
        stem, ext = os.path.splitext(dest)
        i = 1
        while os.path.exists(dest):
            dest = f"{stem}_{i}{ext}"; i += 1
        if not dest.lower().endswith(config.VIDEO_EXTS):
            continue
        with open(dest, "wb") as out:
            out.write(await uf.read())
        saved.append(dest)
    added = 0
    for p in saved:
        pool.decompose(p, exposure=bool(exposure))
        added += 1
    return {"imported": added, **pool.stats()}


@app.post("/api/decompose")
def api_decompose(exposure: bool = Form(True)):
    added = pool.decompose_library(exposure=bool(exposure))
    return {"decomposed": added, **pool.stats()}


@app.post("/api/scrape")
def api_scrape(channel: str = Form(...), count: int = Form(5),
               exposure: bool = Form(True)):
    msg = ingest_tg.scrape(channel, int(count))
    added = pool.decompose_library(exposure=bool(exposure))
    return {"message": msg, "decomposed": added, **pool.stats()}


# ── render ───────────────────────────────────────────────────────────────────────
@app.post("/api/render")
def api_render(
    strip_width: int = Form(2),
    span_seconds: float = Form(1.5),
    direction: str = Form("forward"),        # forward | back
    orientation: str = Form("vertical"),     # vertical | horizontal
    draw_from: str = Form("corpus"),         # one | corpus
    video_id: str = Form(""),
    preview: bool = Form(False),
):
    vids = pool.list_videos()
    if not vids:
        return JSONResponse({"error": "The pool is empty — add clips first."}, status_code=400)

    if draw_from == "one":
        chosen = next((v for v in vids if v["video_id"] == video_id), vids[0])
        videos, source_mode = [chosen], "single"
        fps = float(chosen["fps"] or config.FPS_FALLBACK)
    else:
        videos, source_mode = vids, "rotate"
        fps = _working_fps(vids)

    axis_len = config.WORKING_WIDTH if orientation != "horizontal" else config.WORKING_HEIGHT
    n_strips = (axis_len + max(1, strip_width) - 1) // max(1, strip_width)
    # artistic control: the frame should hold `span_seconds` of time across its
    # width → total displacement = span*fps frames spread over n_strips strips.
    frame_step = (span_seconds * fps / n_strips) if n_strips else 1.0
    if direction == "back":
        frame_step = -frame_step

    num_frames = config.PREVIEW_MAX_FRAMES if preview else None
    out_path = os.path.join(config.OUTPUT_DIR,
                            f"{'preview' if preview else 'render'}_{int(time.time())}.mp4")
    m = pool_weave.render_weave(
        videos, out_path, strip_width=int(strip_width), frame_step=frame_step,
        orientation=orientation, source_mode=source_mode, fps=fps,
        num_frames=num_frames)
    m["url"] = "/out/" + os.path.basename(out_path)
    m["fps"] = fps
    m["frame_step"] = round(frame_step, 3)
    m["span_seconds"] = span_seconds
    return m


if os.path.isdir(config.WEB_DIR):
    app.mount("/web", StaticFiles(directory=config.WEB_DIR), name="web")
