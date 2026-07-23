"""
FastAPI backend — serves the static instrument frontend (web/index.html) and a
small JSON API over the frame pool and the streaming weave renderer.

No Gradio. Effects are siblings (see effects/ and pipeline.py): a render picks one
generator — weave | superimpose | frames — then optionally applies post stages
(hyper-imposition, blur). Endpoints:

  GET  /                     the single-page frontend
  GET  /api/stats            {clips, frames, fps, width, height}
  GET  /api/corpus           [{video_id, name, n_frames, fps}] + thumbnails
  GET  /thumb/{video_id}     filmstrip thumbnail (jpeg)
  POST /api/import           multipart upload → library → decompose into pool
  POST /api/decompose        decompose any library clips not yet in the pool
  POST /api/scrape           optional Telegram scrape → library → decompose
  POST /api/remove           drop one video from the pool
  POST /api/wipe             reset the pool
  POST /api/render           run the chosen effect chain; returns {url, ...metrics}
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
import pipeline
import pool
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
    from effects import blur as blurmod, superimpose as supmod, depth as depthmod
    from effects import parallax as pxmod, feedback as fbmod
    d_ok, d_why = depthmod.available()
    s.update({"fps": _working_fps(vids),
              "width": config.WORKING_WIDTH, "height": config.WORKING_HEIGHT,
              "telegram": {"ok": bool(tg_ok), "msg": tg_msg},
              "blends": list(supmod.BLENDS),
              "parallax_modes": list(pxmod.MODES),
              "trail_modes": list(fbmod.MODES),
              "blur_kinds": list(blurmod.KINDS),
              "blur_guide_only": list(blurmod.GUIDE_ONLY),
              "depth": {"ok": bool(d_ok), "msg": d_why}})
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


@app.post("/api/remove")
def api_remove(video_id: str = Form(...), delete_source: bool = Form(False)):
    rec = pool.remove_video(video_id, delete_source=bool(delete_source))
    return {"removed": rec.get("name", video_id), **pool.stats()}


@app.post("/api/wipe")
def api_wipe(delete_library: bool = Form(False)):
    return pool.wipe(delete_library=bool(delete_library))


# ── render ───────────────────────────────────────────────────────────────────────
@app.post("/api/render")
def api_render(
    mode: str = Form("weave"),               # weave | superimpose | frames
    video_ids: str = Form(""),               # ordered, comma-separated selection
    guide_id: str = Form(""),                # optional guide clip (blur structure)
    # weave
    strip_width: int = Form(2),
    span_seconds: float = Form(1.5),
    direction: str = Form("forward"),        # forward | back
    orientation: str = Form("vertical"),     # vertical | horizontal
    # superimpose
    layers: int = Form(1),
    spread: float = Form(0.0),
    blend: str = Form("mean"),
    parallax: str = Form("none"),
    amount: float = Form(0.0),
    px_zoom: bool = Form(True),
    px_pan: bool = Form(False),
    px_rotate: bool = Form(False),
    # post: hyper-imposition
    persistence: float = Form(0.0),
    trail_mode: str = Form("mean"),
    trail_zoom: float = Form(0.0),
    trail_rotate: float = Form(0.0),
    # post: blur
    blur_kind: str = Form(""),
    blur_amount: float = Form(0.0),
    preview: bool = Form(False),
):
    pooled = {v["video_id"]: v for v in pool.list_videos()}
    if not pooled:
        return JSONResponse({"error": "The pool is empty — add clips first."}, status_code=400)

    ids = [i for i in (video_ids.split(",") if video_ids else []) if i in pooled]
    if not ids:
        return JSONResponse({"error": "Select at least one clip in the filmstrip."},
                            status_code=400)
    videos = [pooled[i] for i in ids]
    fps = float(videos[0]["fps"] or config.FPS_FALLBACK) if len(videos) == 1 \
        else _working_fps(videos)

    params = {"mode": mode if mode in ("weave", "superimpose", "frames") else "weave",
              "fps": fps,
              "persistence": persistence, "trail_mode": trail_mode,
              "trail_zoom": trail_zoom, "trail_rotate": trail_rotate,
              "blur_kind": blur_kind, "blur_amount": blur_amount}

    if guide_id and guide_id in pooled:
        params["guide_id"] = guide_id
        params["guide_len"] = int(pooled[guide_id]["n_frames"])

    if params["mode"] == "weave":
        # one clip → slit-scan (single); many → rotation across them, in order.
        params["source_mode"] = "single" if len(videos) == 1 else "rotate"
        axis_len = (config.WORKING_WIDTH if orientation != "horizontal"
                    else config.WORKING_HEIGHT)
        n_strips = (axis_len + max(1, strip_width) - 1) // max(1, strip_width)
        # the frame holds `span_seconds` of time across its width
        step = (span_seconds * fps / n_strips) if n_strips else 1.0
        params.update({"strip_width": int(strip_width),
                       "frame_step": -step if direction == "back" else step,
                       "orientation": orientation})
    elif params["mode"] == "superimpose":
        params.update({"layers": int(layers), "spread": float(spread),
                       "blend": blend, "parallax": parallax, "amount": float(amount),
                       "px_zoom": px_zoom, "px_pan": px_pan, "px_rotate": px_rotate})

    num_frames = config.PREVIEW_MAX_FRAMES if preview else None
    out_path = os.path.join(config.OUTPUT_DIR,
                            f"{'preview' if preview else 'render'}_{int(time.time())}.mp4")
    m = pipeline.render(videos, out_path, params, num_frames=num_frames)
    m["mode"] = params["mode"]
    if params["mode"] == "weave":
        m["frame_step"] = round(params["frame_step"], 3)
        m["span_seconds"] = span_seconds
    return m


if os.path.isdir(config.WEB_DIR):
    app.mount("/web", StaticFiles(directory=config.WEB_DIR), name="web")
