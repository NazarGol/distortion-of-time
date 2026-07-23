"""
Launcher — serves the FastAPI backend + static instrument frontend with uvicorn.

    python app.py            # http://127.0.0.1:8000

The UI is a single static page (web/index.html); all logic lives in the Python
engine (pool, preprocess, weave, pool_weave) exposed through server.py.
"""
from __future__ import annotations

import uvicorn

import config
import pool  # noqa: F401 — ensures the pool/index exists on startup

HOST = "127.0.0.1"
PORT = 8000

if __name__ == "__main__":
    s = pool.stats()
    print(f"[Distortion of Time] pool: {s['clips']} clips / {s['frames']} frames "
          f"· {config.WORKING_WIDTH}×{config.WORKING_HEIGHT}")
    print(f"[Distortion of Time] http://{HOST}:{PORT}")
    uvicorn.run("server:app", host=HOST, port=PORT, log_level="warning")
