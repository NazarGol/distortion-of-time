"""
Optional Telegram scraper — a thin wrapper around the existing Corpus Editor
scraper (auth + rate-limiting + FloodWait handling reused, not reinvented).

Everything here is optional. If credentials or the scraper aren't present, the
functions return a friendly message and the rest of the app is unaffected — the
compositor never imports this module.

Reused scraper: editing_soft/corpus-editor/scraper/scraper.py -> scrape_channel().
That module does `import config` internally; since we also have a `config`
module, the import is isolated (its own dir on sys.path, our modules temporarily
removed, then restored) so the two never clash.
"""
from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from datetime import datetime, timezone

from dotenv import load_dotenv

import config as our_config

# The spec's <<PATH TO YOUR EXISTING CORPUS EDITOR SCRAPER>> resolved to:
CORPUS_SCRAPER_DIR = os.environ.get(
    "CORPUS_SCRAPER_DIR",
    "/home/new_admin/Desktop/Projects/editing_soft/corpus-editor/scraper",
)


# ── credentials / availability ───────────────────────────────────────────────────
def _creds() -> tuple[str | None, str | None, str | None]:
    load_dotenv(os.path.join(our_config.ROOT, ".env"))
    return (os.getenv("TG_API_ID"), os.getenv("TG_API_HASH"), os.getenv("TG_PHONE"))


def availability() -> tuple[bool, str]:
    """(ready?, message). Message is safe to show in the UI verbatim."""
    api_id, api_hash, _ = _creds()
    try:
        import telethon  # noqa: F401
    except Exception:
        return False, "⚠️ Telethon isn't installed. `pip install telethon` to enable scraping."
    if not api_id or not api_hash:
        return False, (
            "🔒 No Telegram credentials found. Copy `.env.example` → `.env` and add "
            "`TG_API_ID` / `TG_API_HASH` (from https://my.telegram.org) to enable "
            "scraping. **Everything else in the app works without this.**"
        )
    if not os.path.isfile(os.path.join(CORPUS_SCRAPER_DIR, "scraper.py")):
        return False, f"⚠️ Corpus Editor scraper not found at `{CORPUS_SCRAPER_DIR}`."
    return True, "✅ Credentials found — ready to scrape into `library/`."


# ── isolated import of the reused scraper ────────────────────────────────────────
def _load_scrape_channel():
    scraper_path = os.path.join(CORPUS_SCRAPER_DIR, "scraper.py")
    saved = {n: sys.modules.pop(n) for n in ("config", "db") if n in sys.modules}
    sys.path.insert(0, CORPUS_SCRAPER_DIR)
    try:
        spec = importlib.util.spec_from_file_location("corpus_scraper", scraper_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)          # runs module body -> its own config/db
        return mod.scrape_channel
    finally:
        try:
            sys.path.remove(CORPUS_SCRAPER_DIR)
        except ValueError:
            pass
        for n in ("config", "db", "corpus_scraper"):
            sys.modules.pop(n, None)
        sys.modules.update(saved)             # restore OUR config


# ── scrape ────────────────────────────────────────────────────────────────────────
async def _run(channel, date_from, date_to, max_videos, max_size_mb):
    from telethon import TelegramClient

    api_id, api_hash, _ = _creds()
    scrape_channel = _load_scrape_channel()

    # Reuse the Corpus Editor's already-authenticated session so no re-login is
    # needed (auth is created once via corpus-editor/scraper/auth_setup.py).
    session_path = os.path.join(CORPUS_SCRAPER_DIR, "tg_session")

    client = TelegramClient(session_path, int(api_id), api_hash)
    await client.connect()
    if not await client.is_user_authorized():
        await client.disconnect()
        return None, (
            "Telegram session isn't authorized. Run "
            "`python auth_setup.py` in the Corpus Editor scraper folder once to "
            "create a session, then retry."
        )
    try:
        n = await scrape_channel(
            client, channel, date_from, date_to,
            int(max_videos), float(max_size_mb), our_config.LIBRARY_DIR,
        )
    finally:
        await client.disconnect()
    return n, None


def scrape(channel: str, max_videos: int = 5,
           date_from: str = "2022-01-01", date_to: str = "2030-01-01",
           max_size_mb: float = 50.0) -> str:
    """Download up to `max_videos` videos from a public channel into library/.
    Returns a status string suitable for direct display."""
    ok, msg = availability()
    if not ok:
        return msg
    channel = (channel or "").strip().lstrip("@")
    if not channel:
        return "Enter a channel username (e.g. `some_channel`)."
    try:
        df = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
        dt = datetime.fromisoformat(date_to).replace(
            tzinfo=timezone.utc, hour=23, minute=59, second=59)
    except ValueError:
        return "Dates must be ISO format, e.g. 2024-06-13."
    try:
        n, err = asyncio.run(_run(channel, df, dt, max_videos, max_size_mb))
    except Exception as e:                      # noqa: BLE001 — surface any failure to the UI
        return f"Scrape failed: {e}"
    if err:
        return err
    return f"✅ Downloaded {n} video(s) from @{channel} into library/."
