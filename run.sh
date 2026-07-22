#!/usr/bin/env bash
# Launch the Distortion of Time tool. Creates the venv on first run.
set -e
cd "$(dirname "$0")"

if [ ! -d venv ]; then
  echo "Creating venv and installing requirements (first run only)…"
  python3 -m venv venv
  ./venv/bin/python -m pip install --upgrade pip
  ./venv/bin/python -m pip install -r requirements.txt
fi

exec ./venv/bin/python app.py
