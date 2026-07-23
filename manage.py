"""
Pool maintenance CLI.

    python manage.py list                     # every pooled video + stats
    python manage.py remove <video_id>        # drop one video's frames + index row
    python manage.py remove <video_id> --source   # …and delete its library file
    python manage.py wipe --yes               # reset the pool to empty
    python manage.py wipe --library --yes     # …and clear library/ too (clean slate)

The tool never synthesises material. To start fresh, wipe, then add your own
footage through the Corpus panel (or drop files into library/ and decompose).
"""
from __future__ import annotations

import argparse

import pool


def main() -> None:
    ap = argparse.ArgumentParser(description="Distortion of Time — pool maintenance")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("list", help="list pooled videos")
    r = sub.add_parser("remove", help="remove one video from the pool")
    r.add_argument("video_id")
    r.add_argument("--source", action="store_true", help="also delete the library source file")
    w = sub.add_parser("wipe", help="reset the pool to empty")
    w.add_argument("--library", action="store_true", help="also delete source clips from library/")
    w.add_argument("--yes", action="store_true", help="confirm (required)")

    args = ap.parse_args()
    if args.cmd == "list":
        for v in pool.list_videos():
            print(f"  {v['video_id']}  {v['n_frames']:5d}f  {v['name']}")
        s = pool.stats()
        print(f"\n{s['clips']} clips · {s['frames']} frames")
    elif args.cmd == "remove":
        rec = pool.remove_video(args.video_id, delete_source=args.source)
        print(f"removed {rec.get('name', args.video_id)}"
              + (" (+ source)" if args.source else "") + f"  →  {pool.stats()}")
    elif args.cmd == "wipe":
        if not args.yes:
            extra = " AND delete library files" if args.library else ""
            print(f"This will wipe the pool{extra}. Re-run with --yes to confirm.")
            return
        print("wiped →", pool.wipe(delete_library=args.library))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
