"""Build data/danbooru_characters.csv (or, with --category artist, data/danbooru_artists_ge100.csv)
from the public Danbooru wiki dump.

  python datagen/fetch_character_catalog.py
  python datagen/fetch_character_catalog.py --category artist --min-post-count 100 --output data/danbooru_artists_ge100.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import WIKI_DUMP_URL, fetch_character_catalog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-post-count", type=int, default=500)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "danbooru_characters.csv")
    parser.add_argument("--url", default=WIKI_DUMP_URL)
    parser.add_argument("--category", default="character", choices=["character", "artist"])
    args = parser.parse_args()
    count = fetch_character_catalog(args.output, args.min_post_count, args.url, args.category)
    print(f"wrote {count:,} {args.category} tags to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
