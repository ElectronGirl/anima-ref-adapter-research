"""Tag catalogs for data generation.

Artists (data/danbooru_artists_ge100.csv: Danbooru artist tags with 100+ posts) and characters
(category "character") both come from the public Danbooru wiki dump on Hugging Face; see
data/README.md.
"""
from __future__ import annotations

import csv
import json
import re
import urllib.request
from pathlib import Path
from typing import Iterable, Iterator

WIKI_DUMP_URL = (
    "https://huggingface.co/datasets/kierarkia/danbooru-wiki-2026/resolve/main/"
    "danbooru_wiki_dataset_2026-04-28.jsonl"
)


def anima_tag(danbooru_tag: str) -> str:
    """Danbooru tag -> Anima prompt spelling (spaces, escaped parentheses)."""
    text = re.sub(r"\s+", " ", danbooru_tag.replace("_", " ")).strip()
    text = text.replace("\\", "\\\\")
    return text.replace("(", "\\(").replace(")", "\\)")


def iter_jsonl(stream: Iterable[bytes | str]) -> Iterator[dict]:
    for raw in stream:
        line = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        line = line.strip()
        if line:
            yield json.loads(line)


def iter_character_records(records: Iterable[dict], min_post_count: int, category: str = "character") -> Iterator[dict]:
    """Wiki records of one tag category; artist tags are spelled "@name" in Anima prompts."""
    prefix = "@" if category == "artist" else ""
    for record in records:
        if str(record.get("category_name", "")).casefold() != category:
            continue
        if record.get("is_deleted", False):
            continue
        post_count = int(record.get("post_count") or 0)
        title = str(record.get("title") or "").strip()
        if post_count < min_post_count or not title:
            continue
        yield {
            "danbooru_tag": title,
            "anima_prompt_tag": prefix + anima_tag(title),
            "post_count": post_count,
        }


def fetch_character_catalog(path: Path, min_post_count: int = 500, url: str = WIKI_DUMP_URL,
                            category: str = "character") -> int:
    request = urllib.request.Request(url, headers={"User-Agent": "anima-ref-adapter-datagen/0.1"})
    with urllib.request.urlopen(request, timeout=120) as stream:
        rows = sorted(iter_character_records(iter_jsonl(stream), min_post_count, category),
                      key=lambda row: -row["post_count"])
    write_csv(path, rows, ["danbooru_tag", "anima_prompt_tag", "post_count"])
    return len(rows)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    tmp.replace(path)


def load_artists(path: Path, min_post_count: int) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["post_count"]) >= min_post_count]
    return [{"tag": row["anima_prompt_tag"], "post_count": int(row["post_count"])} for row in rows]


def load_characters(path: Path, min_post_count: int) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if int(row["post_count"]) >= min_post_count]
    return [{"tag": row["anima_prompt_tag"], "danbooru_tag": row["danbooru_tag"],
             "post_count": int(row["post_count"])} for row in rows]


def load_name_set(path: Path) -> set[str]:
    """One Danbooru tag per line; blank lines and '#' comments ignored."""
    if not path.exists():
        return set()
    names = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            names.add(line)
    return names
