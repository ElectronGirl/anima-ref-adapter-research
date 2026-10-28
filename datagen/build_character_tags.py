"""Per-character identity tags (appearance + outfit) from real Danbooru post statistics.

For every character in the catalog, single-character posts are scanned and each general tag's
frequency is measured. Tags are kept by Danbooru-taxonomy slot (Mexes/danbooru-tag-taxonomy):
hair, eyes, a whitelist of identity anatomy (ears, tail, horns, wings, ...), species
demographics, and clothing slots except underwear. Tags are stored with their exact Danbooru
names; the manifest converts them to Anima spelling.

Source posts: trojblue/danbooru2025-metadata (MIT, 9.1M posts), read column-projected per file.

Output data/character_tags.jsonl:
  {"character": danbooru_tag, "posts": n, "gender": "1girl"|"1boy"|null,
   "copyright": tag|null, "appearance": [[tag, freq], ...], "outfit": [[tag, freq], ...]}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from catalog import load_characters  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POSTS_REPO = "hf://datasets/trojblue/danbooru2025-metadata/data"
N_FILES = 43

APPEARANCE_SLOTS = {"appearance:hair", "appearance:eyes"}
OUTFIT_SLOTS = {"clothing:outfit", "clothing:top", "clothing:bottom", "clothing:headwear",
                "clothing:accents", "clothing:legwear", "clothing:footwear", "clothing:eyewear"}
ANATOMY_KEYWORDS = ("ears", "tail", "horn", "wings", "halo", "fang", "skin", "mole", "scar",
                    "freckles", "facial_mark", "pointy", "antenna", "third_eye", "heterochromia")
EXCLUDE = {
    "closed_eyes", "one_eye_closed", "half-closed_eyes", "eyelashes", "full_body", "upper_body",
    "alternate_costume", "official_alternate_costume", "alternate_hairstyle", "alternate_hair_length",
    "alternate_hair_color", "cosplay", "swimsuit", "bikini", "nude", "naked_shirt", "navel", "midriff",
    "cleavage", "cleavage_cutout", "underboob", "sideboob", "no_bra", "no_panties", "panties",
    "pantyshot", "bare_legs", "barefoot", "topless", "bottomless", "covered_nipples", "see-through",
    "wet", "wet_clothes", "torn_clothes", "open_clothes", "clothes_lift", "skirt_lift", "shirt_lift",
    "hair_down", "hair_up", "messy_hair", "wet_hair", "floating_hair", "hair_flowing_over",
}
SPECIES_SUFFIXES = ("_girl", "_boy")
NON_SPECIES = {"magical_girl", "shrine_maiden", "school_girl", "gyaru", "multiple_girls", "multiple_boys"}


def tag_slots() -> dict[str, str]:
    tax = pl.read_parquet(ROOT / "data" / "danbooru_tag_taxonomy.parquet",
                          columns=["tag", "danbooru_category", "slot"])
    tax = tax.filter(pl.col("danbooru_category") == "general")
    return dict(zip(tax["tag"].to_list(), tax["slot"].to_list()))


def classify(tag: str, slot: str | None) -> str | None:
    if tag in EXCLUDE or slot is None:
        return None
    if slot in APPEARANCE_SLOTS:
        return "appearance"
    if slot == "appearance:anatomy" and any(k in tag for k in ANATOMY_KEYWORDS):
        return "appearance"
    if slot == "appearance:demographics" and tag.endswith(SPECIES_SUFFIXES) and tag not in NON_SPECIES \
            and not tag[0].isdigit():
        return "appearance"
    if slot in OUTFIT_SLOTS:
        return "outfit"
    return None


def scan_file(index: int, characters: set[str]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    path = f"{POSTS_REPO}/train-{index:05d}-of-{N_FILES:05d}.parquet"
    posts = (
        pl.scan_parquet(path)
        .select("tag_string_character", "tag_string_general", "tag_string_copyright", "tag_count_character", "is_deleted")
        .filter((pl.col("tag_count_character") == 1) & ~pl.col("is_deleted").fill_null(False))
        .rename({"tag_string_character": "character"})
        .filter(pl.col("character").is_in(list(characters)))
        .collect()
    )
    n_posts = posts.group_by("character").agg(pl.len().alias("posts"))
    general = (posts.select("character", pl.col("tag_string_general").str.split(" ").alias("tag"))
               .explode("tag").group_by("character", "tag").agg(pl.len().alias("count")))
    copyright = (posts.select("character", pl.col("tag_string_copyright").str.split(" ").alias("tag"))
                 .explode("tag").filter(pl.col("tag") != "")
                 .group_by("character", "tag").agg(pl.len().alias("count")))
    return n_posts, general, copyright


def merge(acc: pl.DataFrame | None, new: pl.DataFrame, keys: list[str], value: str) -> pl.DataFrame:
    if acc is None:
        return new
    return pl.concat([acc, new]).group_by(keys).agg(pl.col(value).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-post-count", type=int, default=500)
    parser.add_argument("--files", type=int, default=N_FILES, help="scan only the first N files (testing)")
    parser.add_argument("--appearance-min", type=float, default=0.40)
    parser.add_argument("--outfit-min", type=float, default=0.30)
    parser.add_argument("--max-appearance", type=int, default=8)
    parser.add_argument("--max-outfit", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "character_tags.jsonl")
    args = parser.parse_args()

    chars = {c["danbooru_tag"] for c in load_characters(ROOT / "data" / "danbooru_characters.csv", args.min_post_count)}
    slots = tag_slots()
    acc_posts = acc_general = acc_copy = None
    for i in range(args.files):
        t0 = time.perf_counter()
        n_posts, general, copyright = scan_file(i, chars)
        acc_posts = merge(acc_posts, n_posts, ["character"], "posts")
        acc_general = merge(acc_general, general, ["character", "tag"], "count")
        acc_copy = merge(acc_copy, copyright, ["character", "tag"], "count")
        print(f"file {i + 1}/{args.files}: {n_posts['posts'].sum()} posts, {time.perf_counter() - t0:.1f}s", flush=True)

    posts = dict(acc_posts.iter_rows())
    freq = acc_general.join(acc_posts, on="character").with_columns((pl.col("count") / pl.col("posts")).alias("freq"))
    best_copy = (acc_copy.sort("count", descending=True).group_by("character").first()
                 .join(acc_posts, on="character").filter(pl.col("count") / pl.col("posts") >= 0.5))
    copy_of = dict(zip(best_copy["character"].to_list(), best_copy["tag"].to_list()))

    rows = []
    for (character,), group in freq.sort("freq", descending=True).group_by(["character"], maintain_order=True):
        tags = dict(zip(group["tag"].to_list(), group["freq"].to_list()))
        girl, boy = tags.get("1girl", 0.0), tags.get("1boy", 0.0)
        gender = "1girl" if girl >= 0.6 else "1boy" if boy >= 0.6 else None
        appearance, outfit = [], []
        for tag, f in tags.items():
            kind = classify(tag, slots.get(tag))
            if kind == "appearance" and f >= args.appearance_min and len(appearance) < args.max_appearance:
                appearance.append([tag, round(f, 3)])
            elif kind == "outfit" and f >= args.outfit_min and len(outfit) < args.max_outfit:
                outfit.append([tag, round(f, 3)])
        rows.append({"character": character, "posts": posts[character], "gender": gender,
                     "copyright": copy_of.get(character), "appearance": appearance, "outfit": outfit})
    rows.sort(key=lambda r: -r["posts"])
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {len(rows)} characters to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
