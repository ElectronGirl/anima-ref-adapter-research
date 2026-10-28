# Tag catalogs

Not committed (they are derived from third-party dumps); rebuild them into this folder:

| File | Built by | Source |
|---|---|---|
| `danbooru_characters.csv` | `python datagen/fetch_character_catalog.py` | Danbooru wiki dump [`kierarkia/danbooru-wiki-2026`](https://huggingface.co/datasets/kierarkia/danbooru-wiki-2026), category character |
| `danbooru_artists_ge100.csv` | `python datagen/fetch_character_catalog.py --category artist --min-post-count 100 --output data/danbooru_artists_ge100.csv` | same dump, category artist (tags written as `@name`) |
| `danbooru_tag_taxonomy.parquet` | download | [`Mexes/danbooru-tag-taxonomy`](https://huggingface.co/datasets/Mexes/danbooru-tag-taxonomy) (CC BY 4.0) |
| `character_tags.jsonl` | `python datagen/build_character_tags.py` | per-character appearance/outfit tag frequencies over [`trojblue/danbooru2025-metadata`](https://huggingface.co/datasets/trojblue/danbooru2025-metadata) (MIT) |

CSV columns: `danbooru_tag, anima_prompt_tag, post_count`. A newer wiki dump gives slightly different
catalogs than the ones used for the runs in `docs/experiments.md`.
