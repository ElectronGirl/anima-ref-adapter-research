"""Pack a generation run into Parquet shards for a Hugging Face dataset repo, and unpack it back.

One row per image: the WebP (Hugging Face Image feature, so the dataset viewer shows it), the raw
VAE latent the image was decoded from, flattened prompt/scene/identity columns, and the full
results row as JSON so unpacking restores the run exactly.

  python deploy/pack_dataset.py pack /workspace/data/pilot --run-dir runs/pilot --out /workspace/hf_dataset --name pilot
  python deploy/pack_dataset.py unpack /path/to/repo/data/pilot --out /workspace/data/pilot

Pack writes OUT/data/NAME/train-XXXXX-of-YYYYY.parquet and copies the run's manifest files to
OUT/manifests/NAME/. Unpack writes the generator's layout (GROUP/RECORD.webp,
GROUP/RECORD.latent.safetensors, results_gpu0.jsonl) that train/prepare*.py read.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROWS_PER_SHARD = 800  # ~0.55 GB per shard (0.5 MB latent + ~0.18 MB WebP per row)
MANIFEST_FILES = ("manifest.jsonl", "groups.jsonl", "manifest_summary.json")
SCENE_KEYS = ("framing", "camera", "pose", "expression", "background", "lighting")

STRING_COLUMNS = ("record_id", "group_id", "character", "character_danbooru", "artists", "rating", "gender",
                  "copyright", *SCENE_KEYS, "prompt", "negative", "record_json")
SCHEMA = pa.schema(
    [(name, pa.string()) for name in STRING_COLUMNS[:8]]
    + [("appearance", pa.list_(pa.string())), ("outfit", pa.list_(pa.string()))]
    + [(name, pa.string()) for name in STRING_COLUMNS[8:]]
    + [("width", pa.int32()), ("height", pa.int32()), ("seed", pa.uint64()),
       ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
       ("latent", pa.binary()), ("latent_shape", pa.list_(pa.int32())), ("latent_dtype", pa.string())])


def hf_features() -> dict:
    """The Hugging Face `datasets` feature spec for SCHEMA (stored in the Parquet metadata)."""
    value = lambda dtype: {"dtype": dtype, "_type": "Value"}  # noqa: E731
    features = {}
    for field in SCHEMA:
        if field.name == "image":
            features["image"] = {"_type": "Image"}
        elif pa.types.is_list(field.type):
            inner = "string" if pa.types.is_string(field.type.value_type) else "int32"
            features[field.name] = {"feature": value(inner), "_type": "Sequence"}
        else:
            features[field.name] = value({pa.string(): "string", pa.int32(): "int32", pa.uint64(): "uint64",
                                          pa.binary(): "binary"}[field.type])
    return {"info": {"features": features}}


def read_results(data_dir: Path) -> list[dict]:
    rows = {}
    for path in sorted(data_dir.glob("results_gpu*.jsonl")):
        if path.name.endswith(".log.jsonl"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") == "complete" and row.get("latent"):
                rows[row["record_id"]] = row
    return [rows[k] for k in sorted(rows)]


def to_row(data_dir: Path, record: dict) -> dict:
    from safetensors import safe_open
    image_path = data_dir / record["image"].replace("\\", "/")
    with safe_open(str(data_dir / record["latent"].replace("\\", "/")), "np") as handle:
        latent = handle.get_tensor("latent")
    latent = latent[0] if latent.shape[0] == 1 else latent  # saved as a batch of one
    ident, scene = record["identity"], record["scene"]
    return {
        "record_id": record["record_id"], "group_id": record["group_id"], "character": record["character"],
        "character_danbooru": record.get("character_danbooru", ""), "artists": json.dumps(record["artists"]),
        "rating": record["rating"], "gender": ident["gender"], "copyright": ident.get("copyright") or "",
        "appearance": ident["appearance"], "outfit": ident["outfit"],
        **{key: scene.get(key, "") for key in SCENE_KEYS},
        "prompt": record["prompt"], "negative": record["negative"], "record_json": json.dumps(record, ensure_ascii=False),
        "width": record["width"], "height": record["height"], "seed": record["seed"],
        "image": {"bytes": image_path.read_bytes(), "path": image_path.name},
        "latent": np.ascontiguousarray(latent).tobytes(), "latent_shape": list(latent.shape),
        "latent_dtype": str(latent.dtype),
    }


def pack(args) -> int:
    data_dir = Path(args.data_dir)
    records = read_results(data_dir)
    out = Path(args.out) / "data" / args.name
    out.mkdir(parents=True, exist_ok=True)
    shards = (len(records) + ROWS_PER_SHARD - 1) // ROWS_PER_SHARD
    schema = SCHEMA.with_metadata({"huggingface": json.dumps(hf_features())})
    for shard in range(shards):
        path = out / f"train-{shard:05d}-of-{shards:05d}.parquet"
        chunk = records[shard * ROWS_PER_SHARD:(shard + 1) * ROWS_PER_SHARD]
        if path.exists() and pq.ParquetFile(path).metadata.num_rows == len(chunk):
            continue  # resumable: a finished shard with the right row count is kept
        with pq.ParquetWriter(path.with_suffix(".tmp"), schema, compression="zstd") as writer:
            for start in range(0, len(chunk), 100):
                writer.write_table(pa.Table.from_pylist([to_row(data_dir, r) for r in chunk[start:start + 100]],
                                                        schema=schema))
        path.with_suffix(".tmp").rename(path)
        print(f"{path.name}: {len(chunk)} rows, {path.stat().st_size / 2**20:.0f} MB", flush=True)
    for stale in out.glob("train-*.parquet"):  # shards of an earlier, smaller pack
        if not stale.name.endswith(f"-of-{shards:05d}.parquet"):
            stale.unlink()
    if args.run_dir:
        manifests = Path(args.out) / "manifests" / args.name
        manifests.mkdir(parents=True, exist_ok=True)
        for name in MANIFEST_FILES:
            if (Path(args.run_dir) / name).exists():
                shutil.copy(Path(args.run_dir) / name, manifests / name)
    print(f"packed {len(records)} records of {args.name} into {shards} shards", flush=True)
    return 0


def unpack(args) -> int:
    import torch
    from safetensors.torch import save_file
    out = Path(args.out)
    lines = []
    for path in sorted(Path(args.parquet_dir).glob("*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=100, columns=["record_json", "image", "latent",
                                                                                 "latent_shape", "latent_dtype"]):
            for row in batch.to_pylist():
                record = json.loads(row["record_json"])
                (out / record["group_id"]).mkdir(parents=True, exist_ok=True)
                (out / record["image"].replace("\\", "/")).write_bytes(row["image"]["bytes"])
                latent = np.frombuffer(row["latent"], dtype=row["latent_dtype"]).reshape(row["latent_shape"])
                save_file({"latent": torch.from_numpy(latent.copy())[None]}, str(out / record["latent"].replace("\\", "/")),
                          metadata={"space": "vae_raw", "record_id": record["record_id"]})
                lines.append(row["record_json"])
    (out / "results_gpu0.jsonl").write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    print(f"unpacked {len(lines)} records into {out}", flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pack")
    p.add_argument("data_dir")
    p.add_argument("--out", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--run-dir", help="runs/NAME with manifest.jsonl / groups.jsonl / manifest_summary.json")
    u = sub.add_parser("unpack")
    u.add_argument("parquet_dir")
    u.add_argument("--out", required=True)
    args = parser.parse_args()
    return pack(args) if args.command == "pack" else unpack(args)


if __name__ == "__main__":
    raise SystemExit(main())
