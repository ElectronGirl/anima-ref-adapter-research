"""Drive one ComfyUI process (one GPU) through a manifest shard.

Pending records are grouped by resolution into fixed-size heterogeneous batches. Up to
--queue-depth batches are kept queued in ComfyUI so the GPU never waits for the driver.
Completion is read from the results file written by AnimaRefDatagenSave, so a restarted
worker resumes exactly the missing records.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from comfy_api import ComfyApiError, ComfyClient  # noqa: E402
from workflow import build_batch_workflow, build_flush_workflow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # torn last line after a crash; the record is simply regenerated
    return rows


def completed_ids(results: Path) -> set[str]:
    return {row["record_id"] for row in read_jsonl(results) if row.get("status") == "complete"}


def make_batches(records: list[dict], batch_size: int) -> list[list[dict]]:
    by_res = defaultdict(list)
    for record in records:
        by_res[(record["width"], record["height"])].append(record)
    batches = []
    for key in sorted(by_res, key=lambda k: -len(by_res[k])):
        items = by_res[key]
        batches.extend(items[i:i + batch_size] for i in range(0, len(items), batch_size))
    # Full batches first so the compiled shapes stay stable; remainders run at the end.
    batches.sort(key=lambda b: len(b) != batch_size)
    return batches


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "datagen.json")
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--comfy-url", default="http://127.0.0.1:8190")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--profile")
    parser.add_argument("--queue-depth", type=int, default=2)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--timeout", type=float, default=3600)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.batch_size:
        config["acceleration"]["batch_size"] = args.batch_size
    if args.profile:
        config["acceleration"]["profile"] = args.profile
    batch_size = int(config["acceleration"]["batch_size"])

    records = read_jsonl(args.shard)
    done = completed_ids(args.results)
    pending = [r for r in records if r["record_id"] not in done]
    batches = make_batches(pending, batch_size)
    if args.max_batches:
        batches = batches[: args.max_batches]
    print(f"shard={args.shard.name} records={len(records)} done={len(done)} pending={len(pending)} "
          f"batches={len(batches)} batch_size={batch_size} profile={config['acceleration']['profile']}", flush=True)
    if not batches:
        return 0

    client = ComfyClient(args.comfy_url)
    stats = client.system_stats()
    print("devices:", [d.get("name") for d in stats.get("devices", [])], flush=True)
    needed = {node["class_type"] for node in build_batch_workflow(batches[0], config, "", "").values()}
    missing = sorted(c for c in needed if not client.has_node(c))
    if missing:
        print(f"ComfyUI is missing nodes {missing}; check the custom node import errors in its log", flush=True)
        return 2
    args.out_dir.mkdir(parents=True, exist_ok=True)

    queued: deque = deque()
    started = time.perf_counter()
    finished_images = 0
    errors = 0
    next_index = 0
    log_path = args.results.with_suffix(".log.jsonl")

    def submit(batch: list[dict]) -> None:
        wf = build_batch_workflow(batch, config, str(args.out_dir.resolve()), str(args.results.resolve()))
        queued.append((client.queue_prompt(wf), batch, time.perf_counter()))

    while next_index < len(batches) or queued:
        while next_index < len(batches) and len(queued) < args.queue_depth:
            submit(batches[next_index])
            next_index += 1
        prompt_id, batch, t_submit = queued.popleft()
        t0 = time.perf_counter()
        try:
            entry = client.wait(prompt_id, timeout=args.timeout)
            ok = True
        except ComfyApiError as exc:
            ok, entry = False, {"error": str(exc)}
            errors += 1
            print(f"[error] batch of {len(batch)} ({batch[0]['record_id']}..): {str(exc)[:500]}", flush=True)
        now = time.perf_counter()
        if ok:
            finished_images += len(batch)
        elapsed = now - started
        rate = finished_images / elapsed if elapsed > 0 else 0.0
        remaining = sum(len(b) for b in batches[next_index:]) + sum(len(b) for _, b, _ in queued)
        line = {"t": time.time(), "prompt_id": prompt_id, "ok": ok, "n": len(batch),
                "res": f"{batch[0]['width']}x{batch[0]['height']}", "wait_s": round(now - t0, 2),
                "since_submit_s": round(now - t_submit, 2), "images_done": finished_images,
                "img_per_s": round(rate, 3), "eta_h": round(remaining / rate / 3600, 2) if rate else None}
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(line) + "\n")
        print(json.dumps(line), flush=True)
        if errors >= 3 and finished_images == 0:
            print("three failures before any success; stopping", flush=True)
            return 3

    client.wait(client.queue_prompt(build_flush_workflow(str(time.time()))), timeout=args.timeout)
    total = time.perf_counter() - started
    print(f"done: {finished_images} images in {total/3600:.2f} h ({finished_images/total:.3f} img/s), errors={errors}", flush=True)
    return 0 if errors == 0 else 4


if __name__ == "__main__":
    raise SystemExit(main())
