#!/usr/bin/env bash
# Progress, throughput, GPU and disk state of a run.
#   deploy/status.sh RUN_NAME
set -uo pipefail
RUN="${1:?run name}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/workspace/data}"
TOTAL=$(cat "$PROJECT_ROOT/runs/$RUN"/shard_*.jsonl | wc -l)
DONE=$(cat "$DATA_ROOT/$RUN"/results_gpu*.jsonl 2>/dev/null | grep -c '"status": "complete"')
echo "complete: $DONE / $TOTAL"
for log in "$DATA_ROOT/$RUN"/results_gpu*.log.jsonl; do
    [[ -f "$log" ]] && echo "$(basename "$log"): $(tail -n 1 "$log")"
done
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader
df -h "$DATA_ROOT" | tail -1
du -sh "$DATA_ROOT/$RUN" 2>/dev/null
