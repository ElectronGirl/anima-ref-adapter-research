"""Local dashboard for training runs on the Vast instance: metric charts, eval loss gaps and eval images.

A background thread copies each run's train_log.jsonl and finished eval folders (samples/stepNNNNN)
from the instance over SSH into a local cache; the page reads them from /api/runs and /files/.

  python dashboard/server.py --host my-instance --run runs/train/char_v2 --run runs/train/dual_r32
  -> http://127.0.0.1:8770

The SSH config (default .vast/ssh_config, git-ignored) needs a Host entry for --host.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GIT_SSH = Path("C:/Program Files/Git/usr/bin")
IMAGE_SUFFIXES = (".jpg", ".png")


class Sync:
    def __init__(self, args):
        self.args, self.runs = args, args.run
        self.cache = Path(args.cache)
        bindir = GIT_SSH if (GIT_SSH / "ssh.exe").exists() else None
        self.ssh = [str(bindir / "ssh.exe") if bindir else "ssh", "-F", str(ROOT / args.ssh_config), args.host]
        self.scp = [str(bindir / "scp.exe") if bindir else "scp", "-q", "-F", str(ROOT / args.ssh_config)]
        self.status = {"last_sync": None, "error": None, "remote": {}}

    @staticmethod
    def check(result: subprocess.CompletedProcess, what: str) -> None:
        if result.returncode != 0:
            # The instance prints a login banner on stderr; keep only the real error lines.
            lines = [l for l in result.stderr.splitlines() if l.strip() and "vast.ai" not in l and "Have fun" not in l]
            raise RuntimeError(" ".join(lines)[-300:] or f"{what} exit {result.returncode}")

    def remote(self, command: str) -> str:
        result = subprocess.run(self.ssh + [command], capture_output=True, text=True, timeout=120)
        self.check(result, "ssh")
        return result.stdout

    def fetch(self, remote_paths: list[str], local_dir: Path) -> None:
        local_dir.mkdir(parents=True, exist_ok=True)
        sources = [f"{self.args.host}:{self.args.remote_root}/{p}" for p in remote_paths]
        result = subprocess.run(self.scp + sources + [str(local_dir)], capture_output=True, text=True, timeout=600)
        self.check(result, "scp")

    def once(self) -> None:
        for run in self.runs:
            local = self.cache / Path(run).name
            # One listing per run: log/config presence, eval folders and whether each is finished,
            # and whether the training process is alive.
            listing = self.remote(
                f"cd {self.args.remote_root}/{run} 2>/dev/null || exit 0; ls train_log.jsonl config.json 2>/dev/null; "
                f"for d in samples/step*; do [ -d \"$d\" ] && echo \"$d $(ls $d | tr '\\n' ',')\"; done; "
                f"if [ -f train.pid ]; then kill -0 $(cat train.pid) 2>/dev/null && echo ALIVE || echo DEAD; fi; true")
            lines = [l for l in listing.splitlines() if l.strip()]
            files = [l for l in lines if l in ("train_log.jsonl", "config.json")]
            if files:
                self.fetch([f"{run}/{f}" for f in files], local)
            self.status["remote"][run] = "running" if "ALIVE" in lines else ("stopped" if "DEAD" in lines else "unknown")
            for line in lines:
                if not line.startswith("samples/"):
                    continue
                step_dir, _, names = line.partition(" ")
                names = [n for n in names.split(",") if n]
                target = local / step_dir
                marker = target / ".complete"
                if marker.exists() or not names:
                    continue
                # The contact sheet is written last; until then copy new images and refresh the JSON.
                wanted = [n for n in names if n.endswith(".json")
                          or (n.endswith(IMAGE_SUFFIXES) and not (target / n).exists())]
                if wanted:
                    self.fetch([f"{run}/{step_dir}/{n}" for n in wanted], target)
                if "contact_sheet.jpg" in names:
                    marker.write_text("ok")

    def loop(self) -> None:
        while True:
            try:
                self.once()
                self.status.update(last_sync=time.time(), error=None)
            except Exception as error:  # keep serving the cache when the instance is unreachable
                self.status["error"] = str(error)
            time.sleep(self.args.interval)


def run_data(cache: Path, runs: list[str], status: dict) -> dict:
    out = {"runs": [], "status": {k: v for k, v in status.items() if k != "remote"}}
    for run in runs:
        name = Path(run).name
        local = cache / name
        log, evals = [], []
        log_path = local / "train_log.jsonl"
        if log_path.exists():
            for line in log_path.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                (evals if "eval_step" in row else log).append(row)
        samples = {}
        for step_dir in sorted((local / "samples").glob("step*")):
            gaps_path = step_dir / "loss_gaps.json"
            images = sorted(p.name for p in step_dir.iterdir() if p.suffix in IMAGE_SUFFIXES and p.name != "contact_sheet.jpg")
            samples[step_dir.name] = {"images": images,
                                      "gaps": json.loads(gaps_path.read_text(encoding="utf-8")) if gaps_path.exists() else None}
        config = json.loads((local / "config.json").read_text(encoding="utf-8")) if (local / "config.json").exists() else None
        out["runs"].append({"name": name, "path": run, "state": status["remote"].get(run, "unknown"),
                            "log": log, "evals": evals, "samples": samples,
                            "steps": (config or {}).get("train", {}).get("steps")})
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True, help="run dir relative to the remote project")
    parser.add_argument("--host", default="anima-ref", help="Host entry in the SSH config")
    parser.add_argument("--ssh-config", default=".vast/ssh_config", help="relative to the repo root")
    parser.add_argument("--remote-root", default="/workspace/anima-ref-adapter-research", help="repo checkout on the instance")
    parser.add_argument("--cache", default=str(ROOT / "runs" / "dashboard_cache"))
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    sync = Sync(args)
    threading.Thread(target=sync.loop, daemon=True).start()
    cache = Path(args.cache)

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in ("/", "/index.html"):
                return self.send_file(HERE / "index.html", "text/html; charset=utf-8")
            if path == "/api/runs":
                body = json.dumps(run_data(cache, args.run, sync.status)).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                return self.wfile.write(body)
            if path.startswith("/files/"):
                target = (cache / path[len("/files/"):]).resolve()
                if cache.resolve() in target.parents and target.is_file():
                    kind = "image/jpeg" if target.suffix == ".jpg" else "image/png" if target.suffix == ".png" else "application/json"
                    return self.send_file(target, kind)
            self.send_error(404)

        def send_file(self, file: Path, kind: str):
            data = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    print(f"dashboard on http://127.0.0.1:{args.port} (runs: {', '.join(args.run)})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
