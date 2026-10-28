"""Minimal ComfyUI HTTP client."""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
import uuid


class ComfyApiError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, base_url: str, timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.client_id = str(uuid.uuid4())

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ComfyApiError(f"GET {path} failed: {exc}") from exc

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(self.base_url + path, data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as exc:
            raise ComfyApiError(f"POST {path} HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise ComfyApiError(f"POST {path} failed: {exc}") from exc

    def system_stats(self) -> dict:
        return self._get("/system_stats")

    def has_node(self, class_type: str) -> bool:
        return class_type in self._get(f"/object_info/{class_type}")

    def queue_prompt(self, workflow: dict) -> str:
        response = self._post("/prompt", {"prompt": workflow, "client_id": self.client_id})
        if "prompt_id" not in response:
            raise ComfyApiError(f"no prompt_id in response: {response}")
        return str(response["prompt_id"])

    def history(self, prompt_id: str) -> dict:
        return self._get(f"/history/{prompt_id}").get(prompt_id, {})

    def wait(self, prompt_id: str, timeout: float, poll: float = 0.5) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            entry = self.history(prompt_id)
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                raise ComfyApiError(f"prompt {prompt_id} failed: {json.dumps(status)[:2000]}")
            if status.get("completed"):
                return entry
            time.sleep(poll)
        raise ComfyApiError(f"prompt {prompt_id} timed out after {timeout:.0f}s")
