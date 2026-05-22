#!/usr/bin/env python3
"""Produce before/after local proof for Bedrock parallel tool_choice requests."""
# ruff: noqa: T201

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL_ALIAS = "bedrock-proof-claude"
BEDROCK_MODEL = "bedrock/converse/us.anthropic.claude-sonnet-4-5-20250929-v1:0"
MASTER_KEY = "sk-local-proof"


class CapturingBedrockServer(ThreadingHTTPServer):
    """Tiny Bedrock Converse mock that records and validates request bodies."""

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, CapturingBedrockHandler)
        self.requests: list[dict[str, Any]] = []


class CapturingBedrockHandler(BaseHTTPRequestHandler):
    server: CapturingBedrockServer

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/health":
            self._write_json(200, {"status": "ok"})
            return
        self._write_json(404, {"error": f"unhandled path: {self.path}"})

    def do_POST(self) -> None:
        raw_body = self.rfile.read(int(self.headers.get("content-length", "0")))
        body = json.loads(raw_body.decode("utf-8"))
        validation = _validate_bedrock_converse_body(body)
        self.server.requests.append(
            {
                "path": self.path,
                "body": body,
                "validation": validation,
            }
        )
        if validation["status"] != "accepted":
            self._write_json(400, validation)
            return
        self._write_json(
            200,
            {
                "output": {
                    "message": {
                        "role": "assistant",
                        "content": [{"text": "mock: valid Bedrock request shape"}],
                    }
                },
                "stopReason": "end_turn",
                "usage": {
                    "inputTokens": 1,
                    "outputTokens": 2,
                    "totalTokens": 3,
                },
            },
        )

    def _write_json(self, status_code: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status_code)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def _validate_bedrock_converse_body(body: dict[str, Any]) -> dict[str, Any]:
    additional = body.get("additionalModelRequestFields") or {}
    extension_tool_choice = additional.get("tool_choice") or {}
    native_tool_choice = (body.get("toolConfig") or {}).get("toolChoice")
    reasons: list[str] = []

    if extension_tool_choice.get("disable_parallel_tool_use") is not False:
        reasons.append(
            "additionalModelRequestFields.tool_choice.disable_parallel_tool_use "
            "is not false"
        )
    if "type" not in extension_tool_choice:
        reasons.append("additionalModelRequestFields.tool_choice.type is missing")
    if native_tool_choice is not None:
        reasons.append("toolConfig.toolChoice should not be sent in this request")

    return {
        "status": "accepted" if not reasons else "rejected",
        "reasons": reasons,
        "extension_tool_choice": extension_tool_choice,
        "native_tool_choice_present": native_tool_choice is not None,
    }


@contextlib.contextmanager
def bedrock_mock_server() -> Iterator[tuple[str, CapturingBedrockServer]]:
    server = CapturingBedrockServer(("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_proxy_config(path: Path, bedrock_base_url: str) -> None:
    path.write_text(
        f"""
model_list:
  - model_name: {MODEL_ALIAS}
    litellm_params:
      model: {BEDROCK_MODEL}
      api_base: {bedrock_base_url}
      aws_region_name: us-east-1
      aws_access_key_id: local-proof
      aws_secret_access_key: local-proof

general_settings:
  master_key: {MASTER_KEY}

litellm_settings:
  drop_params: true
  modify_params: true
""".lstrip(),
        encoding="utf-8",
    )


def start_proxy(
    *,
    repo_dir: Path,
    python_executable: str,
    config_path: Path,
    port: int,
    log_path: Path,
) -> subprocess.Popen[bytes]:
    env = {
        **os.environ,
        "LITELLM_TELEMETRY": "False",
        "PYTHONUNBUFFERED": "1",
        "DISABLE_ADMIN_UI": "True",
    }
    with log_path.open("wb") as log_file:
        return subprocess.Popen(
            [
                python_executable,
                "-c",
                "from litellm import run_server; run_server()",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--config",
                str(config_path),
            ],
            cwd=repo_dir,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
    except Exception:
        with contextlib.suppress(Exception):
            os.killpg(process.pid, signal.SIGKILL)


def read_log_tail(log_path: Path, max_chars: int = 4000) -> str:
    if not log_path.exists():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    return content[-max_chars:]


def wait_for_proxy(
    base_url: str, process: subprocess.Popen[bytes], log_path: Path
) -> None:
    deadline = time.monotonic() + 45
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"proxy exited early with code {process.returncode}\n"
                f"--- proxy log tail ---\n{read_log_tail(log_path)}"
            )
        try:
            request = Request(
                f"{base_url}/health",
                headers={"authorization": f"Bearer {MASTER_KEY}"},
                method="GET",
            )
            with urlopen(request, timeout=2) as response:
                if response.status < 500:
                    return
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = str(exc)
        time.sleep(0.5)
    raise TimeoutError(
        f"timed out waiting for proxy at {base_url}: {last_error}\n"
        f"--- proxy log tail ---\n{read_log_tail(log_path)}"
    )


def call_proxy(base_url: str) -> dict[str, Any]:
    payload = {
        "model": MODEL_ALIAS,
        "messages": [
            {
                "role": "user",
                "content": "Call get_weather for Seattle.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Return the weather for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}},
        "parallel_tool_calls": True,
        "max_tokens": 32,
    }
    request = Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "authorization": f"Bearer {MASTER_KEY}",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=30) as response:
            return {
                "status_code": response.status,
                "body": json.loads(response.read().decode("utf-8")),
            }
    except HTTPError as exc:
        return {
            "status_code": exc.code,
            "body": json.loads(exc.read().decode("utf-8")),
        }


def git(*args: str, cwd: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


@contextlib.contextmanager
def detached_worktree(repo_dir: Path, ref: str, parent_dir: Path) -> Iterator[Path]:
    worktree_dir = parent_dir / f"litellm-proof-{ref.replace('/', '-')}"
    subprocess.check_call(
        ["git", "worktree", "add", "--detach", str(worktree_dir), ref], cwd=repo_dir
    )
    try:
        yield worktree_dir
    finally:
        subprocess.check_call(
            ["git", "worktree", "remove", "--force", str(worktree_dir)], cwd=repo_dir
        )


def run_case(
    *,
    label: str,
    repo_dir: Path,
    python_executable: str,
    bedrock_base_url: str,
    work_dir: Path,
    bedrock_server: CapturingBedrockServer,
) -> dict[str, Any]:
    config_path = work_dir / f"{label}.yaml"
    log_path = work_dir / f"{label}.log"
    write_proxy_config(config_path, bedrock_base_url)
    port = find_free_port()
    request_start = len(bedrock_server.requests)
    process = start_proxy(
        repo_dir=repo_dir,
        python_executable=python_executable,
        config_path=config_path,
        port=port,
        log_path=log_path,
    )
    try:
        base_url = f"http://127.0.0.1:{port}"
        wait_for_proxy(base_url, process, log_path)
        proxy_response = call_proxy(base_url)
    finally:
        stop_process(process)

    new_requests = bedrock_server.requests[request_start:]
    bedrock_request = new_requests[-1] if new_requests else None
    return {
        "label": label,
        "repo_dir": str(repo_dir),
        "commit": git("rev-parse", "--short", "HEAD", cwd=repo_dir),
        "proxy_status_code": proxy_response["status_code"],
        "proxy_body": proxy_response["body"],
        "bedrock_request": bedrock_request,
        "proxy_log": str(log_path),
    }


def summarize_case(case: dict[str, Any]) -> list[str]:
    bedrock_request = case.get("bedrock_request") or {}
    validation = bedrock_request.get("validation") or {}
    body = bedrock_request.get("body") or {}
    additional = body.get("additionalModelRequestFields") or {}
    tool_config = body.get("toolConfig") or {}
    lines = [
        f"{case['label'].upper()} {case['commit']}",
        f"  proxy_status={case['proxy_status_code']}",
        f"  mock_bedrock_validation={validation.get('status', 'not-called')}",
        "  additionalModelRequestFields.tool_choice="
        f"{json.dumps(additional.get('tool_choice'), sort_keys=True)}",
        f"  toolConfig.toolChoice_present={'toolChoice' in tool_config}",
    ]
    reasons = validation.get("reasons") or []
    if reasons:
        lines.append(f"  rejection_reasons={json.dumps(reasons, sort_keys=True)}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Start a local Bedrock mock and LiteLLM proxy to show the "
            "parallel_tool_calls + explicit tool_choice request shape before "
            "and after this patch."
        )
    )
    parser.add_argument("--before-ref", default="origin/litellm_oss_staging")
    parser.add_argument("--after-ref", default="HEAD")
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()

    repo_dir = Path(git("rev-parse", "--show-toplevel", cwd=Path.cwd()))
    python_executable = sys.executable
    with tempfile.TemporaryDirectory(prefix="litellm-bedrock-proof-") as temp_name:
        temp_dir = Path(temp_name)
        with bedrock_mock_server() as (bedrock_base_url, bedrock_server):
            with detached_worktree(repo_dir, args.before_ref, temp_dir) as before_dir:
                before = run_case(
                    label="before",
                    repo_dir=before_dir,
                    python_executable=python_executable,
                    bedrock_base_url=bedrock_base_url,
                    work_dir=temp_dir,
                    bedrock_server=bedrock_server,
                )
            with detached_worktree(repo_dir, args.after_ref, temp_dir) as after_dir:
                after = run_case(
                    label="after",
                    repo_dir=after_dir,
                    python_executable=python_executable,
                    bedrock_base_url=bedrock_base_url,
                    work_dir=temp_dir,
                    bedrock_server=bedrock_server,
                )

        report = {
            "before_ref": args.before_ref,
            "after_ref": args.after_ref,
            "before": before,
            "after": after,
        }
        if args.json_output is not None:
            args.json_output.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print("LiteLLM Bedrock parallel tool_choice local proof")
        print(f"before_ref={args.before_ref}")
        print(f"after_ref={args.after_ref}")
        print()
        print("\n".join(summarize_case(before)))
        print()
        print("\n".join(summarize_case(after)))
        print()
        if (
            (before.get("bedrock_request") or {}).get("validation", {}).get("status")
            == "rejected"
            and (after.get("bedrock_request") or {}).get("validation", {}).get("status")
            == "accepted"
            and after["proxy_status_code"] == 200
        ):
            print(
                "RESULT: PASS - before is rejected by the Bedrock-shaped mock; after succeeds."
            )
        else:
            print("RESULT: FAIL - before/after did not show the expected transition.")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
