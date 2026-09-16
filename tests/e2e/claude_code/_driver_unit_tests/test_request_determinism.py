"""The CLI must send the same request bytes from one build to the next.

Markerless harness test: it drives the real `claude` binary against a local
stub instead of a proxy, so it carries no `e2e` marker. The binary is a
prerequisite of this whole suite, so a missing one is a failure rather than a
skip.

Two builds differ in ways the driver does not control: a fresh pod, so no CLI
state survives, and a different candidate checked out at a different commit.
Both used to reach the request body, through the memory path the system prompt
names and through the git block the CLI adds for its working directory, so the
shared provider cache missed on every Claude Code cell. This replays those two
differences across a pair of invocations and holds the bytes equal.

A pinned session id is what makes the second test necessary. The matrix runs
its cells across xdist workers, and the CLI refuses to start a session id that
another live process already holds, so pinning one without also opting out of
session persistence turns most of a parallel run red.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Tuple

import pytest

from claude_code.cli_driver import _stable_cli_state, run_claude
from claude_code.rate_limiter import RateLimiter

_STUB_REPLY = {
    "id": "msg_stub",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "ok"}],
    "stop_reason": "end_turn",
    "usage": {"input_tokens": 10, "output_tokens": 2},
}


def _make_repo(root: Path, subject: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    identity = {"NAME": "t", "EMAIL": "t@e2e"}
    env = dict(
        os.environ,
        **{f"GIT_{role}_{key}": value for role in ("AUTHOR", "COMMITTER") for key, value in identity.items()},
    )
    (root / "file.txt").write_text(subject, encoding="utf-8")
    for args in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", subject]):
        subprocess.run(["git", *args], cwd=root, env=env, check=True, capture_output=True)
    return root


@pytest.fixture(name="captured")
def _captured() -> Tuple[str, List[bytes]]:
    bodies: List[bytes] = []
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("content-length") or 0))
            if "count_tokens" not in self.path:
                with lock:
                    bodies.append(raw)
            payload = json.dumps({"input_tokens": 10} if "count_tokens" in self.path else _STUB_REPLY).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", bodies
    finally:
        server.shutdown()


def test_two_builds_send_the_same_request_bytes(captured: Tuple[str, List[bytes]], tmp_path: Path) -> None:
    base_url, bodies = captured
    limiter = RateLimiter(state_dir=tmp_path / "limiter")
    checkouts = (_make_repo(tmp_path / "build-1", "first"), _make_repo(tmp_path / "build-2", "second"))
    origin = Path.cwd()

    sent = []
    for checkout in checkouts:
        shutil.rmtree(Path(_stable_cli_state()[0]).parent, ignore_errors=True)
        os.chdir(checkout)
        try:
            before = len(bodies)
            run_claude(
                prompt="say ok",
                model="claude-haiku-4-5",
                base_url=base_url,
                api_key="stub",
                extra_env={"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
                rate_limiter=limiter,
            )
            sent.append(bodies[before:])
        finally:
            os.chdir(origin)

    assert sent[0], "the CLI sent no request to the stub, so there is nothing to compare"
    assert sent[0] == sent[1]


def test_concurrent_cells_do_not_collide_on_the_pinned_session(
    captured: Tuple[str, List[bytes]], tmp_path: Path
) -> None:
    base_url, bodies = captured
    limiter = RateLimiter(state_dir=tmp_path / "limiter")

    def one(_index: int) -> int:
        return run_claude(
            prompt="say ok",
            model="claude-haiku-4-5",
            base_url=base_url,
            api_key="stub",
            extra_env={"CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
            rate_limiter=limiter,
        ).exit_code

    with ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(one, range(4)))

    assert codes == [0, 0, 0, 0]
    assert bodies, "the CLI sent no request to the stub, so there is nothing to compare"
    assert set(Counter(bodies).values()) == {4}
