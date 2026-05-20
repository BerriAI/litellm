"""Helpers for running the vendored mock OpenAI / Anthropic / Vertex endpoint.

The mock server itself lives in ``tests/mock_endpoints/example_openai_endpoint/``
and is a vendored copy of https://github.com/BerriAI/example_openai_endpoint.

Many tests in this repo currently point at the Railway-hosted version of that
server (``https://exampleopenaiendpoint-production.up.railway.app``). Railway
outages take those tests down with it. This module exposes:

* :data:`MOCK_OPENAI_BASE_URL` — the URL tests should hit. It reads
  ``LITELLM_MOCK_OPENAI_BASE_URL`` from the environment, falling back to the
  public Railway URL for backwards-compatibility with tests that have not been
  migrated yet.
* :func:`start_mock_server` — spawn the vendored server as a subprocess on a
  free port. Used by the pytest fixture in ``conftest.py`` but also callable
  directly from scripts.

The pytest fixture ``mock_openai_endpoint_server`` (see ``conftest.py``) is
the recommended way for a test suite to opt in: it starts the server once per
session, exposes the URL, and tears the process down on exit.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import urllib.error
import urllib.request

DEFAULT_REMOTE_URL = "https://exampleopenaiendpoint-production.up.railway.app"

_SERVER_DIR = Path(__file__).resolve().parent / "example_openai_endpoint"
_SERVER_MAIN = _SERVER_DIR / "main.py"


def _resolve_base_url() -> str:
    url = os.environ.get("LITELLM_MOCK_OPENAI_BASE_URL")
    if url:
        return url.rstrip("/")
    return DEFAULT_REMOTE_URL


MOCK_OPENAI_BASE_URL = _resolve_base_url()


def _pick_free_port() -> int:
    """Bind to port 0 and immediately release, returning the kernel-assigned port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_ready(url: str, timeout: float = 30.0) -> None:
    """Poll the mock until ``/chat/completions`` returns a 2xx response."""
    deadline = time.monotonic() + timeout
    last_err: Optional[BaseException] = None
    payload = b'{"model":"gpt-4o-mini","messages":[{"role":"user","content":"ping"}]}'
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(
                f"{url}/chat/completions",
                data=payload,
                headers={
                    "Authorization": "Bearer sk-test",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                if 200 <= resp.status < 300:
                    return
        except (urllib.error.URLError, ConnectionError, OSError) as err:
            last_err = err
        time.sleep(0.2)
    raise RuntimeError(
        f"Mock server at {url} did not become ready within {timeout}s "
        f"(last error: {last_err!r})"
    )


class MockServerHandle:
    """Tiny RAII-style handle for a running mock server subprocess."""

    def __init__(self, process: subprocess.Popen, base_url: str) -> None:
        self.process = process
        self.base_url = base_url

    def stop(self) -> None:
        if self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)


def start_mock_server(
    port: Optional[int] = None,
    *,
    wait: bool = True,
    timeout: float = 30.0,
    log_file: Optional[Path] = None,
    python_executable: Optional[str] = None,
) -> MockServerHandle:
    """Start the vendored mock server as a child process.

    Args:
        port: Port to bind. ``None`` (default) picks a free port automatically.
        wait: If ``True``, block until ``/chat/completions`` returns a 2xx response.
        timeout: Maximum seconds to wait for readiness.
        log_file: Optional file to capture stdout/stderr. ``None`` inherits the parent's streams.
        python_executable: Python interpreter to launch the server with. Defaults to ``sys.executable``.
    """
    if port is None:
        port = _pick_free_port()

    env = {**os.environ, "PORT": str(port)}
    stdout = open(log_file, "w") if log_file is not None else None
    stderr = subprocess.STDOUT if stdout is not None else None

    process = subprocess.Popen(
        [python_executable or sys.executable, str(_SERVER_MAIN)],
        env=env,
        stdout=stdout,
        stderr=stderr,
        cwd=str(_SERVER_DIR),
    )
    base_url = f"http://127.0.0.1:{port}"
    handle = MockServerHandle(process=process, base_url=base_url)

    if wait:
        try:
            _wait_for_ready(base_url, timeout=timeout)
        except Exception:
            handle.stop()
            if stdout is not None:
                stdout.close()
            raise
    return handle


__all__ = [
    "DEFAULT_REMOTE_URL",
    "MOCK_OPENAI_BASE_URL",
    "MockServerHandle",
    "start_mock_server",
]
