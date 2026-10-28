"""Shared access to the canned OpenAI mock used across the test suite.

Tests that need a fake OpenAI-shaped endpoint point ``api_base`` at
``FAKE_OPENAI_API_BASE`` instead of a hosted URL, so the suite never depends on
an external service staying up. ``ensure_fake_openai_endpoint`` returns a base
URL that is actually serving: it reuses a server already listening on that base
(it answers ``/health``) and otherwise spawns ``_fake_openai_endpoint_server.py``
detached from the spawning process so it survives until the host goes away.

We deliberately do not register an interpreter-exit teardown. Under
``pytest-xdist`` the spawn happens inside whichever worker wins the race, and
workers exit independently as their queues drain; a per-worker ``atexit`` would
terminate the shared mock mid-session for the workers still running. In CI the
container is ephemeral, and locally the next run reuses the still-healthy server
via ``/health`` (or a developer can free the port manually).

The resolved base honors a ``FAKE_OPENAI_API_BASE`` env var only when it points
at a loopback host; a remote value (CI sets it to the old hosted mock) is ignored
in favor of the local default. Otherwise this helper would try to bind a local
server to a remote host and time out, which is the exact external dependency the
local mock exists to remove.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path
from typing import Final
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

_LOCAL_DEFAULT: Final = "http://127.0.0.1:8190"
_LOOPBACK_HOSTS: Final = frozenset({"127.0.0.1", "localhost", "::1"})


def _resolve_base() -> str:
    configured = os.environ.get("FAKE_OPENAI_API_BASE")
    if configured and urlsplit(configured).hostname in _LOOPBACK_HOSTS:
        return configured
    return _LOCAL_DEFAULT


FAKE_OPENAI_API_BASE: Final = _resolve_base()

_SERVER_SCRIPT: Final = (
    Path(__file__).resolve().parent / "_fake_openai_endpoint_server.py"
)
_HEALTH_TIMEOUT_SECONDS: Final = 30.0


def _is_healthy(base: str) -> bool:
    try:
        with urlopen(f"{base.rstrip('/')}/health", timeout=1) as response:
            return response.status == 200
    except (URLError, OSError):
        return False


def _wait_until_healthy(base: str) -> None:
    deadline = time.monotonic() + _HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if _is_healthy(base):
            return
        time.sleep(0.5)
    raise RuntimeError(
        f"fake OpenAI endpoint at {base} did not become healthy within {_HEALTH_TIMEOUT_SECONDS}s"
    )


@lru_cache(maxsize=1)
def ensure_fake_openai_endpoint() -> str:
    base = FAKE_OPENAI_API_BASE
    if _is_healthy(base):
        return base
    parts = urlsplit(base)
    subprocess.Popen(
        [
            sys.executable,
            str(_SERVER_SCRIPT),
            "--host",
            parts.hostname or "127.0.0.1",
            "--port",
            str(parts.port or 8190),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    _wait_until_healthy(base)
    return base
