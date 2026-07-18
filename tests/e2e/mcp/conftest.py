"""MCP suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness handling, and the
`e2e`/`covers` markers live in the parent tests/e2e/conftest.py. McpClient holds
the shared Gateway, so the `resources` fixture tears down whatever this suite
creates (keys via the Gateway, MCP servers via the deferred cleanups).
"""

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path
from typing import Protocol, cast
from urllib.parse import urlparse

import pytest

from mcp_client import McpClient, build_client

MCP_UPSTREAM_URL = os.environ.get("E2E_MCP_UPSTREAM_URL", "http://mcp-upstream:8090/mcp")


class DdLogsReader(Protocol):
    def poll_events_for_marker(self, marker: str) -> list[object]: ...


class _DdLogsReaderBuilder(Protocol):
    def __call__(self) -> DdLogsReader: ...


def _build_dd_logs_reader() -> DdLogsReader:
    path = Path(__file__).resolve().parent.parent / "logging" / "datadog_reader.py"
    spec = importlib.util.spec_from_file_location("e2e_logging_datadog_reader", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    builder = cast(_DdLogsReaderBuilder, getattr(module, "build_dd_logs_reader"))
    return builder()


def _mcp_upstream_reachable(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname
    if host is None:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((host, port), timeout=3.0):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def client() -> McpClient:
    return build_client()


@pytest.fixture(scope="session")
def dd_logs() -> DdLogsReader:
    return _build_dd_logs_reader()


@pytest.fixture
def require_math_upstream() -> None:
    if not _mcp_upstream_reachable(MCP_UPSTREAM_URL):
        pytest.skip(
            f"MCP math upstream not reachable at {MCP_UPSTREAM_URL}; "
            "start the mcp-upstream compose service or set E2E_MCP_UPSTREAM_URL"
        )
