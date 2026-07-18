"""MCP suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness handling, and the
`e2e`/`covers` markers live in the parent tests/e2e/conftest.py. McpClient holds
the shared Gateway, so the `resources` fixture tears down whatever this suite
creates (keys via the Gateway, MCP servers via the deferred cleanups).
"""

from __future__ import annotations

import os
import socket
from urllib.parse import urlparse

import pytest

from mcp_client import McpClient, build_client

MCP_UPSTREAM_URL = os.environ.get("E2E_MCP_UPSTREAM_URL", "http://mcp-upstream:8090/mcp")


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


@pytest.fixture(scope="session", autouse=True)
def require_mcp_upstream() -> None:
    if not _mcp_upstream_reachable(MCP_UPSTREAM_URL):
        pytest.skip(
            f"MCP upstream not reachable at {MCP_UPSTREAM_URL}; "
            "start the mcp-upstream compose service or set E2E_MCP_UPSTREAM_URL"
        )


@pytest.fixture(scope="session")
def client() -> McpClient:
    return build_client()
