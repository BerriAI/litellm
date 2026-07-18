"""MCP suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness handling, and the
`e2e`/`covers` markers live in the parent tests/e2e/conftest.py. McpClient holds
the shared ProxyClient, so the `resources` fixture tears down whatever this suite
creates (keys via the ProxyClient, MCP servers via the deferred cleanups).
"""

import pytest

from mcp_client import McpClient, build_client
from proxy_client import ProxyClient


@pytest.fixture(scope="session")
def client(proxy: ProxyClient) -> McpClient:
    return build_client(proxy)
