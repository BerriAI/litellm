"""MCP suite's `client` fixture.

The shared lifecycle (resources/scoped_key), proxy liveness handling, and the
`e2e`/`covers` markers live in the parent tests/e2e/conftest.py. McpClient holds
the shared Gateway, so the `resources` fixture tears down whatever this suite
creates (keys via the Gateway, MCP servers via the deferred cleanups).
"""

import pytest

from mcp_client import McpClient, build_client


@pytest.fixture(scope="session")
def client() -> McpClient:
    return build_client()
