"""Fixtures for the mcp e2e suite.

The shared lifecycle (`resources`, `scoped_key`), the liveness gate, and the
`e2e` marker come from the parent tests/e2e/conftest.py; this adds the suite
client. The mcp SDK lives in the `e2e-dev` dependency group (like playwright
for management/), so its import is guarded to skip the suite rather than
error at collection when it is not installed.
"""

import pytest

pytest.importorskip("mcp", reason="mcp SDK not installed; run `uv sync --inexact --group e2e-dev`")

from mcp_client import McpClient, build_client  # noqa: E402  # import must follow the importorskip guard above


@pytest.fixture(scope="session")
def client() -> McpClient:
    return build_client()
