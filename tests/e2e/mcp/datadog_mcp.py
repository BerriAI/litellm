"""Shared helpers for e2e tests that register the real Datadog remote MCP server."""

from __future__ import annotations

import os

from e2e_config import datadog_mcp_url, unique_marker
from lifecycle import ResourceManager
from mcp_client import McpClient

SEARCH_LOGS_TOOL = "search_datadog_logs"


def _dd_api_key() -> str:
    return os.environ.get("DD_API_KEY", "").strip()


def _dd_app_key() -> str:
    return os.environ.get("DD_APP_KEY", "").strip()


def assert_dd_mcp_creds() -> None:
    if not _dd_api_key() or not _dd_app_key():
        import pytest

        pytest.fail(
            "Datadog MCP e2e requires DD_API_KEY and DD_APP_KEY "
            "(header auth to mcp.<site>/v1/mcp; on the cluster the secret manager "
            "injects them, locally tests/e2e/.env)"
        )


def register_datadog_mcp(client: McpClient, resources: ResourceManager) -> str:
    assert_dd_mcp_creds()
    name = f"e2e_dd_mcp_{unique_marker()}"
    server_id = client.register_server(
        server_name=name,
        alias=name,
        url=datadog_mcp_url(toolsets="core"),
        transport="http",
        static_headers={
            "DD-API-KEY": _dd_api_key(),
            "DD-APPLICATION-KEY": _dd_app_key(),
        },
        allowed_tools=[SEARCH_LOGS_TOOL],
    )
    resources.defer(lambda: client.delete_server(server_id))
    return server_id
