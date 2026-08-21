"""Live e2e: the gateway's allowed_tools filtering on the Datadog MCP server.

A server registered with a narrow allowed_tools list hides every other tool
from tools/list, and tools/call on a hidden tool is blocked (403 or 404,
depending on whether the tool was in the gateway's resolved tool map).

Note: disallowed_tools and allowed_params are config-only fields today, they
have no DB column in the Prisma schema and are silently dropped on the
management API path. Those cells are product gaps, not test gaps.

All against the real Datadog remote MCP server.
"""

from __future__ import annotations

import pytest

from datadog_mcp import SEARCH_LOGS_TOOL, assert_dd_mcp_creds, register_datadog_mcp
from e2e_config import DD_SEARCH_FROM, unique_marker
from e2e_http import UnknownApiError, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient, McpToolArguments

pytestmark = pytest.mark.e2e

BOGUS_TOOL = "nonexistent_e2e_tool"


def _search_args(query: str) -> McpToolArguments:
    return {
        "query": query,
        "from": DD_SEARCH_FROM,
        "to": "now",
        "max_tokens": 500,
    }


def _key(
    client: McpClient, resources: ResourceManager, server_id: str, label: str
) -> str:
    key = client.generate_key(
        user_id=f"e2e-mcp-{label}-{unique_marker()}",
        mcp_servers=[server_id],
    )
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


class TestAllowedToolsScoping:
    @pytest.mark.covers("mcp.list_tools.api_key.allowed_tools_scoped")
    def test_list_tools_hides_non_allowed_tools(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_mcp_creds()
        dd = register_datadog_mcp(client, resources, allowed_tools=[BOGUS_TOOL])
        client.await_registered(dd.server_id)

        key = _key(client, resources, dd.server_id, "allow-list")

        tools = unwrap(client.list_tools(key)).tool_names_for_server(dd.server_id)
        assert SEARCH_LOGS_TOOL not in tools, (
            f"search_datadog_logs must be hidden by the allowed_tools filter "
            f"(only {BOGUS_TOOL!r} is allowed), but it appeared in tools/list: {tools}"
        )

    @pytest.mark.covers("mcp.call_tool.api_key.allowed_tools_scoped")
    def test_call_tool_denied_outside_allowed_tools(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_mcp_creds()
        dd_all = register_datadog_mcp(client, resources)
        client.await_registered(dd_all.server_id)
        dd_narrow = register_datadog_mcp(client, resources, allowed_tools=[BOGUS_TOOL])
        client.await_registered(dd_narrow.server_id)

        key_all = _key(client, resources, dd_all.server_id, "allow-call-control")
        _ = client.await_tool(key_all, dd_all.server_id, SEARCH_LOGS_TOOL)

        key_narrow = _key(client, resources, dd_narrow.server_id, "allow-call-narrow")

        result = client.call_tool(
            key_narrow,
            server_id=dd_narrow.server_id,
            name=SEARCH_LOGS_TOOL,
            arguments=_search_args("service:litellm"),
        )
        match result:
            case UnknownApiError(status_code=403):
                pass
            case UnknownApiError(status_code=404):
                pass
            case _:
                pytest.fail(
                    f"calling a tool outside allowed_tools must be blocked (403 or 404), "
                    f"got: {result}"
                )
