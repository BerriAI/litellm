"""Live e2e: the gateway brokers Datadog's static-header HTTP transport and
multi-server tool namespacing.

Three cells in one spec:
1. upstream_static_auth: a server registered with static_headers (Datadog's
   DD-API-KEY / DD-APPLICATION-KEY) injects them on every upstream call, so
   search_datadog_logs succeeds. The tool result must be non-empty, proving
   the static credentials reached Datadog.
2. transport_http: the Datadog server uses the streamable HTTP transport, and
   a successful tools/list + tools/call round-trip proves that transport path
   works end to end. This is the same call as upstream_static_auth but asserts
   the transport-specific cell.
3. namespaced_multi_server: two registered servers' tools remain
   distinguishable on the aggregate tools/list (each tool carries its own
   mcp_info.server_id), so a multi-server tenant never sees tools collide.
"""

from __future__ import annotations

import pytest

from datadog_mcp import SEARCH_LOGS_TOOL, assert_dd_mcp_creds, register_datadog_mcp
from e2e_config import DD_SEARCH_FROM, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient, McpToolArguments

pytestmark = pytest.mark.e2e


def _search_args(query: str) -> McpToolArguments:
    return {
        "query": query,
        "from": DD_SEARCH_FROM,
        "to": "now",
        "max_tokens": 500,
    }


class TestUpstreamStaticAuthAndTransport:
    @pytest.mark.covers(
        "mcp.call_tool.api_key.upstream_static_auth",
        "mcp.call_tool.api_key.transport_http",
    )
    def test_static_header_http_transport_call_succeeds(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_mcp_creds()
        dd = register_datadog_mcp(client, resources)
        client.await_registered(dd.server_id)

        key = client.generate_key(
            user_id=f"e2e-mcp-static-{unique_marker()}",
            mcp_servers=[dd.server_id],
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        tool_name = client.await_tool(key, dd.server_id, SEARCH_LOGS_TOOL)
        call = client.await_call_tool(
            key,
            server_id=dd.server_id,
            name=tool_name,
            arguments=_search_args("service:litellm"),
        )
        assert call.is_error is not True, (
            f"search_datadog_logs errored with static headers + http transport: {call}"
        )
        assert call.all_text, (
            "search_datadog_logs returned empty text; the static DD-API-KEY / "
            "DD-APPLICATION-KEY headers may not have reached the upstream"
        )


class TestNamespacedMultiServer:
    @pytest.mark.covers("mcp.list_tools.api_key.namespaced_multi_server")
    def test_two_servers_tools_remain_distinguishable(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        assert_dd_mcp_creds()
        dd_a = register_datadog_mcp(client, resources)
        dd_b = register_datadog_mcp(client, resources)
        client.await_registered(dd_a.server_id)
        client.await_registered(dd_b.server_id)

        key_a = client.generate_key(
            user_id=f"e2e-mcp-ns-a-{unique_marker()}",
            mcp_servers=[dd_a.server_id],
        )
        resources.defer(lambda: client.proxy.delete_key(key_a))
        key_b = client.generate_key(
            user_id=f"e2e-mcp-ns-b-{unique_marker()}",
            mcp_servers=[dd_b.server_id],
        )
        resources.defer(lambda: client.proxy.delete_key(key_b))

        _ = client.await_tool(key_a, dd_a.server_id, SEARCH_LOGS_TOOL)
        _ = client.await_tool(key_b, dd_b.server_id, SEARCH_LOGS_TOOL)

        key_both = client.generate_key(
            user_id=f"e2e-mcp-ns-both-{unique_marker()}",
            mcp_servers=[dd_a.server_id, dd_b.server_id],
        )
        resources.defer(lambda: client.proxy.delete_key(key_both))

        tools = unwrap(client.list_tools(key_both))
        a_tools = tools.tool_names_for_server(dd_a.server_id)
        b_tools = tools.tool_names_for_server(dd_b.server_id)

        assert a_tools, (
            f"server A's tools are missing from the aggregate list; "
            f"the multi-server namespace collapsed: {tools.tools}"
        )
        assert b_tools, (
            f"server B's tools are missing from the aggregate list; "
            f"the multi-server namespace collapsed: {tools.tools}"
        )

        a_entries = tuple(
            t for t in tools.tools
            if t.mcp_info and t.mcp_info.server_id == dd_a.server_id
        )
        b_entries = tuple(
            t for t in tools.tools
            if t.mcp_info and t.mcp_info.server_id == dd_b.server_id
        )
        assert len(a_entries) == len(b_tools) and len(b_entries) == len(b_tools), (
            f"each server's tools must carry its own mcp_info.server_id so a "
            f"multi-server tenant can tell them apart; "
            f"A entries={a_entries}, B entries={b_entries}"
        )
