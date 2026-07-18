"""Live e2e: a virtual key without MCP access is denied a real MCP server's tools.

An admin registers the Datadog remote MCP server through the management API
(persisted in the DB, picked up without a restart) and queues its deletion. Two
keys are created against that one server: one granted access through
`object_permission.mcp_servers` and one with no MCP grant at all. The permitted
key is the control that proves the upstream is alive and the tool is callable,
so a failure on the denied key is an authorization denial rather than a dead
server. The denied key must then see none of the server's tools on `tools/list`
and must be refused with a 403 on `tools/call`.
"""

from __future__ import annotations

import pytest

from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import DD_SEARCH_FROM, unique_marker
from e2e_http import UnknownApiError, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient

pytestmark = pytest.mark.e2e


def _key(client: McpClient, resources: ResourceManager, *, mcp_servers: list[str] | None) -> str:
    label = "allowed" if mcp_servers else "denied"
    key = client.generate_key(user_id=f"e2e-mcp-{label}-{unique_marker()}", mcp_servers=mcp_servers)
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


def _assert_registered(client: McpClient, server_id: str) -> None:
    registered = {row.server_id for row in client.registered_servers()}
    assert server_id in registered, f"registered server {server_id} absent from /v1/mcp/server: {registered}"


class TestMcpKeyWithoutAccessIsDenied:
    @pytest.mark.covers("mcp.list_tools.api_key.denied_without_permission")
    def test_list_tools_denied_without_permission(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        server_id = register_datadog_mcp(client, resources)
        _assert_registered(client, server_id)

        permitted_key = _key(client, resources, mcp_servers=[server_id])
        denied_key = _key(client, resources, mcp_servers=None)

        permitted = unwrap(client.list_tools(permitted_key))
        tool_name = permitted.tool_name_containing(server_id, SEARCH_LOGS_TOOL)
        assert tool_name is not None, (
            f"granted key did not see {SEARCH_LOGS_TOOL} (upstream dead or grant not applied): "
            f"{permitted.tool_names_for_server(server_id)}"
        )

        denied_tools = unwrap(client.list_tools(denied_key)).tool_names_for_server(server_id)
        assert denied_tools == frozenset(), (
            f"ungranted key saw the server's tools; tools/list leaked across the permission "
            f"boundary: {denied_tools}"
        )

    @pytest.mark.covers("mcp.call_tool.api_key.denied_without_permission")
    def test_call_tool_denied_without_permission(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        server_id = register_datadog_mcp(client, resources)
        _assert_registered(client, server_id)

        permitted_key = _key(client, resources, mcp_servers=[server_id])
        denied_key = _key(client, resources, mcp_servers=None)

        permitted = unwrap(client.list_tools(permitted_key))
        tool_name = permitted.tool_name_containing(server_id, SEARCH_LOGS_TOOL)
        assert tool_name is not None, (
            f"granted key did not discover {SEARCH_LOGS_TOOL} (upstream dead or grant not applied): "
            f"{permitted.tool_names_for_server(server_id)}"
        )

        search_args = {
            "query": "service:litellm",
            "from": DD_SEARCH_FROM,
            "to": "now",
            "max_tokens": 1000,
            "telemetry": {"intent": "e2e control call proving granted key can invoke Datadog MCP"},
        }
        permitted_call = unwrap(
            client.call_tool(permitted_key, server_id=server_id, name=tool_name, arguments=search_args)
        )
        assert permitted_call.is_error is not True, f"granted key's tool call errored: {permitted_call}"

        match client.call_tool(denied_key, server_id=server_id, name=tool_name, arguments=search_args):
            case UnknownApiError(status_code=403, body=body):
                assert "access_denied" in body, f"403 was not an MCP access denial: {body}"
            case other:
                pytest.fail(f"ungranted key's tool call was not refused with 403 access_denied: {other}")
