"""Live e2e: a virtual key without MCP access is denied an MCP server's tools.

An admin registers an upstream MCP server through the management API (persisted in
the DB, picked up without a restart) and queues its deletion. Two keys are created
against that one server: one granted access through `object_permission.mcp_servers`
and one with no MCP grant at all. The permitted key is the control that proves the
upstream is alive and the tool is callable, so a failure on the denied key is an
authorization denial rather than a dead server. The denied key must then see none
of the server's tools on `tools/list` and must be refused with a 403 on
`tools/call`.

Both the recorded state (the server is registered; the permitted key resolves its
tools) and the enforced behavior (the unpermitted key sees nothing and is blocked)
are asserted, so a regression that leaks tools to an ungranted key or drops the
call-time permission check fails here.
"""

import os

import pytest

from e2e_config import unique_marker
from e2e_http import UnknownApiError, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient

pytestmark = pytest.mark.e2e

MCP_UPSTREAM_URL = os.environ.get("E2E_MCP_UPSTREAM_URL", "http://mcp-upstream:8090/mcp")
MATH_TOOLS = frozenset({"add", "multiply"})


def _register_math_server(client: McpClient, resources: ResourceManager) -> str:
    name = f"e2e_math_{unique_marker()}"
    server_id = client.register_server(server_name=name, alias=name, url=MCP_UPSTREAM_URL)
    resources.defer(lambda: client.delete_server(server_id))
    return server_id


def _key(client: McpClient, resources: ResourceManager, *, mcp_servers: list[str] | None) -> str:
    label = "allowed" if mcp_servers else "denied"
    key = client.generate_key(user_id=f"e2e-mcp-{label}-{unique_marker()}", mcp_servers=mcp_servers)
    resources.defer(lambda: client.gateway.delete_key(key))
    return key


def _assert_registered(client: McpClient, server_id: str) -> None:
    registered = {row.server_id for row in client.registered_servers()}
    assert server_id in registered, f"registered server {server_id} absent from /v1/mcp/server: {registered}"


class TestMcpKeyWithoutAccessIsDenied:
    @pytest.mark.covers("mcp.list_tools.api_key.denied_without_permission")
    def test_list_tools_denied_without_permission(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        server_id = _register_math_server(client, resources)
        _assert_registered(client, server_id)

        permitted_key = _key(client, resources, mcp_servers=[server_id])
        denied_key = _key(client, resources, mcp_servers=None)

        permitted_tools = unwrap(client.list_tools(permitted_key)).tool_names_for_server(server_id)
        assert MATH_TOOLS <= permitted_tools, (
            f"granted key did not see the server's tools (upstream dead or grant not applied): "
            f"{permitted_tools}"
        )

        denied_tools = unwrap(client.list_tools(denied_key)).tool_names_for_server(server_id)
        assert denied_tools == frozenset(), (
            f"ungranted key saw the server's tools; tools/list leaked across the permission "
            f"boundary: {denied_tools}"
        )

    @pytest.mark.covers("mcp.call_tool.api_key.denied_without_permission")
    def test_call_tool_denied_without_permission(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        server_id = _register_math_server(client, resources)
        _assert_registered(client, server_id)

        permitted_key = _key(client, resources, mcp_servers=[server_id])
        denied_key = _key(client, resources, mcp_servers=None)

        permitted_tools = unwrap(client.list_tools(permitted_key)).tool_names_for_server(server_id)
        assert "add" in permitted_tools, (
            f"granted key did not discover the add tool (upstream dead or grant not applied): "
            f"{permitted_tools}"
        )

        permitted_call = unwrap(
            client.call_tool(permitted_key, server_id=server_id, name="add", arguments={"a": 3, "b": 4})
        )
        assert permitted_call.is_error is not True, f"granted key's tool call errored: {permitted_call}"
        assert permitted_call.first_text == "7", (
            f"granted key's add(3, 4) did not return 7 (upstream not reachable): {permitted_call}"
        )

        match client.call_tool(denied_key, server_id=server_id, name="add", arguments={"a": 3, "b": 4}):
            case UnknownApiError(status_code=403, body=body):
                assert "access_denied" in body, f"403 was not an MCP access denial: {body}"
            case other:
                pytest.fail(f"ungranted key's tool call was not refused with 403 access_denied: {other}")
