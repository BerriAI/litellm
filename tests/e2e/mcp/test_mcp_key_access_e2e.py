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
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient

pytestmark = pytest.mark.e2e


def _key(client: McpClient, resources: ResourceManager, *, mcp_servers: list[str] | None) -> str:
    label = "allowed" if mcp_servers else "denied"
    key = client.generate_key(user_id=f"e2e-mcp-{label}-{unique_marker()}", mcp_servers=mcp_servers)
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


class TestMcpKeyGrantByAlias:
    def test_alias_grant_persists_verbatim_and_lists_tools(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        """A key granted an MCP server by its alias must store the alias, not the
        resolved server_id: in a shared-DB multi-region deployment each instance
        derives a different id for the same config server, so only the alias
        grants access on every region. The same key must still see the server's
        tools, proving the alias grant is honored at request time."""
        server_id = register_datadog_mcp(client, resources)
        client.await_registered(server_id)
        alias = next(row.alias for row in client.registered_servers() if row.server_id == server_id)
        assert alias, f"registered server {server_id} has no alias to grant by"

        key = _key(client, resources, mcp_servers=[alias])

        stored = client.proxy.key_info(key).object_permission
        assert stored is not None and stored.mcp_servers == [alias], (
            f"alias grant was rewritten before persisting (expected [{alias!r}]): "
            f"{stored.mcp_servers if stored else None}. A stored server_id is region-local "
            f"and breaks the grant on every other instance sharing this database"
        )

        _ = client.await_tool(key, server_id, SEARCH_LOGS_TOOL)


class TestMcpKeyWithoutAccessIsDenied:
    @pytest.mark.covers("mcp.list_tools.api_key.denied_without_permission")
    def test_list_tools_denied_without_permission(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        server_id = register_datadog_mcp(client, resources)
        client.await_registered(server_id)

        permitted_key = _key(client, resources, mcp_servers=[server_id])
        denied_key = _key(client, resources, mcp_servers=None)

        _ = client.await_tool(permitted_key, server_id, SEARCH_LOGS_TOOL)

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
        client.await_registered(server_id)

        permitted_key = _key(client, resources, mcp_servers=[server_id])
        denied_key = _key(client, resources, mcp_servers=None)

        tool_name = client.await_tool(permitted_key, server_id, SEARCH_LOGS_TOOL)

        search_args = {
            "query": "service:litellm",
            "from": DD_SEARCH_FROM,
            "to": "now",
            "max_tokens": 1000,
        }
        permitted_call = client.await_call_tool(
            permitted_key, server_id=server_id, name=tool_name, arguments=search_args
        )
        assert permitted_call.is_error is not True, f"granted key's tool call errored: {permitted_call}"

        denied = client.await_call_tool_denied(
            denied_key, server_id=server_id, name=tool_name, arguments=search_args
        )
        assert "access_denied" in denied.body, f"403 was not an MCP access denial: {denied.body}"
