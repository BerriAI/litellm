"""Live e2e: MCP tool selection via access group at key creation.

An admin registers the Datadog remote MCP server tagged with a server-side
access group (`mcp_access_groups`). A key minted with that access group
(`object_permission.mcp_access_groups`) sees the server's tools; a key minted
with a different group does not. This exercises access-group-scoped tool
selection, the enterprise MCP surface where keys are granted tool access groups
rather than explicit server ids.

A tools/list that leaks the server across the access-group boundary fails hard.
Requires DD_API_KEY + DD_APP_KEY (the suite's real MCP upstream).
"""

import pytest

from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient

pytestmark = pytest.mark.e2e


class TestMcpAccessGroupToolSelection:
    @pytest.mark.covers("mcp.list_tools.api_key.access_group_scoped")
    def test_access_group_scopes_tool_selection(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        group = f"e2e-mcp-grp-{unique_marker()}"
        server_id = register_datadog_mcp(client, resources, mcp_access_groups=[group])

        granted = client.generate_key(
            user_id=f"e2e-mcp-ag-granted-{unique_marker()}",
            mcp_servers=None,
            mcp_access_groups=[group],
        )
        resources.defer(lambda: client.proxy.delete_key(granted))

        other = client.generate_key(
            user_id=f"e2e-mcp-ag-other-{unique_marker()}",
            mcp_servers=None,
            mcp_access_groups=[f"e2e-mcp-grp-absent-{unique_marker()}"],
        )
        resources.defer(lambda: client.proxy.delete_key(other))

        granted_tools = unwrap(client.list_tools(granted))
        assert granted_tools.tool_name_containing(server_id, SEARCH_LOGS_TOOL) is not None, (
            f"key granted access group {group} did not see the tagged server's tool "
            f"(upstream dead or access-group grant not applied): "
            f"{granted_tools.tool_names_for_server(server_id)}"
        )

        other_tools = unwrap(client.list_tools(other)).tool_names_for_server(server_id)
        assert other_tools == frozenset(), (
            f"key with a different access group saw the server's tools; access-group tool "
            f"selection leaked across the boundary: {other_tools}"
        )
