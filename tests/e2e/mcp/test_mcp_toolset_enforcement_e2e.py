"""Live e2e: an MCP toolset attached to a team, org, or internal user narrows a
real MCP server to exactly the tools the toolset names (LIT-5749).

An admin registers the Datadog remote MCP server with its full core toolset (no
`allowed_tools` cap, so more than one tool exists to be denied), creates a
toolset naming only `search_datadog_logs` on it, and attaches that toolset at
one principal level per test. A control key granted the whole server first
proves the upstream is alive and serves a second tool, so a later denial is an
authorization decision rather than a dead server. The scoped key must then see
exactly the toolset's one tool on `tools/list`, be refused (403) calling any
other tool on the same server, and still successfully call the granted tool.

Levels covered, one per test: a team holding server + toolset, a team that
inherits its grant from an org holding server + toolset, and an internal user
whose toolset ceils the servers their own key grants.
"""

from __future__ import annotations

import pytest

from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import DD_SEARCH_FROM, unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import ObjectPermission

pytestmark = pytest.mark.e2e


def _open_server_and_toolset(client: McpClient, resources: ResourceManager) -> tuple[str, str]:
    """Register the DD server uncapped and a toolset naming only the search-logs
    tool on it, returning (server_id, toolset_id)."""
    server_id = register_datadog_mcp(client, resources, allowed_tools=None)
    client.await_registered(server_id)
    toolset_id = client.create_toolset(
        name=f"e2e-search-only-{unique_marker()}",
        server_id=server_id,
        tool_names=[SEARCH_LOGS_TOOL],
    )
    resources.defer(lambda: client.delete_toolset(toolset_id))
    return server_id, toolset_id


def _control_tools(client: McpClient, resources: ResourceManager, server_id: str) -> tuple[str, str]:
    """A fully-granted control key's view of the server: the fully-qualified
    search-logs tool name and one tool outside the toolset. Proves the upstream
    is alive and actually serves something a toolset can deny."""
    control_key = client.generate_key(user_id=f"e2e-ts-control-{unique_marker()}", mcp_servers=[server_id])
    resources.defer(lambda: client.proxy.delete_key(control_key))
    granted_tool = client.await_tool(control_key, server_id, SEARCH_LOGS_TOOL)
    all_tools = unwrap(client.list_tools(control_key)).tool_names_for_server(server_id)
    outside = sorted(all_tools - {granted_tool})
    assert outside, (
        f"the uncapped Datadog core toolset served only {all_tools}; a toolset "
        f"cannot be proven to narrow a one-tool server"
    )
    return granted_tool, outside[0]


def _assert_toolset_ceiling(
    client: McpClient,
    scoped_key: str,
    *,
    server_id: str,
    granted_tool: str,
    outside_tool: str,
) -> None:
    """The scoped key sees exactly the toolset's tool, is 403-refused on a tool
    outside it, and can still execute the granted one."""
    _ = client.await_tool(scoped_key, server_id, SEARCH_LOGS_TOOL)
    listed = unwrap(client.list_tools(scoped_key)).tool_names_for_server(server_id)
    assert listed == frozenset({granted_tool}), (
        f"toolset-scoped key must list exactly {granted_tool!r}; the toolset ceiling leaked: {sorted(listed)}"
    )

    denied = client.await_call_tool_denied(scoped_key, server_id=server_id, name=outside_tool, arguments={})
    assert denied.status_code == 403

    result = client.await_call_tool(
        scoped_key,
        server_id=server_id,
        name=granted_tool,
        arguments={"query": f"service:e2e-toolset-{unique_marker()}", "from": DD_SEARCH_FROM},
    )
    assert result.is_error is not True, f"the toolset-granted tool must stay callable; upstream said: {result.all_text}"


class TestMcpToolsetEnforcementPerLevel:
    @pytest.mark.covers("mcp.list_tools.api_key.team_toolset_scoped")
    @pytest.mark.covers("mcp.call_tool.api_key.team_toolset_denied_outside")
    def test_team_toolset_narrows_team_key(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        server_id, toolset_id = _open_server_and_toolset(client, resources)
        granted_tool, outside_tool = _control_tools(client, resources, server_id)

        team_id = client.create_team(
            alias=f"e2e-ts-team-{unique_marker()}",
            object_permission=ObjectPermission(mcp_servers=[server_id], mcp_toolsets=[toolset_id]),
        )
        resources.defer(lambda: client.delete_team(team_id))
        team_key = client.generate_key(
            user_id=f"e2e-ts-team-user-{unique_marker()}",
            mcp_servers=None,
            team_id=team_id,
        )
        resources.defer(lambda: client.proxy.delete_key(team_key))

        _assert_toolset_ceiling(
            client,
            team_key,
            server_id=server_id,
            granted_tool=granted_tool,
            outside_tool=outside_tool,
        )

    @pytest.mark.covers("mcp.list_tools.api_key.org_toolset_scoped")
    @pytest.mark.covers("mcp.call_tool.api_key.org_toolset_denied_outside")
    def test_org_toolset_caps_inherited_team_key(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        server_id, toolset_id = _open_server_and_toolset(client, resources)
        granted_tool, outside_tool = _control_tools(client, resources, server_id)

        org_id = client.create_org(
            alias=f"e2e-ts-org-{unique_marker()}",
            object_permission=ObjectPermission(mcp_servers=[server_id], mcp_toolsets=[toolset_id]),
        )
        resources.defer(lambda: client.delete_org(org_id))
        # The team declares no MCP grants of its own; whatever its key can reach
        # comes from the org, so the org's toolset must cap it.
        team_id = client.create_team(alias=f"e2e-ts-org-team-{unique_marker()}", organization_id=org_id)
        resources.defer(lambda: client.delete_team(team_id))
        team_key = client.generate_key(
            user_id=f"e2e-ts-org-user-{unique_marker()}",
            mcp_servers=None,
            team_id=team_id,
        )
        resources.defer(lambda: client.proxy.delete_key(team_key))

        _assert_toolset_ceiling(
            client,
            team_key,
            server_id=server_id,
            granted_tool=granted_tool,
            outside_tool=outside_tool,
        )

    @pytest.mark.covers("mcp.list_tools.api_key.user_toolset_scoped")
    @pytest.mark.covers("mcp.call_tool.api_key.user_toolset_denied_outside")
    def test_user_toolset_ceils_own_key_grant(
        self,
        client: McpClient,
        resources: ResourceManager,
    ) -> None:
        server_id, toolset_id = _open_server_and_toolset(client, resources)
        granted_tool, outside_tool = _control_tools(client, resources, server_id)

        # The user's row holds only the toolset; their key grants the whole
        # server. The user-level toolset is a ceiling over the key's grant.
        user_id = client.create_user(
            user_email=f"e2e-ts-user-{unique_marker()}@example.com",
            object_permission=ObjectPermission(mcp_toolsets=[toolset_id]),
        )
        resources.defer(lambda: client.delete_user(user_id))
        user_key = client.generate_key(user_id=user_id, mcp_servers=[server_id])
        resources.defer(lambda: client.proxy.delete_key(user_key))

        _assert_toolset_ceiling(
            client,
            user_key,
            server_id=server_id,
            granted_tool=granted_tool,
            outside_tool=outside_tool,
        )
