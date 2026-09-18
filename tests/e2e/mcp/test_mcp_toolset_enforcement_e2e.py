"""Live e2e: a key granted a toolset lists exactly the toolset's tools.

An admin registers the real Datadog remote MCP server with its whole core toolset
exposed, discovers two of its tool names through a key granted the server outright,
and curates a toolset naming exactly those two. A second key is granted the server
plus that toolset, and its tools/list must come back as exactly those two names: no
more, so the rest of the server's catalog stays hidden behind the toolset, and no
fewer, so a tool stored under one name and read under another (which granted
nothing) fails here first. Requires DD_API_KEY + DD_APP_KEY (the suite's real MCP
upstream).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Final

import pytest
from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import DD_SEARCH_FROM, unique_marker
from e2e_http import UnknownApiError, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from management.management_client import build_client as build_management_client
from models import (
    KeyGenerateBody,
    ObjectPermission,
    OrgNewBody,
    TeamNewBody,
    ToolsetCreateBody,
    ToolsetTool,
    UserNewBody,
)

pytestmark = pytest.mark.e2e


def _key(
    client: McpClient,
    resources: ResourceManager,
    label: str,
    *,
    server_id: str,
    toolset_id: str | None = None,
) -> str:
    key: Final = client.generate_key(
        user_id=f"e2e-mcp-{label}-{unique_marker()}",
        mcp_servers=[server_id],
        mcp_toolsets=None if toolset_id is None else [toolset_id],
    )
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


def _wire_prefix(wire_name: str, tool_name: str, catalog: frozenset[str]) -> str:
    """The prefix tools/list puts in front of one server's tool names, measured off a
    tool whose own name is known rather than guessed from the alias. A toolset grants
    by the tool's own name, never the wire name, and the prefix is whatever the proxy
    is configured to build (the alias, or a short server id), so measuring it is the
    only way to cross between the two."""
    assert wire_name.endswith(tool_name), f"tools/list served {wire_name!r}, expected it to end with {tool_name!r}"
    prefix: Final = wire_name[: len(wire_name) - len(tool_name)]
    unprefixed: Final = frozenset(name for name in catalog if not name.startswith(prefix))
    assert not unprefixed, (
        f"every tool of one server shares the wire prefix {prefix!r}, so {sorted(unprefixed)} "
        f"cannot be reduced to the names a toolset grants by"
    )
    return prefix


class TestMcpToolsetEnforcement:
    @pytest.mark.covers("mcp.list_tools.api_key.toolset_scoped")
    def test_key_granted_a_toolset_lists_exactly_its_tools(self, client: McpClient, resources: ResourceManager) -> None:
        server_id: Final = register_datadog_mcp(client, resources, allowed_tools=None)
        client.await_registered(server_id)

        catalog_key: Final = _key(client, resources, "catalog", server_id=server_id)
        known_wire: Final = client.await_tool(catalog_key, server_id, SEARCH_LOGS_TOOL)
        catalog: Final = unwrap(client.list_tools(catalog_key)).tool_names_for_server(server_id)
        assert len(catalog) > 2, (
            f"the Datadog core toolset must serve more tools than the toolset names, or the "
            f"restriction has nothing to hide; got {sorted(catalog)}"
        )
        prefix: Final = _wire_prefix(known_wire, SEARCH_LOGS_TOOL, catalog)
        chosen_wire: Final = frozenset(sorted(catalog)[:2])
        chosen: Final = frozenset(name.removeprefix(prefix) for name in chosen_wire)

        toolset: Final = client.proxy.create_toolset(
            ToolsetCreateBody(
                toolset_name=f"e2e_toolset_{unique_marker()}",
                description="two Datadog tools",
                tools=[ToolsetTool(server_id=server_id, tool_name=name) for name in sorted(chosen)],
            )
        )
        resources.defer(lambda: client.proxy.delete_toolset(toolset.toolset_id))
        assert frozenset(tool.tool_name for tool in toolset.tools) == chosen, (
            f"toolset stored {toolset.tools}, expected the two names {sorted(chosen)} verbatim"
        )

        scoped_key: Final = _key(client, resources, "toolset", server_id=server_id, toolset_id=toolset.toolset_id)
        listed: Final = client.await_tools(scoped_key, server_id, expected=chosen_wire)
        assert listed == chosen_wire, (
            f"a key granted the toolset must list exactly its two tools; "
            f"got {sorted(listed)}, expected {sorted(chosen_wire)}"
        )


def _assert_principal_toolset(client: McpClient, resources: ResourceManager, principal: str) -> None:
    server_id = register_datadog_mcp(client, resources, allowed_tools=None)
    client.await_registered(server_id)
    control_key = _key(client, resources, "principal-control", server_id=server_id)
    toolset = client.proxy.create_toolset(
        ToolsetCreateBody(
            toolset_name=f"e2e_principal_{unique_marker()}",
            tools=[ToolsetTool(server_id=server_id, tool_name=SEARCH_LOGS_TOOL)],
        )
    )
    resources.defer(lambda: client.proxy.delete_toolset(toolset.toolset_id))
    assert [(tool.server_id, tool.tool_name) for tool in toolset.tools] == [(server_id, SEARCH_LOGS_TOOL)]
    management = build_management_client(client.proxy)
    permission = ObjectPermission(mcp_servers=[server_id], mcp_toolsets=[toolset.toolset_id])
    if principal == "user":
        user_id = management.create_user(
            UserNewBody(
                user_email=f"e2e-toolset-{unique_marker()}@example.com",
                user_role="internal_user",
                auto_create_key=False,
                object_permission=ObjectPermission(mcp_toolsets=[toolset.toolset_id]),
            )
        )
        resources.defer(lambda: management.delete_user(user_id))
        scoped_key = client.generate_key(user_id=user_id, mcp_servers=[server_id])
    else:
        if principal == "organization":
            org_id = management.create_org(
                OrgNewBody(
                    organization_alias=f"e2e-toolset-{unique_marker()}",
                    object_permission=permission,
                )
            )
            resources.defer(lambda: management.delete_org(org_id))
            team_body = TeamNewBody(team_alias=f"e2e-toolset-{unique_marker()}", organization_id=org_id)
        else:
            team_body = TeamNewBody(team_alias=f"e2e-toolset-{unique_marker()}", object_permission=permission)
        team_id = management.create_team(team_body)
        resources.defer(lambda: management.delete_team(team_id))
        scoped_key = client.proxy.generate_key(KeyGenerateBody(team_id=team_id))
    resources.defer(lambda: client.proxy.delete_key(scoped_key))

    for transport in client.proxy.replicas_for("/mcp-rest/tools/list").values():
        replica = McpClient(proxy=replace(client.proxy, transport=transport))
        granted = replica.await_tool_entry(control_key, server_id, SEARCH_LOGS_TOOL)
        catalog = unwrap(replica.list_tools(control_key)).tool_names_for_server(server_id)
        outside = sorted(catalog - {granted.name})
        assert outside, f"uncapped upstream must expose a tool outside the grant: {catalog}"
        expected = frozenset({granted.name})
        assert replica.await_tools(scoped_key, server_id, expected=expected) == expected
        denied = replica.call_tool(scoped_key, server_id=server_id, name=outside[0], arguments={})
        assert isinstance(denied, UnknownApiError) and denied.status_code == 403, denied
        assert "access_denied" in denied.body, denied.body
        arguments = {"query": f"service:e2e-toolset-{unique_marker()}", "from": DD_SEARCH_FROM}
        granted.assert_arguments_are_documented(arguments)
        result = unwrap(replica.call_tool(scoped_key, server_id=server_id, name=granted.name, arguments=arguments))
        assert result.is_error is not True, result.all_text


class TestMcpToolsetEnforcementPerLevel:
    @pytest.mark.covers(
        "mcp.list_tools.api_key.team_toolset_scoped", "mcp.call_tool.api_key.team_toolset_denied_outside"
    )
    def test_team_toolset_narrows_team_key(self, client: McpClient, resources: ResourceManager) -> None:
        _assert_principal_toolset(client, resources, "team")

    @pytest.mark.covers("mcp.list_tools.api_key.org_toolset_scoped", "mcp.call_tool.api_key.org_toolset_denied_outside")
    def test_org_toolset_caps_inherited_team_key(self, client: McpClient, resources: ResourceManager) -> None:
        _assert_principal_toolset(client, resources, "organization")

    @pytest.mark.covers(
        "mcp.list_tools.api_key.user_toolset_scoped", "mcp.call_tool.api_key.user_toolset_denied_outside"
    )
    def test_user_toolset_ceils_own_key_grant(self, client: McpClient, resources: ResourceManager) -> None:
        _assert_principal_toolset(client, resources, "user")
