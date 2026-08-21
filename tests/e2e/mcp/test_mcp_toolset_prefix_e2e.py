"""Live e2e: the toolset tool-name prefix contract at both ends.

A toolset row names a tool on a specific server, so the stored string is the tool's own
native name; the `{server_prefix}{separator}{tool}` wire prefix is added on the way out.
Two customer incidents sit on the two ends of that contract, and each is a different
failure of the same question: how many prefixes does the served name carry?

Serving (ticket raised 2026-07-19): the prefix must be applied exactly once, even when a
native tool name happens to begin with its server's prefix. `resolve_toolset_tool_permissions`
is explicit that a leading segment is never treated as a prefix and stripped, because doing
so silently renames a real tool. Registering the Datadog server under the alias
`search_datadog` makes its native `search_datadog_logs` exactly that case.

Storage (ticket #7059): rows written before v1.95 hold the already-prefixed name, because
that is what the UI persisted then. PR #34559 changed the contract to bare names with no
migration, and the resolver uses the stored string as written, so those rows now name a tool
that does not exist upstream and every pre-upgrade toolset returns zero tools. Customers were
told to delete and re-add every tool by hand.

Both tests scope their assertions to the server_id they registered, so a concurrent run
reusing the same alias cannot cross-contaminate them.
"""

from __future__ import annotations

import pytest
from datadog_mcp import SEARCH_LOGS_TOOL, assert_dd_mcp_creds
from e2e_config import datadog_mcp_url, unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import KeyGenerateBody, ObjectPermission
from pydantic import BaseModel

pytestmark = pytest.mark.e2e

ALIAS_THAT_PREFIXES_THE_TOOL = "search_datadog"
"""Chosen so the server's native tool (`search_datadog_logs`) begins with the server's own
prefix, which is the shape that provoked the double-prefix and silent-rename reports."""


class ToolsetTool(BaseModel):
    server_id: str
    tool_name: str


class ToolsetCreateBody(BaseModel):
    toolset_name: str
    tools: list[ToolsetTool]


class Toolset(BaseModel):
    toolset_id: str


def _register_datadog_as(client: McpClient, resources: ResourceManager, *, alias: str) -> str:
    """Register the real Datadog MCP under a caller-chosen alias.

    `register_datadog_mcp` derives the alias from a unique marker, which is right
    everywhere else and wrong here: these tests need an alias that is a prefix of the
    server's native tool name.
    """
    import os

    assert_dd_mcp_creds()
    server_id = client.register_server(
        server_name=alias,
        alias=alias,
        url=datadog_mcp_url(toolsets="core"),
        transport="http",
        static_headers={
            "DD-API-KEY": os.environ["DD_API_KEY"].strip(),
            "DD-APPLICATION-KEY": os.environ["DD_APP_KEY"].strip(),
        },
        allowed_tools=[SEARCH_LOGS_TOOL],
    )
    resources.defer(lambda: client.delete_server(server_id))
    return server_id


def _create_toolset(
    client: McpClient, resources: ResourceManager, *, server_id: str, tool_name: str
) -> str:
    toolset = unwrap(
        client.proxy.transport.post(
            "/v1/mcp/toolset",
            headers=client.proxy.transport.master,
            json=ToolsetCreateBody(
                toolset_name=f"e2e-toolset-{unique_marker()}",
                tools=[ToolsetTool(server_id=server_id, tool_name=tool_name)],
            ),
            response_type=Toolset,
        )
    )
    resources.defer(
        lambda: client.proxy.transport.delete(
            f"/v1/mcp/toolset/{toolset.toolset_id}",
            headers=client.proxy.transport.master,
            json=NoBody(),
            response_type=NoBody,
        )
    )
    return toolset.toolset_id


def _key_scoped_to_toolset(client: McpClient, resources: ResourceManager, *, toolset_id: str) -> str:
    key = client.proxy.generate_key(
        KeyGenerateBody(
            user_id=f"e2e-toolset-prefix-{unique_marker()}",
            object_permission=ObjectPermission(mcp_toolsets=[toolset_id]),
        )
    )
    resources.defer(lambda: client.proxy.delete_key(key))
    return key


class TestToolsetToolNamePrefixing:
    @pytest.mark.covers("mcp.list_tools.api_key.prefixes_native_name_once")
    def test_native_name_starting_with_the_alias_is_prefixed_exactly_once(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        server_id = _register_datadog_as(client, resources, alias=ALIAS_THAT_PREFIXES_THE_TOOL)
        client.await_registered(server_id)
        toolset_id = _create_toolset(
            client, resources, server_id=server_id, tool_name=SEARCH_LOGS_TOOL
        )
        key = _key_scoped_to_toolset(client, resources, toolset_id=toolset_id)

        served = client.await_tool(key, server_id, SEARCH_LOGS_TOOL)

        assert served.endswith(SEARCH_LOGS_TOOL), (
            f"the native tool name must survive prefixing intact. Serving {served!r} for "
            f"{SEARCH_LOGS_TOOL!r} means the leading {ALIAS_THAT_PREFIXES_THE_TOOL!r} was mistaken "
            f"for a wire prefix and stripped, silently renaming a real tool"
        )
        assert served.count(SEARCH_LOGS_TOOL) == 1, (
            f"served name {served!r} repeats {SEARCH_LOGS_TOOL!r}, so the prefix was applied to an "
            f"already-prefixed name (the 'greyhound-greyhound_inspect_events' double-prefix report)"
        )
        prefix = served[: -len(SEARCH_LOGS_TOOL)]
        assert prefix[:-1] == ALIAS_THAT_PREFIXES_THE_TOOL and len(prefix) == len(
            ALIAS_THAT_PREFIXES_THE_TOOL
        ) + 1, (
            f"served name {served!r} must be the alias plus exactly one separator character "
            f"before the native tool name; got prefix {prefix!r} for alias "
            f"{ALIAS_THAT_PREFIXES_THE_TOOL!r}"
        )

    @pytest.mark.skip(
        reason="ticket #7059: open product bug. PR #34559 changed toolset rows to store bare tool "
        "names and shipped no migration, and resolve_toolset_tool_permissions uses the stored "
        "string as written, so a row written by <=1.94 names a tool that does not exist upstream "
        "and the toolset serves zero tools. Unskip when a migration or a read-path fallback lands."
    )
    @pytest.mark.covers("mcp.list_tools.api_key.legacy_prefixed_row_resolves")
    def test_a_pre_upgrade_prefixed_row_still_resolves(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        alias = f"e2e_dd_{unique_marker()}"
        server_id = _register_datadog_as(client, resources, alias=alias)
        client.await_registered(server_id)

        legacy_name = f"{alias}_{SEARCH_LOGS_TOOL}"
        toolset_id = _create_toolset(
            client, resources, server_id=server_id, tool_name=legacy_name
        )
        key = _key_scoped_to_toolset(client, resources, toolset_id=toolset_id)

        served = unwrap(client.list_tools(key)).tool_names_for_server(server_id)
        assert any(name.endswith(SEARCH_LOGS_TOOL) for name in served), (
            f"a toolset row holding the pre-v1.95 prefixed name {legacy_name!r} served no tools "
            f"(ticket #7059: every toolset created before the upgrade returned zero tools, and the "
            f"only remedy offered was deleting and re-adding each tool by hand). The read path must "
            f"still resolve legacy rows; got {sorted(served)}"
        )
