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

from typing import Final

import pytest
from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import unique_marker
from e2e_http import unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import ToolsetCreateBody, ToolsetTool

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
