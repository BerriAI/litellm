"""Live e2e: MCP toolset persistence and the tool-name prefix contract (LIT-3419).

A toolset row stores bare `{server_id, tool_name}` pairs; the gateway applies the
`{server_prefix}{separator}{tool}` wire prefix only at serve time. A regression in
either direction broke customers twice: the UI once persisted pre-prefixed names
(so serve-time prefixing double-prefixed them and tools/list stopped matching), and
once the served name lost the prefix entirely (so clients calling the documented
`{server}-{tool}` name got "tool not found"). These tests pin both halves against
the real Datadog remote MCP server:

- the stored toolset row keeps exactly the bare tool name it was created with
- /v1/mcp/tools serves that same tool under the prefixed wire name for a key
  scoped to the toolset via object_permission.mcp_toolsets
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from datadog_mcp import SEARCH_LOGS_TOOL, register_datadog_mcp
from e2e_config import unique_marker
from e2e_http import NoBody, unwrap
from lifecycle import ResourceManager
from mcp_client import McpClient
from models import KeyGenerateBody, ObjectPermission

pytestmark = pytest.mark.e2e


class ToolsetTool(BaseModel):
    server_id: str
    tool_name: str


class ToolsetCreateBody(BaseModel):
    toolset_name: str
    description: str | None = None
    tools: list[ToolsetTool]


class Toolset(BaseModel):
    toolset_id: str
    toolset_name: str
    tools: list[ToolsetTool] = []


class PrefixedTool(BaseModel):
    name: str


class PrefixedToolsResponse(BaseModel):
    tools: list[PrefixedTool] = []


def _delete_toolset(client: McpClient, toolset_id: str) -> None:
    _ = client.proxy.transport.delete(
        f"/v1/mcp/toolset/{toolset_id}",
        headers=client.proxy.transport.master,
        json=NoBody(),
        response_type=NoBody,
    )


def _create_toolset(client: McpClient, resources: ResourceManager, *, server_id: str) -> Toolset:
    toolset = unwrap(
        client.proxy.transport.post(
            "/v1/mcp/toolset",
            headers=client.proxy.transport.master,
            json=ToolsetCreateBody(
                toolset_name=f"e2e-toolset-{unique_marker()}",
                tools=[ToolsetTool(server_id=server_id, tool_name=SEARCH_LOGS_TOOL)],
            ),
            response_type=Toolset,
        )
    )
    resources.defer(lambda: _delete_toolset(client, toolset.toolset_id))
    return toolset


class TestMcpToolsetToolNames:
    @pytest.mark.covers("mcp.toolset.api_key.persists_bare_tool_names")
    def test_stored_toolset_keeps_bare_tool_name(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        server_id = register_datadog_mcp(client, resources)
        client.await_registered(server_id)

        created = _create_toolset(client, resources, server_id=server_id)

        stored = unwrap(
            client.proxy.transport.get(
                f"/v1/mcp/toolset/{created.toolset_id}",
                headers=client.proxy.transport.master,
                params=NoBody(),
                response_type=Toolset,
            )
        )
        assert stored.tools == [ToolsetTool(server_id=server_id, tool_name=SEARCH_LOGS_TOOL)], (
            f"toolset row must store the bare tool name paired with its server_id; a "
            f"pre-prefixed or rewritten name breaks serve-time prefixing: {stored.tools}"
        )

    @pytest.mark.covers("mcp.toolset.api_key.serves_prefixed_tool_names")
    def test_toolset_scoped_key_sees_prefixed_tool_name(
        self, client: McpClient, resources: ResourceManager
    ) -> None:
        server_id = register_datadog_mcp(client, resources)
        client.await_registered(server_id)
        toolset = _create_toolset(client, resources, server_id=server_id)

        key = client.proxy.generate_key(
            KeyGenerateBody(
                user_id=f"e2e-toolset-{unique_marker()}",
                object_permission=ObjectPermission(mcp_toolsets=[toolset.toolset_id]),
            )
        )
        resources.defer(lambda: client.proxy.delete_key(key))

        served = unwrap(
            client.proxy.transport.get(
                "/v1/mcp/tools",
                headers=client.proxy.transport.bearer(key),
                params=NoBody(),
                response_type=PrefixedToolsResponse,
            )
        )
        served_names = [tool.name for tool in served.tools]
        alias = next(
            (row.alias for row in client.registered_servers() if row.server_id == server_id),
            None,
        )
        assert alias is not None, f"registered server {server_id} absent from /v1/mcp/server listing"
        expected_name = f"{alias}-{SEARCH_LOGS_TOOL}"
        assert expected_name in served_names, (
            f"a toolset-scoped key must be served {SEARCH_LOGS_TOOL} under its exact documented "
            f"wire name {expected_name!r} ('<alias>-<tool>'). A wrong prefix or separator still "
            f"ends in {SEARCH_LOGS_TOOL} but breaks every client calling the documented name with "
            f"'tool not found'; got {served_names}"
        )
