import json
from collections.abc import Iterator
from unittest.mock import AsyncMock, patch

import pytest
from mcp.types import CallToolResult, TextContent, Tool

from litellm.proxy._experimental.mcp_server import server
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import AggregateToolListing
from litellm.proxy._experimental.mcp_server.mcp_server_manager import global_mcp_server_manager
from litellm.proxy._experimental.mcp_server.tool_search import (
    MCP_PROXY_CALL_TOOL_NAME,
    MCP_PROXY_SCHEMA_TOOL_NAME,
    MCP_PROXY_SEARCH_TOOL_NAME,
    handle_mcp_proxy_tool,
    mcp_proxy_tool_id,
    with_mcp_proxy_identity,
)
from litellm.proxy._types import LiteLLM_ObjectPermissionTable, UserAPIKeyAuth
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

TOOL = Tool.model_validate(
    {
        "name": "math_stdio-add",
        "description": "Add two numbers",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        "outputSchema": {"type": "object"},
        "_meta": {"litellm.ai/proxy_tool_identity": {"server_id": "server-1", "tool_name": "math_stdio-add"}},
    }
)
AUTH = UserAPIKeyAuth(api_key="key")


def _text(result: CallToolResult) -> object:
    return json.loads(result.content[0].text)


@pytest.mark.asyncio
async def test_proxy_search_returns_opaque_id_and_schema() -> None:
    with (
        patch(  # test-quality-ok: isolate authorized catalog owner
            "litellm.proxy._experimental.mcp_server.server._list_mcp_tools",
            new_callable=AsyncMock,
            return_value=AggregateToolListing(tools=[TOOL], outcomes={}),
        ),
    ):
        result = await handle_mcp_proxy_tool(MCP_PROXY_SEARCH_TOOL_NAME, {"query": "add"}, AUTH)

    item = _text(result)[0]
    assert item["tool_id"] == mcp_proxy_tool_id(TOOL)
    assert item["name"] == TOOL.name
    assert "inputSchema" not in item
    assert "outputSchema" not in item
    assert len(item["tool_id"]) == 32


@pytest.mark.asyncio
async def test_proxy_schema_and_call_resolve_current_authorized_catalog() -> None:
    executed = CallToolResult(content=[TextContent(type="text", text="3")], isError=False)
    with (
        patch(  # test-quality-ok: isolate authorized catalog owner
            "litellm.proxy._experimental.mcp_server.server._list_mcp_tools",
            new_callable=AsyncMock,
            return_value=AggregateToolListing(tools=[TOOL], outcomes={}),
        ),
        patch(  # test-quality-ok: isolate execution delegate seam
            "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_call",
            new_callable=AsyncMock,
            return_value=executed,
        ) as call,
    ):
        schema = await handle_mcp_proxy_tool(
            MCP_PROXY_SCHEMA_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL)},
            AUTH,
        )
        result = await handle_mcp_proxy_tool(
            MCP_PROXY_CALL_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL), "arguments": {"a": 1, "b": 2}},
            AUTH,
        )

    assert _text(schema)["inputSchema"] == TOOL.inputSchema
    assert result is executed
    assert call.await_args.kwargs["tool_name"] == TOOL.name
    assert call.await_args.kwargs["arguments"] == {"a": 1, "b": 2}
    assert call.await_args.kwargs["requested_server_id"] == "server-1"


@pytest.mark.asyncio
async def test_proxy_rejects_stale_id_and_invalid_arguments_before_dispatch() -> None:
    with (
        patch(  # test-quality-ok: isolate authorized catalog owner
            "litellm.proxy._experimental.mcp_server.server._list_mcp_tools",
            new_callable=AsyncMock,
            return_value=AggregateToolListing(tools=[TOOL], outcomes={}),
        ),
        patch(  # test-quality-ok: isolate execution delegate seam
            "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_tool_call", new_callable=AsyncMock
        ) as call,
    ):
        stale = await handle_mcp_proxy_tool(MCP_PROXY_SCHEMA_TOOL_NAME, {"tool_id": "stale"}, AUTH)
        invalid = await handle_mcp_proxy_tool(
            MCP_PROXY_CALL_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL), "arguments": "wrong"},
            AUTH,
        )
        falsy = await handle_mcp_proxy_tool(
            MCP_PROXY_CALL_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL), "arguments": False},
            AUTH,
        )
        invalid_schema = await handle_mcp_proxy_tool(
            MCP_PROXY_CALL_TOOL_NAME,
            {"tool_id": mcp_proxy_tool_id(TOOL), "arguments": {"a": "wrong"}},
            AUTH,
        )

    assert stale.isError is True
    assert invalid.isError is True
    assert falsy.isError is True
    assert invalid_schema.isError is True
    call.assert_not_awaited()


@pytest.mark.asyncio
async def test_proxy_call_builds_logging_object() -> None:
    from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode

    sentinel = object()
    result = CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
    token = _mcp_proxy_mode.set(True)
    try:
        with (
            patch.object(  # test-quality-ok: isolate logging pipeline seam
                server, "_build_virtual_call_logging_obj", new_callable=AsyncMock, return_value=sentinel
            ) as build,
            patch(  # test-quality-ok: isolate proxy dispatch seam
                "litellm.proxy._experimental.mcp_server.tool_search.handle_mcp_proxy_tool",
                new_callable=AsyncMock,
                return_value=result,
            ) as handle,
        ):
            actual = await server._dispatch_virtual_mcp_tool(
                name=MCP_PROXY_CALL_TOOL_NAME,
                arguments={"tool_id": "id", "arguments": {}},
                user_api_key_auth=AUTH,
                client_ip=None,
            )
    finally:
        _mcp_proxy_mode.reset(token)

    assert actual is result
    build.assert_awaited_once()
    assert handle.await_args.kwargs["litellm_logging_obj"] is sentinel


@pytest.mark.asyncio
async def test_proxy_call_rejects_non_proxy_tool_names() -> None:
    from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode

    token = _mcp_proxy_mode.set(True)
    try:
        result = await server._dispatch_virtual_mcp_tool(
            name="math_stdio-add",
            arguments={"a": 1, "b": 2},
            user_api_key_auth=AUTH,
            client_ip=None,
        )
    finally:
        _mcp_proxy_mode.reset(token)

    assert result is not None
    assert result.isError is True
    assert "unavailable" in result.content[0].text


@pytest.mark.asyncio
async def test_proxy_rejects_non_tool_protocol_operations() -> None:
    from mcp.shared.exceptions import McpError
    from pydantic import AnyUrl

    from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode

    token = _mcp_proxy_mode.set(True)
    try:
        with pytest.raises(McpError):
            await server.list_prompts()
        with pytest.raises(McpError):
            await server.get_prompt("prompt", {})
        with pytest.raises(McpError):
            await server.list_resources()
        with pytest.raises(McpError):
            await server.list_resource_templates()
        with pytest.raises(McpError):
            await server.read_resource(AnyUrl("https://example.com/resource"))
    finally:
        _mcp_proxy_mode.reset(token)


@pytest.mark.asyncio
async def test_proxy_list_mode_has_fixed_definitions_without_search_flag() -> None:
    from litellm.proxy._experimental.mcp_server.mcp_context import _mcp_proxy_mode

    token = _mcp_proxy_mode.set(True)
    try:
        with patch(  # test-quality-ok: isolate authenticated MCP context seam
            "litellm.proxy._experimental.mcp_server.server.get_or_extract_auth_context",
            new_callable=AsyncMock,
            return_value=(AUTH, None, None, None, None, None, None),
        ):
            tools = await server.handle_list_tools()
            options = server.server.create_initialization_options()
    finally:
        _mcp_proxy_mode.reset(token)

    assert {tool.name for tool in tools} == {
        MCP_PROXY_SEARCH_TOOL_NAME,
        MCP_PROXY_SCHEMA_TOOL_NAME,
        MCP_PROXY_CALL_TOOL_NAME,
    }
    assert options.capabilities.prompts is None
    assert options.capabilities.resources is None
    assert options.capabilities.tools is not None


def _server(server_id: str, name: str, **overrides: object) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=name,
        server_name=name,
        url=f"http://{name}.test",
        transport=MCPTransport.http,
        **overrides,
    )


def _upstream_tool(prefix: str, name: str) -> Tool:
    return Tool(
        name=f"{prefix}-{name}",
        description=f"{name} numbers",
        inputSchema={"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]},
    )


def _auth(**object_permission: object) -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        api_key="sk-scope",
        object_permission=LiteLLM_ObjectPermissionTable(object_permission_id="scope", **object_permission),
    )


def _ids(result: CallToolResult) -> dict[str, str]:
    return {item["name"]: item["tool_id"] for item in _text(result)}


class TestMcpProxyAuthorizationScope:
    """The real catalog resolver runs (server grants, tool grants, scope header, sentinel); only the
    upstream tools/list fetch and the final upstream dispatch are faked."""

    ALPHA = _server("srv-alpha", "alpha", tool_name_to_display_name={"add": "Add Numbers"})
    BETA = _server("srv-beta", "beta")
    ALPHA_ADD = with_mcp_proxy_identity(_upstream_tool("alpha", "add"), "srv-alpha")
    ALPHA_MULTIPLY = with_mcp_proxy_identity(_upstream_tool("alpha", "multiply"), "srv-alpha")
    BETA_ADD = with_mcp_proxy_identity(_upstream_tool("beta", "add"), "srv-beta")

    @pytest.fixture
    def rig(self) -> Iterator[AsyncMock]:
        upstream = {
            "srv-alpha": [_upstream_tool("alpha", "add"), _upstream_tool("alpha", "multiply")],
            "srv-beta": [_upstream_tool("beta", "add")],
        }

        async def fetch(server: MCPServer, **_: object) -> list[Tool]:
            return list(upstream[server.server_id])

        dispatched = AsyncMock(
            return_value=CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)
        )
        global_mcp_server_manager.registry.update({"srv-alpha": self.ALPHA, "srv-beta": self.BETA})
        with (
            patch.object(  # test-quality-ok: the upstream MCP server is the only faked collaborator
                global_mcp_server_manager, "_get_tools_from_server", new=AsyncMock(side_effect=fetch)
            ),
            patch.object(global_mcp_server_manager, "call_tool", new=dispatched),  # test-quality-ok: dispatch seam
        ):
            yield dispatched

    async def _proxy(
        self, name: str, arguments: dict[str, object], auth: UserAPIKeyAuth, **kwargs: object
    ) -> CallToolResult:
        return await handle_mcp_proxy_tool(name, arguments, auth, **kwargs)

    @pytest.mark.asyncio
    async def test_search_and_schema_are_bounded_by_the_key_server_grant(self, rig: AsyncMock) -> None:
        granted = _auth(mcp_servers=["srv-alpha"])

        assert _ids(await self._proxy(MCP_PROXY_SEARCH_TOOL_NAME, {"query": "numbers"}, granted)) == {
            "alpha-add": mcp_proxy_tool_id(self.ALPHA_ADD),
            "alpha-multiply": mcp_proxy_tool_id(self.ALPHA_MULTIPLY),
        }
        denied_schema = await self._proxy(
            MCP_PROXY_SCHEMA_TOOL_NAME, {"tool_id": mcp_proxy_tool_id(self.BETA_ADD)}, granted
        )
        denied_call = await self._proxy(
            MCP_PROXY_CALL_TOOL_NAME, {"tool_id": mcp_proxy_tool_id(self.BETA_ADD), "arguments": {"a": 1}}, granted
        )
        assert denied_schema.isError is True and denied_call.isError is True
        rig.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_mcp_servers_sentinel_hides_every_tool(self, rig: AsyncMock) -> None:
        result = await self._proxy(
            MCP_PROXY_SEARCH_TOOL_NAME, {"query": "numbers"}, _auth(mcp_servers=["no-mcp-servers"])
        )
        assert _text(result) == []
        rig.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tool_grant_hides_ungranted_tools_and_blocks_their_ids(self, rig: AsyncMock) -> None:
        scoped = _auth(mcp_servers=["srv-alpha"], mcp_tool_permissions={"srv-alpha": ["add"]})

        assert set(_ids(await self._proxy(MCP_PROXY_SEARCH_TOOL_NAME, {"query": "numbers"}, scoped))) == {"alpha-add"}
        blocked = await self._proxy(
            MCP_PROXY_CALL_TOOL_NAME, {"tool_id": mcp_proxy_tool_id(self.ALPHA_MULTIPLY), "arguments": {"a": 1}}, scoped
        )
        assert blocked.isError is True
        rig.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_same_named_tools_keep_distinct_ids_and_dispatch_to_their_own_server(self, rig: AsyncMock) -> None:
        both = _auth(mcp_servers=["srv-alpha", "srv-beta"])

        ids = _ids(await self._proxy(MCP_PROXY_SEARCH_TOOL_NAME, {"query": "add"}, both))
        assert set(ids) == {"alpha-add", "beta-add"}, "display-name overrides must not rename proxy identities"
        assert ids["alpha-add"] != ids["beta-add"]

        result = await self._proxy(MCP_PROXY_CALL_TOOL_NAME, {"tool_id": ids["beta-add"], "arguments": {"a": 1}}, both)
        assert result.isError is False
        rig.assert_awaited_once()
        assert rig.await_args.kwargs["server_name"] == "beta"
        assert rig.await_args.kwargs["name"] == "add"

    @pytest.mark.asyncio
    async def test_server_scope_header_narrows_search_within_the_grant(self, rig: AsyncMock) -> None:
        both = _auth(mcp_servers=["srv-alpha", "srv-beta"])
        scoped = await self._proxy(MCP_PROXY_SEARCH_TOOL_NAME, {"query": "add"}, both, mcp_servers=["beta"])
        assert set(_ids(scoped)) == {"beta-add"}
        rig.assert_not_awaited()
