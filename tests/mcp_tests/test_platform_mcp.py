import json
import os
import sys
from typing import Optional

import pytest
from mcp.types import CallToolResult, TextContent, Tool

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.proxy._experimental.mcp_server import platform_mcp
from litellm.proxy._experimental.mcp_server import server as mcp_server_module
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"{name} description",
        inputSchema={"type": "object", "properties": {}},
    )


def _rich_tool(name: str) -> Tool:
    return Tool(
        name=name,
        title="Get Ticket",
        description=f"{name} description",
        inputSchema={"type": "object", "properties": {}},
        outputSchema={"type": "object", "properties": {"id": {"type": "string"}}},
        _meta={"source": "platform_mcp-test"},
    )


def _server(
    *,
    server_id: str = "server-1",
    name: str = "servicenow",
    alias: Optional[str] = None,
    description: str = "Service management",
) -> MCPServer:
    return MCPServer(
        server_id=server_id,
        name=name,
        alias=alias,
        server_name=name,
        url="https://example.com/mcp",
        transport=MCPTransport.http,
        mcp_info={"description": description},
    )


async def _merge_toolset_permissions(user_api_key_auth):
    return user_api_key_auth


async def _enabled_platform_mcp() -> bool:
    return True


async def _disabled_platform_mcp() -> bool:
    return False


@pytest.mark.asyncio
async def test_platform_mcp_disabled_returns_normal_tools(monkeypatch):
    normal_tools = [_tool(f"tool_{idx}") for idx in range(11)]

    async def fake_get_tools(**kwargs):
        return normal_tools

    monkeypatch.setattr(
        mcp_server_module,
        "_merge_toolset_permissions",
        _merge_toolset_permissions,
    )
    monkeypatch.setattr(
        mcp_server_module,
        "_get_tools_from_mcp_servers",
        fake_get_tools,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_platform_mcp_enabled",
        _disabled_platform_mcp,
    )

    tools = await mcp_server_module._list_mcp_tools()

    assert tools == normal_tools


def test_platform_mcp_advertises_tool_list_changed_capability():
    options = mcp_server_module.server.create_initialization_options()

    assert options.capabilities.tools is not None
    assert options.capabilities.tools.listChanged is True


@pytest.mark.asyncio
async def test_platform_mcp_enabled_accepts_config_field_string_values(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {"platform_mcp_enabled": "true"})
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    assert await platform_mcp.get_platform_mcp_enabled() is True


@pytest.mark.asyncio
async def test_platform_mcp_enabled_defaults_to_false(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "prisma_client", None)

    assert await platform_mcp.get_platform_mcp_enabled() is False


@pytest.mark.asyncio
async def test_platform_mcp_enabled_does_not_change_aggregate_tool_list(monkeypatch):
    normal_tools = [_tool(f"tool_{idx}") for idx in range(11)]

    async def fake_get_tools(**kwargs):
        return normal_tools

    monkeypatch.setattr(
        mcp_server_module,
        "_merge_toolset_permissions",
        _merge_toolset_permissions,
    )
    monkeypatch.setattr(
        mcp_server_module,
        "_get_tools_from_mcp_servers",
        fake_get_tools,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_platform_mcp_enabled",
        _enabled_platform_mcp,
    )

    tools = await mcp_server_module._list_mcp_tools()

    assert tools == normal_tools


@pytest.mark.asyncio
async def test_platform_mcp_enabled_does_not_change_scoped_server_tools(monkeypatch):
    normal_tools = [_tool(f"tool_{idx}") for idx in range(11)]

    async def fake_get_tools(**kwargs):
        return normal_tools

    monkeypatch.setattr(
        mcp_server_module,
        "_merge_toolset_permissions",
        _merge_toolset_permissions,
    )
    monkeypatch.setattr(
        mcp_server_module,
        "_get_tools_from_mcp_servers",
        fake_get_tools,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_platform_mcp_enabled",
        _enabled_platform_mcp,
    )

    tools = await mcp_server_module._list_mcp_tools(mcp_servers=["servicenow"])

    assert tools == normal_tools


@pytest.mark.asyncio
async def test_platform_mcp_virtual_server_returns_only_platform_tools(
    monkeypatch,
):
    async def fake_get_allowed_mcp_servers(*args, **kwargs):
        return [platform_mcp.build_platform_mcp_server()]

    monkeypatch.setattr(
        mcp_server_module,
        "_merge_toolset_permissions",
        _merge_toolset_permissions,
    )
    monkeypatch.setattr(
        mcp_server_module,
        "_get_allowed_mcp_servers",
        fake_get_allowed_mcp_servers,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_platform_mcp_enabled",
        _enabled_platform_mcp,
    )

    tools = await mcp_server_module._list_mcp_tools(mcp_servers=["platform_mcp"])

    assert [tool.name for tool in tools] == [
        "platform_mcp-list_servers",
        "platform_mcp-get_server_tools",
        "platform_mcp-call_tool",
    ]


@pytest.mark.asyncio
async def test_platform_mcp_list_servers_returns_names_and_descriptions(monkeypatch):
    async def fake_get_allowed_mcp_servers(*args, **kwargs):
        return [
            platform_mcp.build_platform_mcp_server(),
            _server(name="servicenow"),
            _server(name="github", description="Code"),
        ]

    monkeypatch.setattr(
        mcp_server_module,
        "_get_allowed_mcp_servers",
        fake_get_allowed_mcp_servers,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_platform_mcp_enabled",
        _enabled_platform_mcp,
    )

    result = await mcp_server_module._handle_platform_mcp_tool_call(
        name="list_servers",
        arguments={},
        user_api_key_auth=UserAPIKeyAuth(api_key="test"),
        mcp_auth_header=None,
        mcp_servers=None,
        mcp_server_auth_headers=None,
        oauth2_headers=None,
        raw_headers=None,
    )

    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload == {
        "servers": [
            {"name": "github", "description": "Code"},
            {"name": "servicenow", "description": "Service management"},
        ]
    }


@pytest.mark.asyncio
async def test_platform_mcp_get_server_tools_returns_selected_server_tool_definitions(
    monkeypatch,
):
    class FakeSession:
        def __init__(self):
            self.tool_list_changed_count = 0

        async def send_tool_list_changed(self):
            self.tool_list_changed_count += 1

    platform_server = platform_mcp.build_platform_mcp_server()
    selected_server = _server(name="servicenow")
    selected_tools = [_rich_tool("servicenow_get_ticket")]
    normal_tools = [_tool("github_list_issues")]
    requested_servers = []

    async def fake_get_allowed_mcp_servers(*args, **kwargs):
        return [platform_server, selected_server]

    async def fake_get_tools(**kwargs):
        requested_servers.append(kwargs["mcp_servers"])
        if kwargs["mcp_servers"] == ["servicenow"]:
            return selected_tools
        return normal_tools

    monkeypatch.setattr(
        mcp_server_module,
        "_merge_toolset_permissions",
        _merge_toolset_permissions,
    )
    monkeypatch.setattr(
        mcp_server_module,
        "_get_allowed_mcp_servers",
        fake_get_allowed_mcp_servers,
    )
    monkeypatch.setattr(
        mcp_server_module,
        "_get_tools_from_mcp_servers",
        fake_get_tools,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_platform_mcp_enabled",
        _enabled_platform_mcp,
    )
    session = FakeSession()
    token = mcp_server_module.active_mcp_session_var.set(session)
    try:
        result = await mcp_server_module._handle_platform_mcp_tool_call(
            name="get_server_tools",
            arguments={"server_name": "servicenow"},
            user_api_key_auth=UserAPIKeyAuth(api_key="test"),
            mcp_auth_header=None,
            mcp_servers=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
        )
        tools = await mcp_server_module._list_mcp_tools()
    finally:
        mcp_server_module.active_mcp_session_var.reset(token)

    assert result.isError is False
    payload = json.loads(result.content[0].text)
    assert payload["server"] == {
        "name": "servicenow",
        "description": "Service management",
    }
    assert payload["tools"] == [
        {
            "name": "servicenow_get_ticket",
            "title": "Get Ticket",
            "description": "servicenow_get_ticket description",
            "inputSchema": {"type": "object", "properties": {}},
            "outputSchema": {
                "type": "object",
                "properties": {"id": {"type": "string"}},
            },
            "_meta": {"source": "platform_mcp-test"},
        }
    ]
    assert session.tool_list_changed_count == 0
    assert requested_servers == [["servicenow"], None]
    assert tools == normal_tools


@pytest.mark.asyncio
async def test_platform_mcp_call_tool_dispatches_to_selected_server(monkeypatch):
    platform_server = platform_mcp.build_platform_mcp_server()
    selected_server = _server(name="servicenow")
    auth = UserAPIKeyAuth(api_key="test")
    calls = []

    async def fake_get_allowed_mcp_servers(*args, **kwargs):
        return [platform_server, selected_server]

    async def fake_execute_mcp_tool(**kwargs):
        calls.append(kwargs)
        return CallToolResult(
            content=[TextContent(type="text", text="ticket result")],
            isError=False,
        )

    monkeypatch.setattr(
        mcp_server_module,
        "_get_allowed_mcp_servers",
        fake_get_allowed_mcp_servers,
    )
    monkeypatch.setattr(
        mcp_server_module,
        "execute_mcp_tool",
        fake_execute_mcp_tool,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_platform_mcp_enabled",
        _enabled_platform_mcp,
    )

    result = await mcp_server_module._handle_platform_mcp_tool_call(
        name="call_tool",
        arguments={
            "server_name": "servicenow",
            "tool_name": "servicenow_get_ticket",
            "arguments": {"ticket_id": "INC-1"},
        },
        user_api_key_auth=auth,
        mcp_auth_header=None,
        mcp_servers=["platform_mcp"],
        mcp_server_auth_headers=None,
        oauth2_headers=None,
        raw_headers=None,
    )

    assert result.isError is False
    assert result.content[0].text == "ticket result"
    assert calls == [
        {
            "name": "servicenow_get_ticket",
            "arguments": {"ticket_id": "INC-1"},
            "allowed_mcp_servers": [selected_server],
            "start_time": calls[0]["start_time"],
            "user_api_key_auth": auth,
            "mcp_auth_header": None,
            "mcp_server_auth_headers": None,
            "oauth2_headers": None,
            "raw_headers": None,
            "requested_server_id": selected_server.server_id,
        }
    ]


@pytest.mark.asyncio
async def test_platform_mcp_call_tool_rejects_unavailable_downstream_server(
    monkeypatch,
):
    platform_server = platform_mcp.build_platform_mcp_server()
    selected_server = _server(name="servicenow")
    calls = []

    async def fake_get_allowed_mcp_servers(*args, **kwargs):
        return [platform_server, selected_server]

    async def fake_execute_mcp_tool(**kwargs):
        calls.append(kwargs)
        return CallToolResult(content=[], isError=False)

    monkeypatch.setattr(
        mcp_server_module,
        "_get_allowed_mcp_servers",
        fake_get_allowed_mcp_servers,
    )
    monkeypatch.setattr(
        mcp_server_module,
        "execute_mcp_tool",
        fake_execute_mcp_tool,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_platform_mcp_enabled",
        _enabled_platform_mcp,
    )

    result = await mcp_server_module._handle_platform_mcp_tool_call(
        name="call_tool",
        arguments={
            "server_name": "github",
            "tool_name": "list_issues",
            "arguments": {},
        },
        user_api_key_auth=UserAPIKeyAuth(api_key="test"),
        mcp_auth_header=None,
        mcp_servers=["platform_mcp"],
        mcp_server_auth_headers=None,
        oauth2_headers=None,
        raw_headers=None,
    )

    assert result.isError is True
    assert "not available to this key" in result.content[0].text
    assert calls == []
