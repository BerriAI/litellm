import json
import os
import sys
from typing import Optional

import pytest
from mcp.types import Tool

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
        _meta={"source": "platform-mcp-test"},
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


async def _enabled_platform_settings() -> tuple[bool, int]:
    return True, 10


async def _disabled_platform_settings() -> tuple[bool, int]:
    return False, 10


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
        "get_platform_mcp_settings",
        _disabled_platform_settings,
    )

    tools = await mcp_server_module._list_mcp_tools()

    assert tools == normal_tools


def test_platform_mcp_advertises_tool_list_changed_capability():
    options = mcp_server_module.server.create_initialization_options()

    assert options.capabilities.tools is not None
    assert options.capabilities.tools.listChanged is True


@pytest.mark.asyncio
async def test_platform_mcp_compresses_aggregate_tools_over_threshold(monkeypatch):
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
        "get_platform_mcp_settings",
        _enabled_platform_settings,
    )

    tools = await mcp_server_module._list_mcp_tools()

    assert [tool.name for tool in tools] == ["list_servers", "enable_server"]


@pytest.mark.asyncio
async def test_platform_mcp_does_not_compress_scoped_server_tools(monkeypatch):
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
        "get_platform_mcp_settings",
        _enabled_platform_settings,
    )

    tools = await mcp_server_module._list_mcp_tools(mcp_servers=["servicenow"])

    assert tools == normal_tools


@pytest.mark.asyncio
async def test_platform_mcp_enabled_session_returns_meta_tools_and_enabled_server_tools(
    monkeypatch,
):
    selected_tools = [_tool("servicenow_get_ticket")]

    async def fake_get_tools(**kwargs):
        assert kwargs["mcp_servers"] == ["servicenow"]
        return selected_tools

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
        "get_platform_mcp_settings",
        _enabled_platform_settings,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_enabled_server_names_for_session",
        lambda _session: ("servicenow",),
    )

    tools = await mcp_server_module._list_mcp_tools()

    assert [tool.name for tool in tools] == [
        "list_servers",
        "enable_server",
        "servicenow_get_ticket",
    ]


@pytest.mark.asyncio
async def test_platform_mcp_list_servers_returns_names_and_descriptions(monkeypatch):
    async def fake_get_allowed_mcp_servers(*args, **kwargs):
        return [_server(name="servicenow"), _server(name="github", description="Code")]

    monkeypatch.setattr(
        mcp_server_module,
        "_get_allowed_mcp_servers",
        fake_get_allowed_mcp_servers,
    )
    monkeypatch.setattr(
        platform_mcp,
        "get_platform_mcp_settings",
        _enabled_platform_settings,
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
async def test_platform_mcp_enable_server_returns_selected_server_tool_definitions(
    monkeypatch,
):
    selected_server = _server(name="servicenow")
    selected_tools = [_rich_tool("servicenow_get_ticket")]

    async def fake_get_allowed_mcp_servers(*args, **kwargs):
        return [selected_server]

    async def fake_get_tools(**kwargs):
        assert kwargs["mcp_servers"] == ["servicenow"]
        return selected_tools

    async def fake_send_tool_list_changed(_session):
        return None

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
        "get_platform_mcp_settings",
        _enabled_platform_settings,
    )
    monkeypatch.setattr(
        platform_mcp,
        "enable_server_for_session",
        lambda _session, _server: None,
    )
    monkeypatch.setattr(
        mcp_server_module,
        "_send_platform_mcp_tool_list_changed",
        fake_send_tool_list_changed,
    )

    result = await mcp_server_module._handle_platform_mcp_tool_call(
        name="enable_server",
        arguments={"server_name": "servicenow"},
        user_api_key_auth=UserAPIKeyAuth(api_key="test"),
        mcp_auth_header=None,
        mcp_servers=None,
        mcp_server_auth_headers=None,
        oauth2_headers=None,
        raw_headers=None,
    )

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
            "_meta": {"source": "platform-mcp-test"},
        }
    ]


@pytest.mark.asyncio
async def test_platform_mcp_enable_server_updates_same_session_list_tools(monkeypatch):
    class FakeSession:
        def __init__(self):
            self.tool_list_changed_count = 0

        async def send_tool_list_changed(self):
            self.tool_list_changed_count += 1

    selected_server = _server(name="servicenow")
    selected_tools = [_tool("servicenow_get_ticket")]
    requested_servers = []

    async def fake_get_allowed_mcp_servers(*args, **kwargs):
        return [selected_server]

    async def fake_get_tools(**kwargs):
        requested_servers.append(kwargs["mcp_servers"])
        return selected_tools

    session = FakeSession()

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
        "get_platform_mcp_settings",
        _enabled_platform_settings,
    )
    platform_mcp._enabled_servers_by_session.clear()
    token = mcp_server_module.active_mcp_session_var.set(session)
    try:
        enable_result = await mcp_server_module._handle_platform_mcp_tool_call(
            name="enable_server",
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
        platform_mcp._enabled_servers_by_session.clear()

    assert enable_result.isError is False
    assert session.tool_list_changed_count == 1
    assert requested_servers == [["servicenow"], ["servicenow"]]
    assert [tool.name for tool in tools] == [
        "list_servers",
        "enable_server",
        "servicenow_get_ticket",
    ]
