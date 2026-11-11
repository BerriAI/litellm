from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from mcp.types import CallToolResult, TextContent

from litellm.proxy.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp import MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer


@pytest.mark.asyncio
async def test_call_regular_mcp_tool_forwards_client_meta_when_enabled() -> None:
    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-meta-1",
        name="meta_server",
        url="http://localhost:8090/mcp",
        transport=MCPTransport.http,
        forward_client_meta=True,
    )

    captured_params = []

    async def fake_call_tool(params, **kwargs):
        captured_params.append(params)
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)

    mock_client = MagicMock()
    mock_client.call_tool = AsyncMock(side_effect=fake_call_tool)

    with (
        patch.object(manager, "create_mcp_client", AsyncMock(return_value=mock_client)),
        patch.object(manager, "_upstream_session_for", AsyncMock(return_value=None)),
    ):
        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="echo_tool",
            arguments={"query": "hello"},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
            proxy_logging_obj=None,
            client_meta={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
        )

    assert len(captured_params) == 1
    meta_dump = (
        captured_params[0].meta.model_dump(exclude_none=True)
        if hasattr(captured_params[0].meta, "model_dump")
        else dict(captured_params[0].meta or {})
    )
    assert meta_dump.get("traceparent") == "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"


@pytest.mark.asyncio
async def test_call_regular_mcp_tool_omits_client_meta_by_default() -> None:
    manager = MCPServerManager()
    server = MCPServer(
        server_id="srv-meta-2",
        name="meta_server_default",
        url="http://localhost:8090/mcp",
        transport=MCPTransport.http,
    )

    captured_params = []

    async def fake_call_tool(params, **kwargs):
        captured_params.append(params)
        return CallToolResult(content=[TextContent(type="text", text="ok")], isError=False)

    mock_client = MagicMock()
    mock_client.call_tool = AsyncMock(side_effect=fake_call_tool)

    with (
        patch.object(manager, "create_mcp_client", AsyncMock(return_value=mock_client)),
        patch.object(manager, "_upstream_session_for", AsyncMock(return_value=None)),
    ):
        await manager._call_regular_mcp_tool(
            mcp_server=server,
            original_tool_name="echo_tool",
            arguments={"query": "hello"},
            tasks=[],
            mcp_auth_header=None,
            mcp_server_auth_headers=None,
            oauth2_headers=None,
            raw_headers=None,
            proxy_logging_obj=None,
            client_meta={"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"},
        )

    assert len(captured_params) == 1
    assert getattr(captured_params[0], "meta", None) is None
