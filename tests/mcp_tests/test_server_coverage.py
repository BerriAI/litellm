import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.proxy._experimental.mcp_server.server import (
    _get_prompts_from_mcp_servers,
    _get_resources_from_mcp_servers,
    _get_resource_templates_from_mcp_servers,
    _get_tools_from_mcp_servers,
)


@pytest.mark.asyncio
async def test_get_prompts_from_mcp_servers_coverage():
    server1 = MCPServer(
        server_id="test-1", name="test1", transport="stdio", url="http://test1"
    )
    server2 = MCPServer(
        server_id="test-2", name="test2", transport="stdio", url="http://test2"
    )
    mock_prompt = MagicMock()
    mock_prompt.name = "test_prompt"

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        return_value=[server1, server2],
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_prompts_from_server",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.side_effect = [[mock_prompt], Exception("Server error")]
            result = await _get_prompts_from_mcp_servers(
                user_api_key_auth=None,
                mcp_auth_header=None,
                mcp_servers=["test1", "test2"],
            )
            assert len(result) == 1
            assert result[0] == mock_prompt


@pytest.mark.asyncio
async def test_get_resources_from_mcp_servers_coverage():
    server1 = MCPServer(
        server_id="test-1", name="test1", transport="stdio", url="http://test1"
    )
    mock_resource = MagicMock()
    mock_resource.name = "test_resource"

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        return_value=[server1],
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_resources_from_server",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = [mock_resource]
            result = await _get_resources_from_mcp_servers(
                user_api_key_auth=None, mcp_auth_header=None, mcp_servers=["test1"]
            )
            assert len(result) == 1
            assert result[0] == mock_resource


@pytest.mark.asyncio
async def test_get_resource_templates_from_mcp_servers_coverage():
    server1 = MCPServer(
        server_id="test-1", name="test1", transport="stdio", url="http://test1"
    )
    mock_template = MagicMock()
    mock_template.name = "test_template"

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        return_value=[server1],
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager.get_resource_templates_from_server",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = [mock_template]
            result = await _get_resource_templates_from_mcp_servers(
                user_api_key_auth=None, mcp_auth_header=None, mcp_servers=["test1"]
            )
            assert len(result) == 1
            assert result[0] == mock_template


@pytest.mark.asyncio
async def test_get_tools_from_mcp_servers_coverage():
    server1 = MCPServer(
        server_id="test-1", name="test1", transport="stdio", url="http://test1"
    )
    mock_tool = MagicMock()
    mock_tool.name = "test_tool"

    with patch(
        "litellm.proxy._experimental.mcp_server.server._get_allowed_mcp_servers",
        return_value=[server1],
    ):
        with patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager._get_tools_from_server",
            new_callable=AsyncMock,
        ) as mock_get:
            mock_get.return_value = [mock_tool]
            # test with some tracking headers
            result = await _get_tools_from_mcp_servers(
                user_api_key_auth=None,
                mcp_auth_header=None,
                mcp_servers=["test1"],
                log_list_tools_to_spendlogs=True,
                litellm_trace_id="test-trace",
            )
            assert len(result) == 1
            assert result[0] == mock_tool


@pytest.mark.asyncio
async def test_handle_stale_mcp_session():
    """Test the handle_stale_mcp_session logic for handling missing session IDs on multiple workers."""
    from litellm.proxy._experimental.mcp_server.server import _handle_stale_mcp_session

    # 1. DELETE request for non-existent session
    scope_delete = {
        "headers": [(b"mcp-session-id", b"stale-session-123")],
        "method": "DELETE",
        "type": "http",
    }
    mock_mgr = MagicMock()
    mock_mgr._server_instances = {}
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    result_delete = await _handle_stale_mcp_session(
        scope_delete, mock_receive, mock_send, mock_mgr
    )
    assert result_delete is True
    # The JSONResponse success should have been sent
    assert mock_send.call_count >= 1

    # 2. POST request for non-existent session -> header should be stripped
    scope_post = {
        "headers": [
            (b"mcp-session-id", b"stale-session-123"),
            (b"content-type", b"application/json"),
        ],
        "method": "POST",
    }
    result_post = await _handle_stale_mcp_session(
        scope_post, mock_receive, mock_send, mock_mgr
    )
    assert result_post is False
    headers = dict(scope_post["headers"])
    assert b"mcp-session-id" not in headers
    assert b"content-type" in headers

    # 3. Request with valid session -> should return False immediately
    mock_mgr._server_instances = {"valid-session-123": MagicMock()}
    scope_valid = {
        "headers": [(b"mcp-session-id", b"valid-session-123")],
        "method": "POST",
    }
    result_valid = await _handle_stale_mcp_session(
        scope_valid, mock_receive, mock_send, mock_mgr
    )
    assert result_valid is False
    # Header should not be stripped
    headers_valid = dict(scope_valid["headers"])
    assert b"mcp-session-id" in headers_valid
