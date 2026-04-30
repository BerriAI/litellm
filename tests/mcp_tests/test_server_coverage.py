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


@pytest.mark.asyncio
async def test_handle_sse_post_messages_auth_failure():
    """Test the POST-auth-discard behavior when extraction fails."""
    from litellm.proxy._experimental.mcp_server.server import handle_sse_post_messages

    scope = {"type": "http", "path": "/messages"}
    mock_receive = AsyncMock()
    mock_send = AsyncMock()

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new_callable=AsyncMock,
    ) as mock_extract:
        # Simulate extraction failure
        mock_extract.side_effect = Exception("Extraction failed")

        await handle_sse_post_messages(scope, mock_receive, mock_send)

        # Verify JSONResponse 500 was sent
        assert mock_send.call_count >= 1
        # Extract response from mock_send
        response_body = b"".join(
            [
                call.args[0].get("body", b"")
                for call in mock_send.mock_calls
                if call.args[0].get("type") == "http.response.body"
            ]
        )
        assert b"Authentication processing failed" in response_body


def test_get_active_auth_context_session_storage_fallback():
    """Test the _session_auth_storage fallback in get_active_auth_context."""
    from litellm.proxy._experimental.mcp_server.server import (
        get_active_auth_context,
        _session_auth_storage,
    )

    class MockRequestCtx:
        def __init__(self, stream):
            self.session = MagicMock()
            self.session._read_stream = stream

    mock_stream = MagicMock()
    mock_auth = MagicMock()

    with patch(
        "litellm.proxy._experimental.mcp_server.server.auth_context_var"
    ) as mock_var:
        mock_var.get.return_value = None  # Force fallback
        with patch("mcp.server.lowlevel.server.request_ctx") as mock_req_ctx:
            mock_req_ctx.get.return_value = MockRequestCtx(mock_stream)

            # 1. Test when missing from storage
            assert get_active_auth_context() is None

            # 2. Test when present in storage
            _session_auth_storage[mock_stream] = mock_auth
            assert get_active_auth_context() == mock_auth

            # Cleanup
            del _session_auth_storage[mock_stream]
