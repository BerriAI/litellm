from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

@pytest.mark.asyncio
async def test_list_tool_rest_api_raises_http_exception_on_server_failure():
    from litellm.proxy._experimental.mcp_server.rest_endpoints import list_tool_rest_api
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )

    mock_request = MagicMock()
    mock_request.headers = {}
    mock_user_api_key_dict = MagicMock()

    with patch.object(
        MCPRequestHandler, "_get_mcp_auth_header_from_headers"
    ) as mock_get_auth:
        with patch.object(
            MCPRequestHandler, "_get_mcp_server_auth_headers_from_headers"
        ) as mock_get_server_auth:
            mock_get_auth.return_value = None
            mock_get_server_auth.return_value = {}

            with patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints.global_mcp_server_manager"
            ) as mock_manager:
                mock_server = MagicMock()
                mock_server.name = "test-server"
                mock_server.alias = "test-server"
                mock_server.server_name = "test-server"
                mock_server.mcp_info = {"server_name": "test-server"}

                mock_manager.get_mcp_server_by_id.return_value = mock_server

                failing_get_tools = AsyncMock(side_effect=Exception("backend failure"))

                with patch(
                    "litellm.proxy._experimental.mcp_server.rest_endpoints._get_tools_for_single_server",
                    failing_get_tools,
                ):
                    with pytest.raises(HTTPException) as exc_info:
                        await list_tool_rest_api(
                            request=mock_request,
                            server_id="test-server",
                            user_api_key_dict=mock_user_api_key_dict,
                        )

    exc = exc_info.value
    assert exc.status_code == 500
    assert exc.detail["error"] == "server_error"
    assert exc.detail["tools"] == []
    assert "An unexpected error occurred" in exc.detail["message"]


@pytest.mark.asyncio
async def test_list_tool_rest_api_returns_tools_successfully():
    from litellm.proxy._experimental.mcp_server.rest_endpoints import list_tool_rest_api
    from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import (
        MCPRequestHandler,
    )
    from litellm.proxy._experimental.mcp_server.server import (
        ListMCPToolsRestAPIResponseObject,
    )

    mock_request = MagicMock()
    mock_request.headers = {
        "authorization": "Bearer user_token",
        "x-mcp-authorization": "Bearer default_token",
    }
    mock_user_api_key_dict = MagicMock()

    with patch.object(
        MCPRequestHandler, "_get_mcp_auth_header_from_headers"
    ) as mock_get_auth:
        with patch.object(
            MCPRequestHandler, "_get_mcp_server_auth_headers_from_headers"
        ) as mock_get_server_auth:
            mock_get_auth.return_value = "Bearer default_token"
            mock_get_server_auth.return_value = {}

            with patch(
                "litellm.proxy._experimental.mcp_server.rest_endpoints.global_mcp_server_manager"
            ) as mock_manager:
                mock_server = MagicMock()
                mock_server.name = "test-server"
                mock_server.alias = "test-server"
                mock_server.server_name = "test-server"
                mock_server.mcp_info = {"server_name": "test-server"}

                mock_manager.get_mcp_server_by_id.return_value = mock_server

                tool = ListMCPToolsRestAPIResponseObject(
                    name="send_email",
                    description="Send an email",
                    inputSchema={"type": "object"},
                    mcp_info={"server_name": "test-server"},
                )

                with patch(
                    "litellm.proxy._experimental.mcp_server.rest_endpoints._get_tools_for_single_server",
                    AsyncMock(return_value=[tool]),
                ) as mock_get_tools:
                    result = await list_tool_rest_api(
                        request=mock_request,
                        server_id="test-server",
                        user_api_key_dict=mock_user_api_key_dict,
                    )

    assert result["error"] is None
    assert result["message"] == "Successfully retrieved tools"
    assert len(result["tools"]) == 1
    assert result["tools"][0].name == "send_email"
    mock_get_tools.assert_awaited_once()
