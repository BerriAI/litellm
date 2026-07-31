"""
Unit tests for the MCPClient class - critical functionality only.
"""

import base64
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, ANY

# Add the project root to the path
sys.path.insert(0, os.path.abspath("../../.."))

import litellm.experimental_mcp_client.client as mcp_client_module
from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPAuth, MCPTransport
from mcp.types import CallToolResult as MCPCallToolResult
from mcp.types import ListToolsResult, PaginatedRequestParams
from mcp.types import Tool as MCPTool


def test_mcp_client_uses_configurable_default_timeout():
    """MCPClient should use MCP_CLIENT_TIMEOUT constant when no timeout is passed."""
    with patch("litellm.experimental_mcp_client.client.MCP_CLIENT_TIMEOUT", 120.0):
        # Client reads constant at runtime when timeout is None
        client = MCPClient(
            server_url="http://example.com",
            transport_type=MCPTransport.sse,
        )
        assert client.timeout == 120.0


class TestMCPClientUnitTests:
    """Unit tests for MCPClient functionality."""

    def test_init_with_auth(self):
        """Test initialization with authentication."""
        client = MCPClient(
            server_url="http://example.com",
            transport_type=MCPTransport.sse,
            auth_type=MCPAuth.bearer_token,
            auth_value="test_token",
            timeout=30.0,
        )
        assert client.server_url == "http://example.com"
        assert client.transport_type == MCPTransport.sse
        assert client.auth_type == MCPAuth.bearer_token
        assert client.timeout == 30.0
        assert client._mcp_auth_value == "test_token"

    def test_get_auth_headers(self):
        """Test authentication header generation for different auth types."""
        # Bearer token
        client = MCPClient(
            "http://example.com",
            auth_type=MCPAuth.bearer_token,
            auth_value="test_token",
        )
        headers = client._get_auth_headers()
        assert headers == {
            "Authorization": "Bearer test_token",
        }

        # Basic auth
        client = MCPClient(
            "http://example.com", auth_type=MCPAuth.basic, auth_value="user:pass"
        )
        expected_encoded = base64.b64encode("user:pass".encode("utf-8")).decode()
        headers = client._get_auth_headers()
        assert headers == {
            "Authorization": f"Basic {expected_encoded}",
        }

        # API key
        client = MCPClient(
            "http://example.com", auth_type=MCPAuth.api_key, auth_value="api_key_123"
        )
        headers = client._get_auth_headers()
        assert headers == {
            "X-API-Key": "api_key_123",
        }

        # Custom authorization header
        client = MCPClient(
            "http://example.com",
            auth_type=MCPAuth.authorization,
            auth_value="Token custom_token",
        )
        headers = client._get_auth_headers()
        assert headers == {
            "Authorization": "Token custom_token",
        }

        # OAuth2
        client = MCPClient(
            "http://example.com",
            auth_type=MCPAuth.oauth2,
            auth_value="oauth2-access-token-xyz",
        )
        headers = client._get_auth_headers()
        assert headers == {
            "Authorization": "Bearer oauth2-access-token-xyz",
        }

        # OAuth2 with extra_headers (per-user flow overrides auth_value)
        client = MCPClient(
            "http://example.com",
            auth_type=MCPAuth.oauth2,
            auth_value="static-server-token",
            extra_headers={"Authorization": "Bearer per-user-token"},
        )
        headers = client._get_auth_headers()
        assert headers["Authorization"] == "Bearer per-user-token"

        # No auth
        client = MCPClient("http://example.com")
        headers = client._get_auth_headers()
        assert headers == {}

    @pytest.mark.asyncio
    @patch.object(mcp_client_module, "streamable_http_client")
    @patch.object(mcp_client_module, "ClientSession")
    async def test_run_with_session(self, mock_session_class, mock_transport):
        """Test run_with_session establishes session with auth headers."""
        # Setup mocks
        mock_transport_ctx = AsyncMock()
        mock_transport.return_value = mock_transport_ctx
        mock_transport_instance = MagicMock()
        mock_transport_ctx.__aenter__ = AsyncMock(return_value=mock_transport_instance)

        mock_session_ctx = AsyncMock()
        mock_session_class.return_value = mock_session_ctx
        mock_session_instance = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_instance)

        client = MCPClient(
            "http://example.com",
            auth_type=MCPAuth.bearer_token,
            auth_value="test_token",
        )

        async def _operation(session):
            return "ok"

        await client.run_with_session(_operation)

        # Verify transport was created with auth headers
        call_args = mock_transport.call_args
        http_client = call_args[1]["http_client"]
        assert http_client.headers.get("Authorization") == "Bearer test_token"

        # Verify session was initialized
        mock_session_instance.initialize.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(mcp_client_module, "streamable_http_client")
    @patch.object(mcp_client_module, "ClientSession")
    async def test_list_tools(self, mock_session_class, mock_transport):
        """Test listing tools from the server."""
        # Setup mocks
        mock_transport_ctx = AsyncMock()
        mock_transport.return_value = mock_transport_ctx
        mock_transport_instance = MagicMock()
        mock_transport_ctx.__aenter__ = AsyncMock(return_value=mock_transport_instance)

        mock_session_ctx = AsyncMock()
        mock_session_class.return_value = mock_session_ctx
        mock_session_instance = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_instance)

        mock_tools = [
            MCPTool(
                name="test_tool",
                description="Test tool",
                inputSchema={
                    "type": "object",
                    "properties": {"arg1": {"type": "string"}},
                    "required": ["arg1"],
                },
            )
        ]
        mock_result = MagicMock()
        mock_result.tools = mock_tools
        mock_session_instance.list_tools.return_value = mock_result

        client = MCPClient("http://example.com")
        result = await client.list_tools()

        assert result == mock_tools
        mock_session_instance.initialize.assert_called_once()
        mock_session_instance.list_tools.assert_called_once()

    @pytest.mark.asyncio
    @patch.object(mcp_client_module, "streamable_http_client")
    @patch.object(mcp_client_module, "ClientSession")
    async def test_list_tools_follows_next_cursor_until_exhausted(
        self,
        mock_session_class,
        mock_transport,
    ):
        """Test listing tools follows MCP pagination cursors until exhausted."""
        mock_transport_ctx = AsyncMock()
        mock_transport.return_value = mock_transport_ctx
        mock_transport_instance = MagicMock()
        mock_transport_ctx.__aenter__ = AsyncMock(return_value=mock_transport_instance)

        mock_session_ctx = AsyncMock()
        mock_session_class.return_value = mock_session_ctx
        mock_session_instance = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_instance)

        first_page_tools = [
            MCPTool(name=f"tool_{idx}", description=f"Tool {idx}", inputSchema={}) for idx in range(100)
        ]
        second_page_tool = MCPTool(
            name="tool_100",
            description="Tool 100",
            inputSchema={},
        )
        mock_session_instance.list_tools.side_effect = [
            ListToolsResult(tools=first_page_tools, nextCursor="page-2"),
            ListToolsResult(tools=[second_page_tool]),
        ]

        client = MCPClient("http://example.com")
        result = await client.list_tools()

        assert result == [*first_page_tools, second_page_tool]
        assert mock_session_instance.list_tools.call_count == 2
        second_call_params = mock_session_instance.list_tools.call_args_list[1].kwargs["params"]
        assert isinstance(second_call_params, PaginatedRequestParams)
        assert second_call_params.cursor == "page-2"

    @pytest.mark.asyncio
    @patch.object(mcp_client_module, "streamable_http_client")
    @patch.object(mcp_client_module, "ClientSession")
    async def test_list_tools_stops_when_pagination_reaches_page_cap(
        self,
        mock_session_class,
        mock_transport,
        monkeypatch,
    ):
        """Test listing tools returns accumulated tools if an upstream keeps returning new cursors."""
        monkeypatch.setattr(mcp_client_module, "MCP_TOOL_LISTING_MAX_PAGES", 2, raising=False)

        mock_transport_ctx = AsyncMock()
        mock_transport.return_value = mock_transport_ctx
        mock_transport_instance = MagicMock()
        mock_transport_ctx.__aenter__ = AsyncMock(return_value=mock_transport_instance)

        mock_session_ctx = AsyncMock()
        mock_session_class.return_value = mock_session_ctx
        mock_session_instance = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_instance)

        mock_session_instance.list_tools.side_effect = [
            ListToolsResult(
                tools=[MCPTool(name="tool_0", description="Tool 0", inputSchema={})],
                nextCursor="page-2",
            ),
            ListToolsResult(
                tools=[MCPTool(name="tool_1", description="Tool 1", inputSchema={})],
                nextCursor="page-3",
            ),
            ListToolsResult(
                tools=[MCPTool(name="tool_2", description="Tool 2", inputSchema={})],
            ),
        ]

        client = MCPClient("http://example.com")
        result = await client.list_tools(raise_on_error=True)

        assert [tool.name for tool in result] == ["tool_0", "tool_1"]
        assert mock_session_instance.list_tools.call_count == 2

    @pytest.mark.asyncio
    @patch.object(mcp_client_module, "streamable_http_client")
    @patch.object(mcp_client_module, "ClientSession")
    async def test_list_tools_raises_on_repeated_next_cursor(
        self,
        mock_session_class,
        mock_transport,
    ):
        """Test listing tools fails if an upstream repeats a cursor."""
        mock_transport_ctx = AsyncMock()
        mock_transport.return_value = mock_transport_ctx
        mock_transport_instance = MagicMock()
        mock_transport_ctx.__aenter__ = AsyncMock(return_value=mock_transport_instance)

        mock_session_ctx = AsyncMock()
        mock_session_class.return_value = mock_session_ctx
        mock_session_instance = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_instance)

        mock_session_instance.list_tools.side_effect = [
            ListToolsResult(tools=[], nextCursor="same-cursor"),
            ListToolsResult(tools=[], nextCursor="same-cursor"),
        ]

        client = MCPClient("http://example.com")
        with pytest.raises(RuntimeError, match="repeated tools/list cursor"):
            await client.list_tools(raise_on_error=True)

        assert mock_session_instance.list_tools.call_count == 2

    @pytest.mark.asyncio
    @patch.object(mcp_client_module, "streamable_http_client")
    @patch.object(mcp_client_module, "ClientSession")
    async def test_call_tool(self, mock_session_class, mock_transport):
        """Test calling a tool."""
        from mcp.types import CallToolRequestParams

        # Setup mocks
        mock_transport_ctx = AsyncMock()
        mock_transport.return_value = mock_transport_ctx
        mock_transport_instance = MagicMock()
        mock_transport_ctx.__aenter__ = AsyncMock(return_value=mock_transport_instance)

        mock_session_ctx = AsyncMock()
        mock_session_class.return_value = mock_session_ctx
        mock_session_instance = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session_instance)

        mock_result = MCPCallToolResult(content=[])
        mock_session_instance.call_tool.return_value = mock_result

        client = MCPClient("http://example.com")
        params = CallToolRequestParams(name="test_tool", arguments={"arg1": "value1"})
        result = await client.call_tool(params)

        assert result == mock_result
        mock_session_instance.initialize.assert_called_once()
        mock_session_instance.call_tool.assert_called_once_with(
            name="test_tool", arguments={"arg1": "value1"}, progress_callback=ANY
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
