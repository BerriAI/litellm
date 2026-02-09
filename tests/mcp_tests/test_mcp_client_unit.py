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
from mcp.types import Tool as MCPTool, CallToolResult as MCPCallToolResult


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
            name="test_tool", arguments={"arg1": "value1"},progress_callback=ANY
        )



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
