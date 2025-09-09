"""
Unit tests for the MCPClient class - critical functionality only.
"""
import base64
import os
import sys
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

# Add the project root to the path
sys.path.insert(0, os.path.abspath("../../.."))

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
            timeout=30.0
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
            auth_value="test_token"
        )
        headers = client._get_auth_headers()
        assert headers == {"Authorization": "Bearer test_token", "MCP-Protocol-Version": "2025-06-18"}
        
        # Basic auth
        client = MCPClient(
            "http://example.com",
            auth_type=MCPAuth.basic,
            auth_value="user:pass"
        )
        expected_encoded = base64.b64encode("user:pass".encode("utf-8")).decode()
        headers = client._get_auth_headers()
        assert headers == {"Authorization": f"Basic {expected_encoded}", "MCP-Protocol-Version": "2025-06-18"}
        
        # API key
        client = MCPClient(
            "http://example.com",
            auth_type=MCPAuth.api_key,
            auth_value="api_key_123"
        )
        headers = client._get_auth_headers()
        assert headers == {"X-API-Key": "api_key_123", "MCP-Protocol-Version": "2025-06-18"}

        # Custom authorization header
        client = MCPClient(
            "http://example.com",
            auth_type=MCPAuth.authorization,
            auth_value="Token custom_token",
        )
        headers = client._get_auth_headers()
        assert headers == {
            "Authorization": "Token custom_token",
            "MCP-Protocol-Version": "2025-06-18",
        }
        
        # No auth
        client = MCPClient("http://example.com")
        headers = client._get_auth_headers()
        assert headers == {"MCP-Protocol-Version": "2025-06-18"}
        
        # Custom protocol version
        from litellm.types.mcp import MCPSpecVersion
        client = MCPClient(
            "http://example.com",
            protocol_version=MCPSpecVersion.mar_2025
        )
        headers = client._get_auth_headers()
        assert headers == {"MCP-Protocol-Version": "2025-03-26"}
    
    @pytest.mark.asyncio
    @patch('litellm.experimental_mcp_client.client.streamablehttp_client')
    @patch('litellm.experimental_mcp_client.client.ClientSession')
    async def test_connect(self, mock_session_class, mock_transport):
        """Test connecting to MCP server with authentication."""
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
            auth_value="test_token"
        )
        await client.connect()
        
        # Verify transport was created with auth headers
        call_args = mock_transport.call_args
        assert call_args[1]['headers'] == {"Authorization": "Bearer test_token", "MCP-Protocol-Version": "2025-06-18"}
        
        # Verify session was initialized
        mock_session_instance.initialize.assert_called_once()
        assert client._session == mock_session_instance
    
    @pytest.mark.asyncio
    @patch('litellm.experimental_mcp_client.client.streamablehttp_client')
    @patch('litellm.experimental_mcp_client.client.ClientSession')
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
                    "required": ["arg1"]
                }
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
    @patch('litellm.experimental_mcp_client.client.streamablehttp_client')
    @patch('litellm.experimental_mcp_client.client.ClientSession')
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
            name="test_tool", 
            arguments={"arg1": "value1"}
        )

    def test_protocol_version_header_extraction(self):
        """Test that MCP protocol version header is correctly extracted from requests."""
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
        
        # Mock scope with headers
        mock_scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [
                (b"authorization", b"Bearer test_token"),
                (b"mcp-protocol-version", b"2025-06-18"),
                (b"content-type", b"application/json"),
            ]
        }
        
        # Mock the user_api_key_auth function
        with patch('litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth') as mock_auth:
            mock_auth.return_value = MagicMock()
            
            # Call process_mcp_request
            import asyncio
            result = asyncio.run(MCPRequestHandler.process_mcp_request(mock_scope))
            
            # Verify the protocol version is extracted
            user_api_key_auth, mcp_auth_header, mcp_servers, mcp_server_auth_headers, mcp_protocol_version = result
            
            assert mcp_protocol_version == "2025-06-18"

    def test_protocol_version_header_missing(self):
        """Test that MCP protocol version header is None when not provided."""
        from litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp import MCPRequestHandler
        
        # Mock scope without protocol version header
        mock_scope = {
            "type": "http",
            "method": "GET",
            "path": "/test",
            "headers": [
                (b"authorization", b"Bearer test_token"),
                (b"content-type", b"application/json"),
            ]
        }
        
        # Mock the user_api_key_auth function
        with patch('litellm.proxy._experimental.mcp_server.auth.user_api_key_auth_mcp.user_api_key_auth') as mock_auth:
            mock_auth.return_value = MagicMock()
            
            # Call process_mcp_request
            import asyncio
            result = asyncio.run(MCPRequestHandler.process_mcp_request(mock_scope))
            
            # Verify the protocol version is None
            user_api_key_auth, mcp_auth_header, mcp_servers, mcp_server_auth_headers, mcp_protocol_version = result
            
            assert mcp_protocol_version is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 