"""
Integration tests for stdio authentication functionality.

These tests demonstrate how the stdio authentication feature integrates
with the broader MCP server management system, particularly for HTTP → stdio bridge scenarios.
"""

import pytest
from unittest.mock import AsyncMock, patch
from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPAuth, MCPTransport, MCPStdioConfig


class TestMCPStdioAuthIntegration:
    """Integration tests for stdio authentication functionality."""

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    async def test_http_to_stdio_bridge_bearer_token(self, mock_stdio_client):
        """
        Test HTTP → stdio bridge scenario with bearer token authentication.

        This simulates a typical use case where:
        1. HTTP request comes in with Authorization: Bearer <token>
        2. MCP server manager extracts the token
        3. Creates stdio client with dynamic auth
        4. Token is passed to stdio process via environment variables
        """
        # Simulate HTTP request with bearer token
        http_auth_header = "Bearer abc123-bearer-token"

        # Extract token from HTTP header (as would be done by proxy)
        token = http_auth_header.replace("Bearer ", "")

        # Setup mocks for stdio client
        mock_transport = (AsyncMock(), AsyncMock())
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_transport
        mock_stdio_client.return_value = mock_context

        # Create stdio client with extracted token
        stdio_config: MCPStdioConfig = {
            "command": "python",
            "args": ["-m", "my_mcp_server"],
            "env": {"SERVER_MODE": "production"},
        }

        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            auth_value=token,
            stdio_config=stdio_config,
        )

        # Mock session initialization
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session_class:
            mock_session = AsyncMock()
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__.return_value = mock_session
            mock_session_class.return_value = mock_session_context

            # Connect (this triggers auth injection)
            await client.connect()

            # Verify stdio process receives auth via environment
            mock_stdio_client.assert_called_once()
            call_args = mock_stdio_client.call_args[0][0]

            expected_env = {
                "SERVER_MODE": "production",  # Original env preserved
                "MCP_AUTH_TOKEN": "abc123-bearer-token",
                "MCP_AUTH_TYPE": "bearer_token",
                "MCP_BEARER_TOKEN": "abc123-bearer-token",
            }
            assert call_args.env == expected_env
            assert call_args.command == "python"
            assert call_args.args == ["-m", "my_mcp_server"]

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    async def test_http_to_stdio_bridge_api_key(self, mock_stdio_client):
        """
        Test HTTP → stdio bridge scenario with API key authentication.

        This simulates API key-based authentication flow:
        1. HTTP request with X-API-Key header
        2. Token extracted and passed to stdio
        3. Stdio server can access via MCP_API_KEY environment variable
        """
        # Simulate HTTP request with API key
        api_key = "sk-1234567890abcdef"

        # Setup mocks
        mock_transport = (AsyncMock(), AsyncMock())
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_transport
        mock_stdio_client.return_value = mock_context

        stdio_config: MCPStdioConfig = {
            "command": "/usr/local/bin/mcp-server",
            "args": ["--config", "/etc/mcp/config.json"],
            "env": {"LOG_LEVEL": "INFO", "CONFIG_PATH": "/etc/mcp"},
        }

        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.api_key,
            auth_value=api_key,
            stdio_config=stdio_config,
        )

        # Mock session
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session_class:
            mock_session = AsyncMock()
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__.return_value = mock_session
            mock_session_class.return_value = mock_session_context

            await client.connect()

            # Verify environment variable injection
            call_args = mock_stdio_client.call_args[0][0]

            expected_env = {
                "LOG_LEVEL": "INFO",
                "CONFIG_PATH": "/etc/mcp",
                "MCP_AUTH_TOKEN": api_key,
                "MCP_AUTH_TYPE": "api_key",
                "MCP_API_KEY": api_key,
            }
            assert call_args.env == expected_env

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    async def test_dynamic_auth_update_scenario(self, mock_stdio_client):
        """
        Test dynamic authentication update scenario.

        This simulates updating authentication credentials dynamically,
        which is useful for token refresh or changing auth contexts.
        """
        # Setup mocks
        mock_transport = (AsyncMock(), AsyncMock())
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_transport
        mock_stdio_client.return_value = mock_context

        stdio_config: MCPStdioConfig = {
            "command": "node",
            "args": ["mcp-server.js"],
            "env": {"NODE_ENV": "development"},
        }

        # Create client with initial auth
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            auth_value="initial-token",
            stdio_config=stdio_config,
        )

        # Update auth dynamically (simulating token refresh)
        client.update_auth_value("refreshed-token-xyz")

        # Mock session for connection
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session_class:
            mock_session = AsyncMock()
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__.return_value = mock_session
            mock_session_class.return_value = mock_session_context

            await client.connect()

            # Verify updated token is used
            call_args = mock_stdio_client.call_args[0][0]

            expected_env = {
                "NODE_ENV": "development",
                "MCP_AUTH_TOKEN": "refreshed-token-xyz",
                "MCP_AUTH_TYPE": "bearer_token",
                "MCP_BEARER_TOKEN": "refreshed-token-xyz",
            }
            assert call_args.env == expected_env

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    async def test_multi_client_auth_isolation(self, mock_stdio_client):
        """
        Test that multiple clients with different auth don't interfere.

        This is important for multi-tenant scenarios where different
        requests may have different authentication credentials.
        """
        # Setup mocks
        mock_transport = (AsyncMock(), AsyncMock())
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_transport
        mock_stdio_client.return_value = mock_context

        stdio_config: MCPStdioConfig = {
            "command": "python",
            "args": ["-m", "shared_mcp_server"],
            "env": {"SHARED_CONFIG": "true"},
        }

        # Create first client with bearer token
        client1 = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            auth_value="user1-bearer-token",
            stdio_config=stdio_config,
        )

        # Create second client with API key
        client2 = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.api_key,
            auth_value="user2-api-key",
            stdio_config=stdio_config,
        )

        # Mock session for both connections
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session_class:
            mock_session = AsyncMock()
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__.return_value = mock_session
            mock_session_class.return_value = mock_session_context

            # Connect first client
            await client1.connect()
            first_call_args = mock_stdio_client.call_args[0][0]

            # Reset mock for second call
            mock_stdio_client.reset_mock()
            mock_context.__aenter__.return_value = mock_transport

            # Connect second client
            await client2.connect()
            second_call_args = mock_stdio_client.call_args[0][0]

            # Verify each client gets its own auth environment
            expected_env1 = {
                "SHARED_CONFIG": "true",
                "MCP_AUTH_TOKEN": "user1-bearer-token",
                "MCP_AUTH_TYPE": "bearer_token",
                "MCP_BEARER_TOKEN": "user1-bearer-token",
            }
            assert first_call_args.env == expected_env1

            expected_env2 = {
                "SHARED_CONFIG": "true",
                "MCP_AUTH_TOKEN": "user2-api-key",
                "MCP_AUTH_TYPE": "api_key",
                "MCP_API_KEY": "user2-api-key",
            }
            assert second_call_args.env == expected_env2

            # Verify environments are completely isolated
            assert first_call_args.env != second_call_args.env
