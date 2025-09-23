"""
Integration test demonstrating stdio MCP server dynamic authentication.

This test shows how dynamic auth tokens are passed to stdio servers
through environment variables, enabling HTTP → stdio bridge scenarios.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, AsyncMock

# Add the path to find the modules
sys.path.insert(
    0, os.path.abspath("../../../..")
)

from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPAuth, MCPTransport, MCPStdioConfig
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class TestMCPStdioAuthIntegration:
    """Integration tests for stdio MCP server dynamic authentication."""

    @pytest.mark.asyncio
    async def test_end_to_end_dynamic_auth_flow(self):
        """Test complete flow from HTTP request to stdio process with dynamic auth."""

        # Setup: Create MCP server manager
        manager = MCPServerManager()

        # Step 1: Define a stdio server configuration
        server_config = {
            "test_stdio_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["-m", "my_mcp_server"],
                "auth_type": "bearer_token",
                "authentication_token": "static-fallback-token",  # Fallback auth
                "env": {
                    "SERVER_CONFIG": "production",
                    "LOG_LEVEL": "info"
                }
            }
        }

        # Step 2: Load server from config (simulates config.yaml loading)
        manager.load_servers_from_config(server_config)

        # Step 3: Get the loaded server
        server_id = list(manager.config_mcp_servers.keys())[0]
        server = manager.config_mcp_servers[server_id]

        # Step 4: Simulate HTTP request with dynamic auth header
        dynamic_auth_token = "dynamic-request-token-123"

        # Step 5: Create MCP client with dynamic auth (simulates call_tool request)
        with patch('litellm.experimental_mcp_client.client.stdio_client') as mock_stdio_client, \
             patch('litellm.experimental_mcp_client.client.ClientSession') as mock_session_class:

            # Mock stdio client
            mock_transport = Mock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=(mock_transport, mock_transport))
            mock_stdio_client.return_value = mock_context

            # Mock session
            mock_session = AsyncMock()
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session_context

            # Create client with dynamic auth (this is what server manager does)
            client = manager._create_mcp_client(
                server=server,
                mcp_auth_header=dynamic_auth_token  # Dynamic auth from request
            )

            # Connect to simulate tool call
            await client.connect()

            # Verify: stdio_client was called with merged environment
            mock_stdio_client.assert_called_once()
            server_params = mock_stdio_client.call_args[0][0]

            # Assert: Command and args are correct
            assert server_params.command == "python"
            assert server_params.args == ["-m", "my_mcp_server"]

            # Assert: Original environment variables are preserved
            env = server_params.env
            assert env["SERVER_CONFIG"] == "production"
            assert env["LOG_LEVEL"] == "info"

            # Assert: Dynamic auth token is injected (takes precedence over static)
            assert env["MCP_AUTH_TOKEN"] == dynamic_auth_token
            assert env["MCP_AUTH_TYPE"] == "bearer_token"
            assert env["MCP_BEARER_TOKEN"] == dynamic_auth_token

            # Verify: Client uses dynamic auth, not static fallback
            assert client._mcp_auth_value == dynamic_auth_token

    @pytest.mark.asyncio
    async def test_http_to_stdio_bridge_scenario(self):
        """Test HTTP → stdio bridge where client can send token per request."""

        # Scenario: HTTP client makes request to LiteLLM proxy
        # Proxy needs to forward auth to stdio MCP server

        manager = MCPServerManager()

        # Server configured for stdio transport
        stdio_server = MCPServer(
            server_id="bridge-server",
            name="HTTP→Stdio Bridge Server",
            transport=MCPTransport.stdio,
            auth_type=MCPAuth.api_key,
            authentication_token=None,  # No static auth - dynamic only
            command="node",
            args=["mcp-server.js"],
            env={"NODE_ENV": "production"}
        )

        with patch('litellm.experimental_mcp_client.client.stdio_client') as mock_stdio_client, \
             patch('litellm.experimental_mcp_client.client.ClientSession') as mock_session_class:

            # Mock setup
            mock_transport = Mock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=(mock_transport, mock_transport))
            mock_stdio_client.return_value = mock_context

            mock_session = AsyncMock()
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session_context

            # Simulate HTTP request with auth header
            http_request_auth = "user-api-key-xyz789"

            # Manager creates client with per-request auth
            client = manager._create_mcp_client(
                server=stdio_server,
                mcp_auth_header=http_request_auth
            )

            await client.connect()

            # Verify stdio process gets the HTTP request auth
            server_params = mock_stdio_client.call_args[0][0]
            env = server_params.env

            assert env["MCP_AUTH_TOKEN"] == http_request_auth
            assert env["MCP_AUTH_TYPE"] == "api_key"
            assert env["MCP_API_KEY"] == http_request_auth
            assert env["NODE_ENV"] == "production"  # Original env preserved

    def test_auth_environment_variable_conventions(self):
        """Test that auth environment variables follow expected conventions."""

        test_cases = [
            {
                "auth_type": MCPAuth.bearer_token,
                "auth_value": "bearer-token-123",
                "expected_vars": {
                    "MCP_AUTH_TOKEN": "bearer-token-123",
                    "MCP_AUTH_TYPE": "bearer_token",
                    "MCP_BEARER_TOKEN": "bearer-token-123"
                }
            },
            {
                "auth_type": MCPAuth.api_key,
                "auth_value": "api-key-456",
                "expected_vars": {
                    "MCP_AUTH_TOKEN": "api-key-456",
                    "MCP_AUTH_TYPE": "api_key",
                    "MCP_API_KEY": "api-key-456"
                }
            },
            {
                "auth_type": MCPAuth.basic,
                "auth_value": "user:pass",
                "expected_vars": {
                    "MCP_AUTH_TOKEN": "dXNlcjpwYXNz",  # base64 encoded
                    "MCP_AUTH_TYPE": "basic",
                    "MCP_BASIC_AUTH": "dXNlcjpwYXNz"
                }
            },
            {
                "auth_type": MCPAuth.authorization,
                "auth_value": "Custom auth-header-value",
                "expected_vars": {
                    "MCP_AUTH_TOKEN": "Custom auth-header-value",
                    "MCP_AUTH_TYPE": "authorization",
                    "MCP_AUTHORIZATION": "Custom auth-header-value"
                }
            }
        ]

        for test_case in test_cases:
            client = MCPClient(
                transport_type=MCPTransport.stdio,
                auth_type=test_case["auth_type"],
                auth_value=test_case["auth_value"],
                stdio_config=MCPStdioConfig(
                    command="python",
                    args=["-m", "test"],
                    env={}
                )
            )

            env_vars = client._get_auth_env_vars()

            for var_name, expected_value in test_case["expected_vars"].items():
                assert env_vars[var_name] == expected_value, \
                    f"Auth type {test_case['auth_type']}: {var_name} should be {expected_value}, got {env_vars.get(var_name)}"

    @pytest.mark.asyncio
    async def test_backward_compatibility_with_static_auth(self):
        """Test that static auth still works when no dynamic auth provided."""

        manager = MCPServerManager()

        # Server with static auth configuration
        server = MCPServer(
            server_id="static-auth-server",
            name="Static Auth Server",
            transport=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            authentication_token="static-token-456",  # Static auth
            command="python",
            args=["-m", "static_server"],
            env={}
        )

        with patch('litellm.experimental_mcp_client.client.stdio_client') as mock_stdio_client, \
             patch('litellm.experimental_mcp_client.client.ClientSession') as mock_session_class:

            # Mock setup
            mock_transport = Mock()
            mock_context = AsyncMock()
            mock_context.__aenter__ = AsyncMock(return_value=(mock_transport, mock_transport))
            mock_stdio_client.return_value = mock_context

            mock_session = AsyncMock()
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session_class.return_value = mock_session_context

            # Create client WITHOUT dynamic auth (simulates config-only usage)
            client = manager._create_mcp_client(
                server=server,
                mcp_auth_header=None  # No dynamic auth
            )

            await client.connect()

            # Verify static auth is used
            server_params = mock_stdio_client.call_args[0][0]
            env = server_params.env

            assert env["MCP_AUTH_TOKEN"] == "static-token-456"
            assert env["MCP_AUTH_TYPE"] == "bearer_token"
            assert env["MCP_BEARER_TOKEN"] == "static-token-456"