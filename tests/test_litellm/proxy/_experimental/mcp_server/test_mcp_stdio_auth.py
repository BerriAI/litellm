"""
Test suite for MCP stdio server dynamic authentication functionality.

Tests that stdio MCP servers can receive dynamic authentication tokens
through environment variables on each request.
"""
import pytest
import os
import sys
from unittest.mock import Mock, patch, AsyncMock
from typing import Dict, Any

# Add the path to find the modules
sys.path.insert(
    0, os.path.abspath("../../../..")
)

from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPAuth, MCPTransport, MCPStdioConfig
from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager
from litellm.types.mcp_server.mcp_server_manager import MCPServer


class TestMCPStdioAuth:
    """Test dynamic authentication for stdio MCP servers."""

    def test_get_auth_env_vars_bearer_token(self):
        """Test that bearer token auth generates correct environment variables."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            auth_value="test-bearer-token",
            stdio_config=MCPStdioConfig(
                command="python",
                args=["-m", "test_server"],
                env={}
            )
        )

        env_vars = client._get_auth_env_vars()

        assert env_vars["MCP_AUTH_TOKEN"] == "test-bearer-token"
        assert env_vars["MCP_AUTH_TYPE"] == "bearer_token"
        assert env_vars["MCP_BEARER_TOKEN"] == "test-bearer-token"

    def test_get_auth_env_vars_api_key(self):
        """Test that API key auth generates correct environment variables."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.api_key,
            auth_value="test-api-key",
            stdio_config=MCPStdioConfig(
                command="python",
                args=["-m", "test_server"],
                env={}
            )
        )

        env_vars = client._get_auth_env_vars()

        assert env_vars["MCP_AUTH_TOKEN"] == "test-api-key"
        assert env_vars["MCP_AUTH_TYPE"] == "api_key"
        assert env_vars["MCP_API_KEY"] == "test-api-key"

    def test_get_auth_env_vars_basic_auth(self):
        """Test that basic auth generates correct environment variables."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.basic,
            auth_value="test:password",  # Raw username:password format
            stdio_config=MCPStdioConfig(
                command="python",
                args=["-m", "test_server"],
                env={}
            )
        )

        env_vars = client._get_auth_env_vars()

        # Should be base64 encoded "test:password"
        expected_encoded = "dGVzdDpwYXNzd29yZA=="
        assert env_vars["MCP_AUTH_TOKEN"] == expected_encoded
        assert env_vars["MCP_AUTH_TYPE"] == "basic"
        assert env_vars["MCP_BASIC_AUTH"] == expected_encoded

    def test_get_auth_env_vars_authorization(self):
        """Test that authorization auth generates correct environment variables."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.authorization,
            auth_value="Custom auth-value",
            stdio_config=MCPStdioConfig(
                command="python",
                args=["-m", "test_server"],
                env={}
            )
        )

        env_vars = client._get_auth_env_vars()

        assert env_vars["MCP_AUTH_TOKEN"] == "Custom auth-value"
        assert env_vars["MCP_AUTH_TYPE"] == "authorization"
        assert env_vars["MCP_AUTHORIZATION"] == "Custom auth-value"

    def test_get_auth_env_vars_no_auth(self):
        """Test that no auth generates empty environment variables."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            stdio_config=MCPStdioConfig(
                command="python",
                args=["-m", "test_server"],
                env={}
            )
        )

        env_vars = client._get_auth_env_vars()

        assert env_vars == {}

    def test_get_auth_env_vars_no_auth_type(self):
        """Test auth token without auth type."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_value="test-token",
            stdio_config=MCPStdioConfig(
                command="python",
                args=["-m", "test_server"],
                env={}
            )
        )

        env_vars = client._get_auth_env_vars()

        assert env_vars["MCP_AUTH_TOKEN"] == "test-token"
        assert "MCP_AUTH_TYPE" not in env_vars

    @pytest.mark.asyncio
    @patch('litellm.experimental_mcp_client.client.stdio_client')
    async def test_stdio_connect_with_auth_injection(self, mock_stdio_client):
        """Test that auth is properly injected into stdio environment during connect."""
        # Mock the stdio client context manager
        mock_transport = Mock()
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=(mock_transport, mock_transport))
        mock_stdio_client.return_value = mock_context

        # Mock the session
        mock_session = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)

        with patch('litellm.experimental_mcp_client.client.ClientSession', return_value=mock_session_context):
            client = MCPClient(
                transport_type=MCPTransport.stdio,
                auth_type=MCPAuth.bearer_token,
                auth_value="test-token",
                stdio_config=MCPStdioConfig(
                    command="python",
                    args=["-m", "test_server"],
                    env={"EXISTING_VAR": "existing_value"}
                )
            )

            await client.connect()

            # Verify stdio_client was called with merged environment
            mock_stdio_client.assert_called_once()
            server_params = mock_stdio_client.call_args[0][0]

            assert server_params.command == "python"
            assert server_params.args == ["-m", "test_server"]

            # Check that auth environment variables were injected
            env = server_params.env
            assert env["EXISTING_VAR"] == "existing_value"  # Existing env preserved
            assert env["MCP_AUTH_TOKEN"] == "test-token"
            assert env["MCP_AUTH_TYPE"] == "bearer_token"
            assert env["MCP_BEARER_TOKEN"] == "test-token"

    @pytest.mark.asyncio
    @patch('litellm.experimental_mcp_client.client.stdio_client')
    async def test_stdio_connect_preserves_existing_env(self, mock_stdio_client):
        """Test that existing environment variables are preserved when auth is injected."""
        # Mock the stdio client
        mock_transport = Mock()
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=(mock_transport, mock_transport))
        mock_stdio_client.return_value = mock_context

        # Mock the session
        mock_session = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)

        existing_env = {
            "PATH": "/usr/bin",
            "SERVER_CONFIG": "production",
            "MCP_CUSTOM_VAR": "custom_value"
        }

        with patch('litellm.experimental_mcp_client.client.ClientSession', return_value=mock_session_context):
            client = MCPClient(
                transport_type=MCPTransport.stdio,
                auth_type=MCPAuth.api_key,
                auth_value="api-key-123",
                stdio_config=MCPStdioConfig(
                    command="node",
                    args=["server.js"],
                    env=existing_env
                )
            )

            await client.connect()

            # Get the environment passed to stdio_client
            server_params = mock_stdio_client.call_args[0][0]
            env = server_params.env

            # Verify existing environment is preserved
            assert env["PATH"] == "/usr/bin"
            assert env["SERVER_CONFIG"] == "production"
            assert env["MCP_CUSTOM_VAR"] == "custom_value"

            # Verify auth variables are added
            assert env["MCP_AUTH_TOKEN"] == "api-key-123"
            assert env["MCP_AUTH_TYPE"] == "api_key"
            assert env["MCP_API_KEY"] == "api-key-123"

    @pytest.mark.asyncio
    @patch('litellm.experimental_mcp_client.client.stdio_client')
    async def test_stdio_connect_without_auth(self, mock_stdio_client):
        """Test that stdio connect works without authentication."""
        # Mock the stdio client
        mock_transport = Mock()
        mock_context = AsyncMock()
        mock_context.__aenter__ = AsyncMock(return_value=(mock_transport, mock_transport))
        mock_stdio_client.return_value = mock_context

        # Mock the session
        mock_session = AsyncMock()
        mock_session_context = AsyncMock()
        mock_session_context.__aenter__ = AsyncMock(return_value=mock_session)

        with patch('litellm.experimental_mcp_client.client.ClientSession', return_value=mock_session_context):
            client = MCPClient(
                transport_type=MCPTransport.stdio,
                stdio_config=MCPStdioConfig(
                    command="python",
                    args=["-m", "test_server"],
                    env={"ONLY_VAR": "only_value"}
                )
            )

            await client.connect()

            # Get the environment passed to stdio_client
            server_params = mock_stdio_client.call_args[0][0]
            env = server_params.env

            # Verify only existing environment is present (no auth vars)
            assert env == {"ONLY_VAR": "only_value"}
            assert "MCP_AUTH_TOKEN" not in env
            assert "MCP_AUTH_TYPE" not in env

    @pytest.mark.asyncio
    async def test_server_manager_dynamic_auth_precedence(self):
        """Test that dynamic auth headers take precedence over static config."""
        manager = MCPServerManager()

        # Create a mock server with static auth
        mock_server = MCPServer(
            server_id="test-server",
            name="Test Server",
            transport=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            authentication_token="static-token",
            command="python",
            args=["-m", "test_server"],
            env={}
        )

        # Test that dynamic auth header takes precedence
        dynamic_auth = "dynamic-token"
        client = manager._create_mcp_client(
            server=mock_server,
            mcp_auth_header=dynamic_auth
        )

        # The client should be created with the dynamic auth value
        assert client._mcp_auth_value == dynamic_auth
        assert client.auth_type == MCPAuth.bearer_token

    @pytest.mark.asyncio
    async def test_server_manager_fallback_to_static_auth(self):
        """Test that static auth is used when no dynamic auth provided."""
        manager = MCPServerManager()

        # Create a mock server with static auth
        mock_server = MCPServer(
            server_id="test-server",
            name="Test Server",
            transport=MCPTransport.stdio,
            auth_type=MCPAuth.api_key,
            authentication_token="static-api-key",
            command="python",
            args=["-m", "test_server"],
            env={}
        )

        # Test fallback to static auth when no dynamic auth provided
        client = manager._create_mcp_client(
            server=mock_server,
            mcp_auth_header=None
        )

        # The client should be created with the static auth value
        assert client._mcp_auth_value == "static-api-key"
        assert client.auth_type == MCPAuth.api_key

    @pytest.mark.asyncio
    async def test_auth_env_vars_override_config_env(self):
        """Test that auth environment variables can override config environment."""
        # This tests the edge case where config already has MCP_AUTH_TOKEN
        existing_env = {
            "MCP_AUTH_TOKEN": "old-token",
            "MCP_AUTH_TYPE": "old-type",
            "OTHER_VAR": "preserved"
        }

        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            auth_value="new-token",
            stdio_config=MCPStdioConfig(
                command="python",
                args=["-m", "test_server"],
                env=existing_env
            )
        )

        auth_env = client._get_auth_env_vars()

        # Verify auth env vars are generated correctly
        assert auth_env["MCP_AUTH_TOKEN"] == "new-token"
        assert auth_env["MCP_AUTH_TYPE"] == "bearer_token"
        assert auth_env["MCP_BEARER_TOKEN"] == "new-token"

        # When merged with existing env, auth should take precedence
        merged_env = existing_env.copy()
        merged_env.update(auth_env)

        assert merged_env["MCP_AUTH_TOKEN"] == "new-token"  # Overridden
        assert merged_env["MCP_AUTH_TYPE"] == "bearer_token"  # Overridden
        assert merged_env["MCP_BEARER_TOKEN"] == "new-token"  # New
        assert merged_env["OTHER_VAR"] == "preserved"  # Preserved

    def test_update_auth_value_stdio_behavior(self):
        """Test that update_auth_value works correctly for stdio clients."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            stdio_config=MCPStdioConfig(
                command="python",
                args=["-m", "test_server"],
                env={}
            )
        )

        # Initially no auth
        assert client._mcp_auth_value is None

        # Update auth value
        client.update_auth_value("updated-token")
        assert client._mcp_auth_value == "updated-token"

        # Verify env vars reflect the update
        env_vars = client._get_auth_env_vars()
        assert env_vars["MCP_AUTH_TOKEN"] == "updated-token"
        assert env_vars["MCP_BEARER_TOKEN"] == "updated-token"

    def test_basic_auth_encoding_stdio(self):
        """Test that basic auth is properly encoded for stdio."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.basic,
            stdio_config=MCPStdioConfig(
                command="python",
                args=["-m", "test_server"],
                env={}
            )
        )

        # Test basic auth encoding
        client.update_auth_value("username:password")

        # The value should be base64 encoded
        import base64
        expected_encoded = base64.b64encode("username:password".encode("utf-8")).decode()
        assert client._mcp_auth_value == expected_encoded

        # Verify env vars use the encoded value
        env_vars = client._get_auth_env_vars()
        assert env_vars["MCP_AUTH_TOKEN"] == expected_encoded
        assert env_vars["MCP_BASIC_AUTH"] == expected_encoded