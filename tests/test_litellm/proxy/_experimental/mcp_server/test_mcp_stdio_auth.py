"""
Test stdio authentication functionality in MCPClient.

This module tests the dynamic authentication capability for stdio MCP servers,
allowing auth credentials to be passed via environment variables.
"""

import pytest
from unittest.mock import AsyncMock, patch
from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPAuth, MCPTransport, MCPStdioConfig


class TestMCPStdioAuth:
    """Test suite for stdio authentication in MCPClient."""

    @pytest.fixture
    def stdio_config(self) -> MCPStdioConfig:
        """Basic stdio configuration fixture."""
        return {
            "command": "python",
            "args": ["-m", "mcp_server"],
            "env": {"BASE_VAR": "base_value"},
        }

    def test_get_auth_env_vars_no_auth(self):
        """Test _get_auth_env_vars with no authentication configured."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            stdio_config={"command": "test", "args": []},
        )

        env_vars = client._get_auth_env_vars()
        assert env_vars == {}

    def test_get_auth_env_vars_bearer_token(self):
        """Test _get_auth_env_vars with bearer token authentication."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            auth_value="test-bearer-token",
            stdio_config={"command": "test", "args": []},
        )

        env_vars = client._get_auth_env_vars()
        expected = {
            "MCP_AUTH_TOKEN": "test-bearer-token",
            "MCP_AUTH_TYPE": "bearer_token",
            "MCP_BEARER_TOKEN": "test-bearer-token",
        }
        assert env_vars == expected

    def test_get_auth_env_vars_api_key(self):
        """Test _get_auth_env_vars with API key authentication."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.api_key,
            auth_value="test-api-key",
            stdio_config={"command": "test", "args": []},
        )

        env_vars = client._get_auth_env_vars()
        expected = {
            "MCP_AUTH_TOKEN": "test-api-key",
            "MCP_AUTH_TYPE": "api_key",
            "MCP_API_KEY": "test-api-key",
        }
        assert env_vars == expected

    def test_get_auth_env_vars_basic_auth(self):
        """Test _get_auth_env_vars with basic authentication."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.basic,
            auth_value="test:password",  # Raw credentials, will be encoded by update_auth_value
            stdio_config={"command": "test", "args": []},
        )

        env_vars = client._get_auth_env_vars()
        expected = {
            "MCP_AUTH_TOKEN": "dGVzdDpwYXNzd29yZA==",  # base64 encoded
            "MCP_AUTH_TYPE": "basic",
            "MCP_BASIC_AUTH": "dGVzdDpwYXNzd29yZA==",
        }
        assert env_vars == expected

    def test_get_auth_env_vars_authorization(self):
        """Test _get_auth_env_vars with custom authorization."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.authorization,
            auth_value="Custom token-value",
            stdio_config={"command": "test", "args": []},
        )

        env_vars = client._get_auth_env_vars()
        expected = {
            "MCP_AUTH_TOKEN": "Custom token-value",
            "MCP_AUTH_TYPE": "authorization",
            "MCP_AUTHORIZATION": "Custom token-value",
        }
        assert env_vars == expected

    def test_get_auth_env_vars_token_only(self):
        """Test _get_auth_env_vars with token but no auth type."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_value="token-without-type",
            stdio_config={"command": "test", "args": []},
        )

        env_vars = client._get_auth_env_vars()
        expected = {"MCP_AUTH_TOKEN": "token-without-type"}
        assert env_vars == expected

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    async def test_stdio_connect_with_auth_injection(
        self, mock_stdio_client, stdio_config
    ):
        """Test that stdio connection properly injects auth environment variables."""
        # Setup mocks
        mock_transport = (AsyncMock(), AsyncMock())
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_transport
        mock_stdio_client.return_value = mock_context

        # Create client with auth
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            auth_value="test-token",
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

            # Verify stdio_client was called with auth environment
            mock_stdio_client.assert_called_once()
            call_args = mock_stdio_client.call_args[0][0]

            expected_env = {
                "BASE_VAR": "base_value",
                "MCP_AUTH_TOKEN": "test-token",
                "MCP_AUTH_TYPE": "bearer_token",
                "MCP_BEARER_TOKEN": "test-token",
            }
            assert call_args.env == expected_env

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    async def test_stdio_connect_without_auth(self, mock_stdio_client, stdio_config):
        """Test that stdio connection works without authentication."""
        # Setup mocks
        mock_transport = (AsyncMock(), AsyncMock())
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_transport
        mock_stdio_client.return_value = mock_context

        # Create client without auth
        client = MCPClient(transport_type=MCPTransport.stdio, stdio_config=stdio_config)

        # Mock session
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session_class:
            mock_session = AsyncMock()
            mock_session_context = AsyncMock()
            mock_session_context.__aenter__.return_value = mock_session
            mock_session_class.return_value = mock_session_context

            await client.connect()

            # Verify stdio_client was called with original environment only
            mock_stdio_client.assert_called_once()
            call_args = mock_stdio_client.call_args[0][0]

            expected_env = {"BASE_VAR": "base_value"}  # Only original env
            assert call_args.env == expected_env

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    async def test_stdio_connect_preserves_original_env(self, mock_stdio_client):
        """Test that auth environment variables don't overwrite existing ones."""
        stdio_config_with_conflict = {
            "command": "python",
            "args": ["-m", "mcp_server"],
            "env": {
                "BASE_VAR": "base_value",
                "MCP_AUTH_TOKEN": "original_token",  # This should be overwritten
            },
        }

        # Setup mocks
        mock_transport = (AsyncMock(), AsyncMock())
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_transport
        mock_stdio_client.return_value = mock_context

        # Create client with auth
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.api_key,
            auth_value="new-api-key",
            stdio_config=stdio_config_with_conflict,
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

            # Verify auth variables override original ones
            mock_stdio_client.assert_called_once()
            call_args = mock_stdio_client.call_args[0][0]

            expected_env = {
                "BASE_VAR": "base_value",  # Preserved
                "MCP_AUTH_TOKEN": "new-api-key",  # Overwritten with new auth
                "MCP_AUTH_TYPE": "api_key",
                "MCP_API_KEY": "new-api-key",
            }
            assert call_args.env == expected_env

    @pytest.mark.asyncio
    async def test_stdio_connect_missing_config_error(self):
        """Test that stdio connection raises error when stdio_config is missing."""
        client = MCPClient(transport_type=MCPTransport.stdio)

        with pytest.raises(
            ValueError, match="stdio_config is required for stdio transport"
        ):
            await client.connect()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    async def test_stdio_connect_empty_env_config(self, mock_stdio_client):
        """Test stdio connection with empty env configuration."""
        stdio_config_empty_env = {
            "command": "python",
            "args": ["-m", "mcp_server"]
            # No env key
        }

        # Setup mocks
        mock_transport = (AsyncMock(), AsyncMock())
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_transport
        mock_stdio_client.return_value = mock_context

        # Create client with auth
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            auth_type=MCPAuth.bearer_token,
            auth_value="test-token",
            stdio_config=stdio_config_empty_env,
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

            # Verify auth variables are added to empty env
            mock_stdio_client.assert_called_once()
            call_args = mock_stdio_client.call_args[0][0]

            expected_env = {
                "MCP_AUTH_TOKEN": "test-token",
                "MCP_AUTH_TYPE": "bearer_token",
                "MCP_BEARER_TOKEN": "test-token",
            }
            assert call_args.env == expected_env

    def test_auth_value_update_preserves_stdio_behavior(self):
        """Test that updating auth value works correctly for stdio transport."""
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            stdio_config={"command": "test", "args": []},
        )

        # Update auth dynamically
        client.auth_type = MCPAuth.api_key
        client.update_auth_value("new-api-key")

        env_vars = client._get_auth_env_vars()
        expected = {
            "MCP_AUTH_TOKEN": "new-api-key",
            "MCP_AUTH_TYPE": "api_key",
            "MCP_API_KEY": "new-api-key",
        }
        assert env_vars == expected
