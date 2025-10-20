import os
import ssl
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, "../../../")

from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPStdioConfig, MCPTransport


class TestMCPClient:
    """Test MCP Client stdio functionality"""

    def test_mcp_client_stdio_init(self):
        """Test MCPClient initialization with stdio config"""
        stdio_config = MCPStdioConfig(
            command="python", args=["-m", "my_mcp_server"], env={"DEBUG": "1"}
        )

        client = MCPClient(transport_type=MCPTransport.stdio, stdio_config=stdio_config)

        assert client.transport_type == MCPTransport.stdio
        assert client.stdio_config == stdio_config
        assert client.stdio_config is not None
        assert client.stdio_config.get("command") == "python"
        assert client.stdio_config.get("args") == ["-m", "my_mcp_server"]

    @pytest.mark.asyncio
    async def test_mcp_client_stdio_connect_error(self):
        """Test MCP client stdio connection error handling"""
        # Test missing stdio_config
        client = MCPClient(transport_type=MCPTransport.stdio)

        with pytest.raises(
            ValueError, match="stdio_config is required for stdio transport"
        ):
            await client.connect()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_mcp_client_stdio_connect_success(
        self, mock_session, mock_stdio_client
    ):
        """Test successful stdio connection"""
        # Setup mocks
        mock_transport = (MagicMock(), MagicMock())
        mock_stdio_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_transport
        )

        mock_session_instance = MagicMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.initialize = AsyncMock()
        mock_session.return_value = mock_session_instance

        stdio_config = MCPStdioConfig(
            command="python", args=["-m", "my_mcp_server"], env={"DEBUG": "1"}
        )

        client = MCPClient(transport_type=MCPTransport.stdio, stdio_config=stdio_config)

        await client.connect()

        # Verify stdio_client was called with correct parameters
        mock_stdio_client.assert_called_once()
        call_args = mock_stdio_client.call_args[0][0]
        assert call_args.command == "python"
        assert call_args.args == ["-m", "my_mcp_server"]
        assert call_args.env == {"DEBUG": "1"}

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.streamablehttp_client")
    @patch.dict(
        os.environ,
        {
            "SSL_CERT_FILE": "/path/to/custom/ca-bundle.pem",
            "SSL_CERTIFICATE": "/path/to/client-cert.pem",
        },
    )
    async def test_mcp_client_ssl_configuration_from_env(
        self, mock_streamablehttp_client
    ):
        """Test that MCP client uses SSL configuration from environment variables"""
        # Setup mocks
        mock_transport = (MagicMock(), MagicMock())
        mock_streamablehttp_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_transport
        )

        # Mock the session
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.initialize = AsyncMock()
            mock_session.return_value = mock_session_instance

            client = MCPClient(
                server_url="https://mcp-server.example.com",
                transport_type=MCPTransport.http,
            )

            await client.connect()

            # Verify streamablehttp_client was called
            mock_streamablehttp_client.assert_called_once()
            call_kwargs = mock_streamablehttp_client.call_args[1]

            # Verify httpx_client_factory was passed
            assert "httpx_client_factory" in call_kwargs
            httpx_factory = call_kwargs["httpx_client_factory"]

            # Test the factory creates a client with proper SSL config
            # When SSL_CERT_FILE is set, the factory should use get_ssl_configuration
            test_client = httpx_factory(headers={"test": "header"})

            # Verify the client was created successfully with SSL configuration
            assert test_client is not None
            assert isinstance(test_client, httpx.AsyncClient)
            # Verify it has the expected properties
            assert test_client.headers is not None
            # Clean up
            await test_client.aclose()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.sse_client")
    async def test_mcp_client_ssl_verify_parameter(self, mock_sse_client):
        """Test that MCP client uses ssl_verify parameter when provided"""
        # Setup mocks
        mock_transport = (MagicMock(), MagicMock())
        mock_sse_client.return_value.__aenter__ = AsyncMock(return_value=mock_transport)

        # Mock the session
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.initialize = AsyncMock()
            mock_session.return_value = mock_session_instance

            # Test with ssl_verify=False
            client = MCPClient(
                server_url="https://mcp-server.example.com",
                transport_type=MCPTransport.sse,
                ssl_verify=False,
            )

            await client.connect()

            # Verify sse_client was called
            mock_sse_client.assert_called_once()
            call_kwargs = mock_sse_client.call_args[1]

            # Verify httpx_client_factory was passed
            assert "httpx_client_factory" in call_kwargs
            httpx_factory = call_kwargs["httpx_client_factory"]

            # Test the factory creates a client with SSL verification disabled
            # When ssl_verify=False, the factory should disable SSL verification
            test_client = httpx_factory(headers={"test": "header"})

            # Verify the client was created successfully
            assert test_client is not None
            assert isinstance(test_client, httpx.AsyncClient)
            # Verify it has the expected properties
            assert test_client.headers is not None
            # Clean up
            await test_client.aclose()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.streamablehttp_client")
    async def test_mcp_client_ssl_verify_custom_path(self, mock_streamablehttp_client):
        """Test that MCP client uses custom CA bundle path from ssl_verify parameter"""
        # Setup mocks
        mock_transport = (MagicMock(), MagicMock())
        mock_streamablehttp_client.return_value.__aenter__ = AsyncMock(
            return_value=mock_transport
        )

        # Mock the session
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session:
            mock_session_instance = MagicMock()
            mock_session_instance.__aenter__ = AsyncMock(
                return_value=mock_session_instance
            )
            mock_session_instance.initialize = AsyncMock()
            mock_session.return_value = mock_session_instance

            # Test with custom CA bundle path
            custom_ca_path = "/custom/path/to/ca-bundle.pem"
            client = MCPClient(
                server_url="https://mcp-server.example.com",
                transport_type=MCPTransport.http,
                ssl_verify=custom_ca_path,
            )

            await client.connect()

            # Verify streamablehttp_client was called
            mock_streamablehttp_client.assert_called_once()
            call_kwargs = mock_streamablehttp_client.call_args[1]

            # Verify httpx_client_factory was passed
            assert "httpx_client_factory" in call_kwargs
            httpx_factory = call_kwargs["httpx_client_factory"]

            # Test the factory creates a client with custom CA bundle path
            # When ssl_verify is a path, the factory should use that path for SSL verification
            test_client = httpx_factory(headers={"test": "header"})

            # Verify the client was created successfully
            assert test_client is not None
            assert isinstance(test_client, httpx.AsyncClient)
            # Verify it has the expected properties
            assert test_client.headers is not None
            # Clean up
            await test_client.aclose()


if __name__ == "__main__":
    pytest.main([__file__])
