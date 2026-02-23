import os
import ssl
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, "../../../")

import litellm.experimental_mcp_client.client as mcp_client_module
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
            async def _noop(session):
                return None

            await client.run_with_session(_noop)

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_mcp_client_stdio_connect_success(
        self, mock_session, mock_stdio_client
    ):
        """Test successful stdio connection"""
        # Setup mocks - create proper async context manager
        mock_transport = (MagicMock(), MagicMock())
        mock_stdio_ctx = AsyncMock()
        mock_stdio_ctx.__aenter__.return_value = mock_transport
        mock_stdio_ctx.__aexit__.return_value = None
        mock_stdio_client.return_value = mock_stdio_ctx

        mock_session_instance = AsyncMock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__.return_value = mock_session_instance
        mock_session_ctx.__aexit__.return_value = None
        mock_session.return_value = mock_session_ctx

        stdio_config = MCPStdioConfig(
            command="python", args=["-m", "my_mcp_server"], env={"DEBUG": "1"}
        )

        client = MCPClient(transport_type=MCPTransport.stdio, stdio_config=stdio_config)

        async def _operation(session):
            return "ok"

        await client.run_with_session(_operation)

        # Verify stdio_client was called with correct parameters
        mock_stdio_client.assert_called_once()
        call_args = mock_stdio_client.call_args[0][0]
        assert call_args.command == "python"
        assert call_args.args == ["-m", "my_mcp_server"]
        assert call_args.env == {"DEBUG": "1"}

    @pytest.mark.asyncio
    @patch.object(mcp_client_module, "streamable_http_client")
    @patch.dict(
        os.environ,
        {
            "SSL_CERT_FILE": "/path/to/custom/ca-bundle.pem",
            "SSL_CERTIFICATE": "/path/to/client-cert.pem",
        },
    )
    async def test_mcp_client_ssl_configuration_from_env(
        self, mock_streamable_http_client
    ):
        """Test that MCP client uses SSL configuration from environment variables"""
        # Setup mocks - create proper async context manager
        mock_transport = (MagicMock(), MagicMock())
        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__.return_value = mock_transport
        mock_http_ctx.__aexit__.return_value = None
        mock_streamable_http_client.return_value = mock_http_ctx

        # Mock the session
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session:
            mock_session_instance = AsyncMock()
            mock_session_instance.initialize = AsyncMock()
            mock_session_ctx = AsyncMock()
            mock_session_ctx.__aenter__.return_value = mock_session_instance
            mock_session_ctx.__aexit__.return_value = None
            mock_session.return_value = mock_session_ctx

            client = MCPClient(
                server_url="https://mcp-server.example.com",
                transport_type=MCPTransport.http,
            )

            async def _operation(session):
                return "ok"

            await client.run_with_session(_operation)

            # Verify streamablehttp_client was called
            mock_streamable_http_client.assert_called_once()
            call_kwargs = mock_streamable_http_client.call_args[1]
            assert "http_client" in call_kwargs
            http_client = call_kwargs["http_client"]
            assert isinstance(http_client, httpx.AsyncClient)

            # Test the factory still creates a client with proper SSL config
            httpx_factory = client._create_httpx_client_factory()
            test_client = httpx_factory(headers={"test": "header"})

            assert test_client is not None
            assert isinstance(test_client, httpx.AsyncClient)
            assert test_client.headers is not None
            await test_client.aclose()

    @pytest.mark.asyncio
    @patch.object(mcp_client_module, "sse_client")
    async def test_mcp_client_ssl_verify_parameter(self, mock_sse_client):
        """Test that MCP client uses ssl_verify parameter when provided"""
        # Setup mocks - create proper async context manager
        mock_transport = (MagicMock(), MagicMock())
        mock_sse_ctx = AsyncMock()
        mock_sse_ctx.__aenter__.return_value = mock_transport
        mock_sse_ctx.__aexit__.return_value = None
        mock_sse_client.return_value = mock_sse_ctx

        # Mock the session
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session:
            mock_session_instance = AsyncMock()
            mock_session_instance.initialize = AsyncMock()
            mock_session_ctx = AsyncMock()
            mock_session_ctx.__aenter__.return_value = mock_session_instance
            mock_session_ctx.__aexit__.return_value = None
            mock_session.return_value = mock_session_ctx

            # Test with ssl_verify=False
            client = MCPClient(
                server_url="https://mcp-server.example.com",
                transport_type=MCPTransport.sse,
                ssl_verify=False,
            )

            async def _operation(session):
                return "ok"

            await client.run_with_session(_operation)

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
    @patch.object(mcp_client_module, "streamable_http_client")
    async def test_mcp_client_ssl_verify_custom_path(self, mock_streamable_http_client):
        """Test that MCP client uses custom CA bundle path from ssl_verify parameter"""
        # Setup mocks - create proper async context manager
        mock_transport = (MagicMock(), MagicMock())
        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__.return_value = mock_transport
        mock_http_ctx.__aexit__.return_value = None
        mock_streamable_http_client.return_value = mock_http_ctx

        # Mock the session
        with patch(
            "litellm.experimental_mcp_client.client.ClientSession"
        ) as mock_session:
            mock_session_instance = AsyncMock()
            mock_session_instance.initialize = AsyncMock()
            mock_session_ctx = AsyncMock()
            mock_session_ctx.__aenter__.return_value = mock_session_instance
            mock_session_ctx.__aexit__.return_value = None
            mock_session.return_value = mock_session_ctx

            # Test with custom CA bundle path
            custom_ca_path = "/custom/path/to/ca-bundle.pem"
            client = MCPClient(
                server_url="https://mcp-server.example.com",
                transport_type=MCPTransport.http,
                ssl_verify=custom_ca_path,
            )

            async def _operation(session):
                return "ok"

            await client.run_with_session(_operation)

            # Verify streamablehttp_client was called
            mock_streamable_http_client.assert_called_once()
            call_kwargs = mock_streamable_http_client.call_args[1]
            assert "http_client" in call_kwargs
            http_client = call_kwargs["http_client"]
            assert isinstance(http_client, httpx.AsyncClient)

            httpx_factory = client._create_httpx_client_factory()
            test_client = httpx_factory(headers={"test": "header"})

            assert test_client is not None
            assert isinstance(test_client, httpx.AsyncClient)
            assert test_client.headers is not None
            await test_client.aclose()


if __name__ == "__main__":
    pytest.main([__file__])
