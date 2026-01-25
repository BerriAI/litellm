import os
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


class TestMCPClientSessionCaching:
    """Test MCP Client session caching (connection pooling) functionality"""

    def test_session_cache_defaults_to_disabled(self):
        """Test that session caching is disabled by default for backwards compatibility"""
        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
        )
        assert client.use_session_cache is False
        assert client.session_cache_ttl == 300.0

    def test_session_cache_can_be_enabled(self):
        """Test that session caching can be enabled"""
        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
            session_cache_ttl=60.0,
        )
        assert client.use_session_cache is True
        assert client.session_cache_ttl == 60.0

    def test_is_session_valid_returns_false_when_no_session(self):
        """Test _is_session_valid returns False when no session is cached"""
        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )
        assert client._is_session_valid() is False

    def test_is_session_valid_returns_false_when_no_timestamp(self):
        """Test _is_session_valid returns False when session exists but no timestamp"""
        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )
        client._cached_session = MagicMock()
        client._session_last_used_at = None
        assert client._is_session_valid() is False

    def test_is_session_valid_returns_false_when_ttl_expired(self):
        """Test _is_session_valid returns False when TTL has expired"""
        import time

        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
            session_cache_ttl=1.0,  # 1 second TTL
        )
        client._cached_session = MagicMock()
        client._session_last_used_at = time.time() - 2.0  # 2 seconds ago
        assert client._is_session_valid() is False

    def test_is_session_valid_returns_true_when_within_ttl(self):
        """Test _is_session_valid returns True when session is within TTL"""
        import time

        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
            session_cache_ttl=300.0,  # 5 minutes
        )
        client._cached_session = MagicMock()
        client._session_last_used_at = time.time()  # Just now
        assert client._is_session_valid() is True

    @pytest.mark.asyncio
    async def test_cleanup_cached_session(self):
        """Test _cleanup_cached_session properly cleans up resources"""
        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )

        # Setup mock cached resources
        mock_session = MagicMock()
        mock_session.__aexit__ = AsyncMock()
        mock_transport_ctx = MagicMock()
        mock_transport_ctx.__aexit__ = AsyncMock()
        mock_http_client = MagicMock()
        mock_http_client.aclose = AsyncMock()

        client._cached_session = mock_session
        client._cached_transport_ctx = mock_transport_ctx
        client._cached_http_client = mock_http_client
        client._session_last_used_at = 12345.0

        await client._cleanup_cached_session()

        # Verify cleanup was called
        mock_session.__aexit__.assert_called_once()
        mock_transport_ctx.__aexit__.assert_called_once()
        mock_http_client.aclose.assert_called_once()

        # Verify state was cleared
        assert client._cached_session is None
        assert client._cached_transport_ctx is None
        assert client._cached_http_client is None
        assert client._session_last_used_at is None

    @pytest.mark.asyncio
    async def test_close_cleans_up_session(self):
        """Test close() properly cleans up cached session"""
        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )

        # Setup mock cached session
        mock_session = MagicMock()
        mock_session.__aexit__ = AsyncMock()
        client._cached_session = mock_session

        await client.close()

        # Verify session was cleaned up
        assert client._cached_session is None

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.sse_client")
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_run_operation_uses_cached_session_when_enabled(
        self, mock_session_class, mock_sse_client
    ):
        """Test _run_operation uses run_with_cached_session when caching is enabled"""
        # Setup mocks
        mock_transport = (MagicMock(), MagicMock())
        mock_sse_client.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
        mock_sse_client.return_value.__aexit__ = AsyncMock()

        mock_session_instance = MagicMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_class.return_value = mock_session_instance

        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )

        call_count = 0

        async def _operation(session):
            nonlocal call_count
            call_count += 1
            return f"result_{call_count}"

        # First call should create session
        result1 = await client._run_operation(_operation)
        assert result1 == "result_1"

        # Session should be cached
        assert client._cached_session is not None

        # Second call should reuse session (not create new one)
        result2 = await client._run_operation(_operation)
        assert result2 == "result_2"

        # sse_client should only be called once (session reused)
        assert mock_sse_client.call_count == 1

        # Clean up
        await client.close()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.sse_client")
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_run_operation_creates_new_session_when_disabled(
        self, mock_session_class, mock_sse_client
    ):
        """Test _run_operation uses run_with_session (new connection) when caching is disabled"""
        # Setup mocks
        mock_transport = (MagicMock(), MagicMock())
        mock_sse_client.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
        mock_sse_client.return_value.__aexit__ = AsyncMock()

        mock_session_instance = MagicMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_class.return_value = mock_session_instance

        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=False,  # Caching disabled
        )

        async def _operation(session):
            return "result"

        # First call
        await client._run_operation(_operation)

        # Second call
        await client._run_operation(_operation)

        # sse_client should be called twice (new session each time)
        assert mock_sse_client.call_count == 2

        # No cached session should exist
        assert client._cached_session is None

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.sse_client")
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_get_or_create_session_creates_new_when_none_cached(
        self, mock_session_class, mock_sse_client
    ):
        """Test _get_or_create_session creates new session when none is cached"""
        # Setup mocks
        mock_transport = (MagicMock(), MagicMock())
        mock_sse_client.return_value.__aenter__ = AsyncMock(return_value=mock_transport)
        mock_sse_client.return_value.__aexit__ = AsyncMock()

        mock_session_instance = MagicMock()
        mock_session_instance.__aenter__ = AsyncMock(return_value=mock_session_instance)
        mock_session_instance.__aexit__ = AsyncMock()
        mock_session_instance.initialize = AsyncMock()
        mock_session_class.return_value = mock_session_instance

        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )

        # No session cached initially
        assert client._cached_session is None

        # Get or create should create a new session
        session = await client._get_or_create_session()

        assert session is not None
        assert client._cached_session is not None
        assert client._session_last_used_at is not None

        # Clean up
        await client.close()

    @pytest.mark.asyncio
    async def test_get_or_create_session_reuses_valid_session(self):
        """Test _get_or_create_session reuses session when valid"""
        import time

        client = MCPClient(
            server_url="http://localhost:8765/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )

        # Pre-cache a mock session
        mock_session = MagicMock()
        client._cached_session = mock_session
        client._session_last_used_at = time.time()

        # Get or create should return the cached session
        session = await client._get_or_create_session()

        assert session is mock_session


if __name__ == "__main__":
    pytest.main([__file__])
