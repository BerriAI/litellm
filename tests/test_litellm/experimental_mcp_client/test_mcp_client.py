import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, "../../../")

import litellm.experimental_mcp_client.client as mcp_client_module
from litellm.experimental_mcp_client.client import MCPClient
from litellm.types.mcp import MCPAuth, MCPStdioConfig, MCPTransport


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

    def test_token_auth_header_generation(self):
        """Test that token auth generates correct Authorization header"""
        client = MCPClient(
            server_url="http://example.com/sse",
            transport_type="sse",
            auth_type=MCPAuth.token,
            auth_value="my-secret-token"
        )
        
        headers = client._get_auth_headers()
        
        assert "Authorization" in headers
        assert headers["Authorization"] == "token my-secret-token"

    def test_token_auth_compatibility_with_existing_auth_types(self):
        """Verify existing auth types are not affected by token auth addition"""
        # Test bearer token
        client = MCPClient(
            server_url="http://example.com/sse",
            transport_type="sse",
            auth_type=MCPAuth.bearer_token,
            auth_value="bearer-token"
        )
        headers = client._get_auth_headers()
        assert headers["Authorization"] == "Bearer bearer-token"
        
        # Test API key
        client = MCPClient(
            server_url="http://example.com/sse",
            transport_type="sse",
            auth_type=MCPAuth.api_key,
            auth_value="api-key"
        )
        headers = client._get_auth_headers()
        assert headers["X-API-Key"] == "api-key"
        
        # Test basic auth (gets base64 encoded)
        client = MCPClient(
            server_url="http://example.com/sse",
            transport_type="sse",
            auth_type=MCPAuth.basic,
            auth_value="user:pass"
        )
        headers = client._get_auth_headers()
        assert headers["Authorization"].startswith("Basic ")

    def test_token_auth_with_extra_headers(self):
        """Test that token auth works alongside extra headers"""
        client = MCPClient(
            server_url="http://example.com/sse",
            transport_type="sse",
            auth_type=MCPAuth.token,
            auth_value="my-token",
            extra_headers={"X-Custom-Header": "custom-value"}
        )
        
        headers = client._get_auth_headers()
        
        assert headers["Authorization"] == "token my-token"
        assert headers["X-Custom-Header"] == "custom-value"

    def test_token_auth_enum_value(self):
        """Test that MCPAuth.token enum exists and has correct value"""
        assert hasattr(MCPAuth, "token")
        assert MCPAuth.token.value == "token"


class TestMCPClientSessionCachingE2E:
    """E2E tests for session caching — exercise full flows through _run_operation."""

    @staticmethod
    def _mock_transport():
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        ctx.__aexit__ = AsyncMock(return_value=False)
        return ctx

    @staticmethod
    def _mock_http_client():
        c = MagicMock()
        c.aclose = AsyncMock()
        return c

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_caching_reuses_session_across_calls(self, mock_session_cls):
        """With caching ON, multiple operations share one transport/session."""
        sessions = []

        def make_session(*a, **kw):
            s = MagicMock()
            s.__aenter__ = AsyncMock(return_value=s)
            s.__aexit__ = AsyncMock()
            s.initialize = AsyncMock()
            sessions.append(s)
            return s

        mock_session_cls.side_effect = make_session

        client = MCPClient(
            server_url="http://test/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )
        transports = []

        def tracked():
            t = self._mock_transport()
            h = self._mock_http_client()
            transports.append(t)
            return t, h

        client._create_transport_context = tracked

        seen = []

        async def op(session):
            seen.append(id(session))
            return "ok"

        await client._run_operation(op)
        await client._run_operation(op)
        await client._run_operation(op)

        assert len(transports) == 1, "should create transport only once"
        assert len(sessions) == 1, "should create session only once"
        assert len(set(seen)) == 1, "all ops should see the same session"
        await client.close()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_caching_disabled_creates_fresh_session_each_call(self, mock_session_cls):
        """With caching OFF, each operation gets a new transport/session."""
        sessions = []

        def make_session(*a, **kw):
            s = MagicMock()
            s.__aenter__ = AsyncMock(return_value=s)
            s.__aexit__ = AsyncMock()
            s.initialize = AsyncMock()
            sessions.append(s)
            return s

        mock_session_cls.side_effect = make_session

        client = MCPClient(
            server_url="http://test/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=False,
        )
        transports = []

        def tracked():
            t = self._mock_transport()
            h = self._mock_http_client()
            transports.append(t)
            return t, h

        client._create_transport_context = tracked

        async def op(session):
            return "ok"

        await client._run_operation(op)
        await client._run_operation(op)

        assert len(transports) == 2, "should create a new transport each time"
        assert len(sessions) == 2, "should create a new session each time"
        assert client._cached_session is None

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_ttl_expiry_creates_new_session_and_closes_old(self, mock_session_cls):
        """After TTL expires, next operation creates a new session and closes old resources."""
        import time as time_mod

        sessions = []

        def make_session(*a, **kw):
            s = MagicMock()
            s.__aenter__ = AsyncMock(return_value=s)
            s.__aexit__ = AsyncMock()
            s.initialize = AsyncMock()
            sessions.append(s)
            return s

        mock_session_cls.side_effect = make_session

        client = MCPClient(
            server_url="http://test/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
            session_cache_ttl=1.0,
        )
        transports = []
        http_clients = []

        def tracked():
            t = self._mock_transport()
            h = self._mock_http_client()
            transports.append(t)
            http_clients.append(h)
            return t, h

        client._create_transport_context = tracked

        async def op(session):
            return "ok"

        await client._run_operation(op)
        assert len(transports) == 1

        # Expire TTL
        client._session_last_used_at = time_mod.time() - 2.0

        await client._run_operation(op)
        assert len(transports) == 2, "should create a second session after TTL expiry"

        # Old resources should have been closed
        sessions[0].__aexit__.assert_called()
        transports[0].__aexit__.assert_called()
        http_clients[0].aclose.assert_called()

        await client.close()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_connection_error_retries_with_new_session(self, mock_session_cls):
        """On connection error, old session is closed and retry uses a new session."""
        sessions = []

        def make_session(*a, **kw):
            s = MagicMock()
            s.__aenter__ = AsyncMock(return_value=s)
            s.__aexit__ = AsyncMock()
            s.initialize = AsyncMock()
            sessions.append(s)
            return s

        mock_session_cls.side_effect = make_session

        client = MCPClient(
            server_url="http://test/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )
        transports = []
        http_clients = []

        def tracked():
            t = self._mock_transport()
            h = self._mock_http_client()
            transports.append(t)
            http_clients.append(h)
            return t, h

        client._create_transport_context = tracked

        # Setup initial session
        async def setup(s):
            return "ok"

        await client._run_operation(setup)
        broken = sessions[0]

        # Operation that fails on the broken session, succeeds on the new one
        async def retry_op(session):
            if session is broken:
                raise ConnectionError("pipe broken")
            return "recovered"

        result = await client._run_operation(retry_op)
        assert result == "recovered"
        assert len(transports) == 2

        # Old resources closed
        broken.__aexit__.assert_called()
        transports[0].__aexit__.assert_called()
        http_clients[0].aclose.assert_called()

        await client.close()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_concurrent_errors_create_single_recovery_session(self, mock_session_cls):
        """Multiple concurrent callers hitting a broken session share one recovery."""
        sessions = []

        def make_session(*a, **kw):
            s = MagicMock()
            s.__aenter__ = AsyncMock(return_value=s)
            s.__aexit__ = AsyncMock()
            s.initialize = AsyncMock()
            sessions.append(s)
            return s

        mock_session_cls.side_effect = make_session

        client = MCPClient(
            server_url="http://test/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )
        transport_count = [0]

        def tracked():
            transport_count[0] += 1
            return self._mock_transport(), self._mock_http_client()

        client._create_transport_context = tracked

        # Create initial session
        async def setup(s):
            return "ok"

        await client._run_operation(setup)
        assert transport_count[0] == 1
        broken = sessions[0]

        async def failing_op(session):
            await asyncio.sleep(0)  # yield so all coroutines start concurrently
            if session is broken:
                raise ConnectionError("connection lost")
            return "recovered"

        results = await asyncio.gather(
            client.run_with_cached_session(failing_op),
            client.run_with_cached_session(failing_op),
            client.run_with_cached_session(failing_op),
        )

        assert all(r == "recovered" for r in results)
        # Only 1 recovery session created (2 total: original + recovery)
        assert transport_count[0] == 2
        assert len(sessions) == 2

        await client.close()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_close_cleans_up_and_prevents_reuse(self, mock_session_cls):
        """close() cleans up resources; further operations raise RuntimeError."""

        def make_session(*a, **kw):
            s = MagicMock()
            s.__aenter__ = AsyncMock(return_value=s)
            s.__aexit__ = AsyncMock()
            s.initialize = AsyncMock()
            return s

        mock_session_cls.side_effect = make_session

        client = MCPClient(
            server_url="http://test/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )
        transports = []
        http_clients = []

        def tracked():
            t = self._mock_transport()
            h = self._mock_http_client()
            transports.append(t)
            http_clients.append(h)
            return t, h

        client._create_transport_context = tracked

        async def op(s):
            return "ok"

        await client._run_operation(op)
        await client.close()

        # Resources cleaned up
        transports[0].__aexit__.assert_called()
        http_clients[0].aclose.assert_called()

        # Further operations should fail
        with pytest.raises(RuntimeError, match="MCPClient is closed"):
            await client._run_operation(op)

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_non_connection_error_propagates_without_retry(self, mock_session_cls):
        """Non-connection errors propagate immediately; no retry, no session churn."""

        def make_session(*a, **kw):
            s = MagicMock()
            s.__aenter__ = AsyncMock(return_value=s)
            s.__aexit__ = AsyncMock()
            s.initialize = AsyncMock()
            return s

        mock_session_cls.side_effect = make_session

        client = MCPClient(
            server_url="http://test/sse",
            transport_type=MCPTransport.sse,
            use_session_cache=True,
        )
        transport_count = [0]

        def tracked():
            transport_count[0] += 1
            return self._mock_transport(), self._mock_http_client()

        client._create_transport_context = tracked

        async def bad_op(session):
            raise ValueError("application error")

        with pytest.raises(ValueError, match="application error"):
            await client.run_with_cached_session(bad_op)

        # No second session created (no retry)
        assert transport_count[0] == 1
        await client.close()

    @pytest.mark.parametrize(
        "exc,expected",
        [
            (type("BrokenResourceError", (Exception,), {})("broken"), True),
            (type("ClosedResourceError", (Exception,), {})("closed"), True),
            (type("EndOfStream", (Exception,), {})("eof"), True),
            (ConnectionError("conn reset"), True),
            (ConnectionResetError("reset"), True),  # subclass of ConnectionError
            (ValueError("bad input"), False),
            (RuntimeError("something failed"), False),
            (TimeoutError("timed out"), False),  # not treated as connection error
        ],
    )
    def test_is_connection_error_classification(self, exc, expected):
        """Connection error classification is precise — no false positives."""
        client = MCPClient(
            server_url="http://test/sse",
            transport_type=MCPTransport.sse,
        )
        assert client._is_connection_error(exc) is expected


class TestMCPClientStdioCachingGuard:
    """Session caching must be rejected for stdio transport."""

    def test_stdio_with_session_cache_raises(self):
        with pytest.raises(ValueError, match="not supported for stdio transport"):
            MCPClient(
                transport_type=MCPTransport.stdio,
                stdio_config={"command": "echo", "args": [], "env": {}},
                use_session_cache=True,
            )

    def test_stdio_without_session_cache_ok(self):
        client = MCPClient(
            transport_type=MCPTransport.stdio,
            stdio_config={"command": "echo", "args": [], "env": {}},
            use_session_cache=False,
        )
        assert client.use_session_cache is False


if __name__ == "__main__":
    pytest.main([__file__])
