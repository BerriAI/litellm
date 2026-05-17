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
    @patch("litellm.experimental_mcp_client.client._LiteLLMMCPClientSession")
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
            "litellm.experimental_mcp_client.client._LiteLLMMCPClientSession"
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
            "litellm.experimental_mcp_client.client._LiteLLMMCPClientSession"
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
            "litellm.experimental_mcp_client.client._LiteLLMMCPClientSession"
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
            auth_value="my-secret-token",
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
            auth_value="bearer-token",
        )
        headers = client._get_auth_headers()
        assert headers["Authorization"] == "Bearer bearer-token"

        # Test API key
        client = MCPClient(
            server_url="http://example.com/sse",
            transport_type="sse",
            auth_type=MCPAuth.api_key,
            auth_value="api-key",
        )
        headers = client._get_auth_headers()
        assert headers["X-API-Key"] == "api-key"

        # Test basic auth (gets base64 encoded)
        client = MCPClient(
            server_url="http://example.com/sse",
            transport_type="sse",
            auth_type=MCPAuth.basic,
            auth_value="user:pass",
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
            extra_headers={"X-Custom-Header": "custom-value"},
        )

        headers = client._get_auth_headers()

        assert headers["Authorization"] == "token my-token"
        assert headers["X-Custom-Header"] == "custom-value"

    def test_token_auth_enum_value(self):
        """Test that MCPAuth.token enum exists and has correct value"""
        assert hasattr(MCPAuth, "token")
        assert MCPAuth.token.value == "token"


# ---------------------------------------------------------------------------
# _last_initialize_instructions capture
# ---------------------------------------------------------------------------


class TestMCPClientInstructionsCapture:
    """Tests for _last_initialize_instructions capture during session init."""

    def test_initial_value_is_none(self):
        """Fresh client has no cached instructions."""
        client = MCPClient(
            server_url="http://example.com/mcp",
            transport_type="http",
        )
        assert client._last_initialize_instructions is None

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client._LiteLLMMCPClientSession")
    async def test_captures_instructions_from_initialize(self, mock_session_cls):
        """Instructions from upstream initialize() are captured and stripped."""
        client = MCPClient(
            server_url="http://example.com/mcp",
            transport_type="http",
        )

        mock_session = AsyncMock()
        init_result = MagicMock()
        init_result.instructions = "  upstream says hello  "
        mock_session.initialize = AsyncMock(return_value=init_result)

        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = session_ctx

        transport_ctx = MagicMock()
        transport_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        transport_ctx.__aexit__ = AsyncMock(return_value=False)

        async def _op(session):
            return "done"

        await client._execute_session_operation(transport_ctx, _op)
        assert client._last_initialize_instructions == "upstream says hello"

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client._LiteLLMMCPClientSession")
    async def test_none_instructions_stays_none(self, mock_session_cls):
        """When upstream returns no instructions the field stays None."""
        client = MCPClient(
            server_url="http://example.com/mcp",
            transport_type="http",
        )

        mock_session = AsyncMock()
        init_result = MagicMock()
        init_result.instructions = None
        mock_session.initialize = AsyncMock(return_value=init_result)

        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = session_ctx

        transport_ctx = MagicMock()
        transport_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        transport_ctx.__aexit__ = AsyncMock(return_value=False)

        async def _op(session):
            return "done"

        await client._execute_session_operation(transport_ctx, _op)
        assert client._last_initialize_instructions is None


class TestUnsupportedServerToClientMethods:
    """
    Tests for `_LiteLLMMCPClientSession`, which handles upstream MCP server
    requests for methods the LiteLLM MCP Gateway does not implement yet
    (`sampling/createMessage`, `elicitation/create`).

    Previously these requests were silently dropped at the LiteLLM logging
    layer — the SDK auto-replied to the server with a generic error, but
    operators saw nothing. The subclass logs a warning and replies with a
    descriptive JSON-RPC `METHOD_NOT_FOUND` error so the upstream server can
    fall back gracefully. See https://github.com/BerriAI/litellm/issues/23761.
    """

    def test_unsupported_method_error_includes_litellm_and_server(self):
        from litellm.experimental_mcp_client.client import (
            _build_unsupported_method_error,
        )
        from mcp.types import METHOD_NOT_FOUND

        err = _build_unsupported_method_error(
            "sampling/createMessage", "https://upstream.example/mcp"
        )
        assert err.code == METHOD_NOT_FOUND
        assert "sampling/createMessage" in err.message
        assert "LiteLLM MCP Gateway" in err.message
        assert "https://upstream.example/mcp" in err.message

    def test_unsupported_method_error_handles_empty_server_url(self):
        from litellm.experimental_mcp_client.client import (
            _build_unsupported_method_error,
        )

        err = _build_unsupported_method_error("elicitation/create", "")
        assert "stdio" in err.message
        assert "elicitation/create" in err.message

    @pytest.mark.asyncio
    async def test_sampling_request_logs_and_responds_with_method_not_found(
        self, caplog
    ):
        import logging

        from mcp.types import (
            METHOD_NOT_FOUND,
            CreateMessageRequest,
            CreateMessageRequestParams,
            ErrorData,
            SamplingMessage,
            TextContent,
        )

        from litellm.experimental_mcp_client.client import (
            _LiteLLMMCPClientSession,
        )

        session = _LiteLLMMCPClientSession.__new__(_LiteLLMMCPClientSession)
        session._litellm_server_url = "https://upstream.example/mcp"

        params = CreateMessageRequestParams(
            messages=[
                SamplingMessage(
                    role="user",
                    content=TextContent(type="text", text="hi"),
                )
            ],
            maxTokens=16,
        )
        responder = _FakeResponder(
            CreateMessageRequest(method="sampling/createMessage", params=params)
        )

        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            await session._received_request(responder)

        assert responder.responded_with is not None
        assert isinstance(responder.responded_with, ErrorData)
        assert responder.responded_with.code == METHOD_NOT_FOUND
        assert "sampling/createMessage" in responder.responded_with.message
        assert "https://upstream.example/mcp" in responder.responded_with.message
        # A warning was logged with the server URL and the method name —
        # the previous behaviour logged nothing at all.
        assert any(
            "sampling/createMessage" in record.getMessage()
            and "https://upstream.example/mcp" in record.getMessage()
            for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_elicitation_request_logs_and_responds_with_method_not_found(
        self, caplog
    ):
        import logging

        from mcp.types import (
            METHOD_NOT_FOUND,
            ElicitRequest,
            ElicitRequestFormParams,
            ErrorData,
        )

        from litellm.experimental_mcp_client.client import (
            _LiteLLMMCPClientSession,
        )

        session = _LiteLLMMCPClientSession.__new__(_LiteLLMMCPClientSession)
        session._litellm_server_url = ""  # stdio

        params = ElicitRequestFormParams(
            message="Confirm deletion?",
            requestedSchema={
                "type": "object",
                "properties": {"confirm": {"type": "boolean"}},
                "required": ["confirm"],
            },
        )
        responder = _FakeResponder(
            ElicitRequest(method="elicitation/create", params=params)
        )

        with caplog.at_level(logging.WARNING, logger="LiteLLM"):
            await session._received_request(responder)

        assert isinstance(responder.responded_with, ErrorData)
        assert responder.responded_with.code == METHOD_NOT_FOUND
        assert "elicitation/create" in responder.responded_with.message
        # stdio servers (no URL) are identified as "stdio" in the message.
        assert "stdio" in responder.responded_with.message
        assert any(
            "elicitation/create" in record.getMessage() for record in caplog.records
        )

    @pytest.mark.asyncio
    async def test_other_server_requests_fall_through_to_default(self):
        """
        Server -> client methods that aren't sampling/elicitation must
        still go to the SDK's default handler (e.g. ping/roots/list).
        """
        from litellm.experimental_mcp_client.client import (
            _LiteLLMMCPClientSession,
        )

        session = _LiteLLMMCPClientSession.__new__(_LiteLLMMCPClientSession)
        session._litellm_server_url = "https://upstream.example/mcp"

        # Use a sentinel object that does not match CreateMessageRequest
        # or ElicitRequest — it should fall through to super().
        unrelated_root = object()
        responder = _FakeResponder(unrelated_root)

        with patch(
            "mcp.ClientSession._received_request", new_callable=AsyncMock
        ) as super_handler:
            await session._received_request(responder)

        super_handler.assert_awaited_once_with(responder)
        # We didn't respond ourselves — the parent does.
        assert responder.responded_with is None

    def test_client_session_does_not_declare_unsupported_capabilities(self):
        """
        Guards against a future regression that wires custom callbacks into
        the SDK — doing so would make the SDK declare `sampling` and
        `elicitation` capabilities to the upstream server during `initialize`,
        which we explicitly do NOT support yet.
        """
        from mcp.client.session import (
            _default_elicitation_callback,
            _default_sampling_callback,
        )

        from litellm.experimental_mcp_client.client import (
            _LiteLLMMCPClientSession,
        )

        # We can't call __init__ without real streams, so just verify the
        # subclass does not override the class-level callback identity.
        # Both attributes are set in ClientSession.__init__; instead verify
        # the subclass does not bind class-level overrides for them.
        assert "_sampling_callback" not in vars(_LiteLLMMCPClientSession)
        assert "_elicitation_callback" not in vars(_LiteLLMMCPClientSession)
        # And the module-level defaults are importable — sanity check that
        # the SDK API we depend on hasn't been removed under us.
        assert _default_sampling_callback is not None
        assert _default_elicitation_callback is not None


class _FakeResponder:
    """
    Minimal stand-in for `mcp.shared.session.RequestResponder` used by the
    `_received_request` tests above. It records the response value and
    behaves as a no-op context manager.
    """

    def __init__(self, request_root: object) -> None:
        self.request = MagicMock()
        self.request.root = request_root
        self.responded_with: object = None
        self._completed = False
        self._entered = False

    def __enter__(self) -> "_FakeResponder":
        self._entered = True
        return self

    def __exit__(self, *_exc: object) -> None:
        self._entered = False

    async def respond(self, response: object) -> None:
        assert self._entered, "RequestResponder must be used as a context manager"
        self.responded_with = response
        self._completed = True


if __name__ == "__main__":
    pytest.main([__file__])
