import asyncio
import base64
import importlib
import json
import os
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from types import ModuleType
from typing import Final
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import anyio
import httpx2
import pytest
from mcp import MCPError
from mcp.client.streamable_http import streamable_http_client
from mcp.shared.message import SessionMessage
from mcp.types import (
    CONNECTION_CLOSED,
    INTERNAL_ERROR,
    REQUEST_TIMEOUT,
    CallToolRequestParams,
    CallToolResult,
    ErrorData,
    Implementation,
    InitializeResult,
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCRequest,
    JSONRPCResponse,
    LoggingMessageNotificationParams,
    ServerCapabilities,
)
from mcp_types.version import LATEST_HANDSHAKE_VERSION
from pydantic import TypeAdapter, ValidationError

# Add the parent directory to the path so we can import litellm
import litellm.experimental_mcp_client.client as mcp_client_module
from litellm.experimental_mcp_client.client import (
    MCPClient,
    _first_non_cancelled_cause,
    _TransportContext,
    as_mcp_read_timeout,
    strip_auth_scheme,
)
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
    classify_list_exception,
    list_fault_http_status,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    _format_byok_openapi_auth_header,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import StaticHeaderAuth
from litellm.types.mcp import MCPAuth, MCPStdioConfig, MCPTransport
from litellm.types.mcp_server.mcp_server_manager import MCPServer

_JSONRPC_MESSAGE_ADAPTER: Final = TypeAdapter(JSONRPCMessage)


class _MockTransportClient(MCPClient):
    """An MCPClient whose streamable-HTTP transport runs on an httpx2 MockTransport."""

    def __init__(self, respond, **kwargs):
        super().__init__(**kwargs)
        self._respond = respond

    def _create_transport_context(self) -> tuple[_TransportContext, httpx2.AsyncClient]:
        http_client: Final = self._create_httpx_client_factory(transport=httpx2.MockTransport(self._respond))(
            headers=self._get_auth_headers(), timeout=httpx2.Timeout(self.timeout)
        )
        return streamable_http_client(self.server_url, http_client=http_client), http_client


class _FakeExceptionGroup(Exception):
    """Duck-typed stand-in for an anyio/builtin ExceptionGroup.

    The production unwrapper reads ``.exceptions`` rather than depending on the
    builtin ``ExceptionGroup`` type, so this exercises the same code path on
    every Python version.
    """

    def __init__(self, message, exceptions):
        super().__init__(message)
        self.exceptions = tuple(exceptions)


class TestMCPClient:
    """Test MCP Client stdio functionality"""

    def test_mcp_client_stdio_init(self):
        """Test MCPClient initialization with stdio config"""
        stdio_config = MCPStdioConfig(command="python", args=["-m", "my_mcp_server"], env={"DEBUG": "1"})

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

        async def _noop(session):
            return None

        with pytest.raises(ValueError, match="stdio_config is required for stdio transport"):
            await client.run_with_session(_noop)

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.stdio_client")
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_mcp_client_stdio_connect_success(self, mock_session, mock_stdio_client):
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

        stdio_config = MCPStdioConfig(command="python", args=["-m", "my_mcp_server"], env={"DEBUG": "1"})

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
    async def test_mcp_client_ssl_configuration_from_env(self, mock_streamable_http_client):
        """Test that MCP client uses SSL configuration from environment variables"""
        # Setup mocks - create proper async context manager
        mock_transport = (MagicMock(), MagicMock())
        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__.return_value = mock_transport
        mock_http_ctx.__aexit__.return_value = None
        mock_streamable_http_client.return_value = mock_http_ctx

        # Mock the session
        with patch("litellm.experimental_mcp_client.client.ClientSession") as mock_session:
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
            assert isinstance(http_client, httpx2.AsyncClient)

            # Test the factory still creates a client with proper SSL config
            httpx_factory = client._create_httpx_client_factory()
            test_client = httpx_factory(headers={"test": "header"})

            assert test_client is not None
            assert isinstance(test_client, httpx2.AsyncClient)
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
        with patch("litellm.experimental_mcp_client.client.ClientSession") as mock_session:
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
            assert isinstance(test_client, httpx2.AsyncClient)
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
        with patch("litellm.experimental_mcp_client.client.ClientSession") as mock_session:
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
            assert isinstance(http_client, httpx2.AsyncClient)

            httpx_factory = client._create_httpx_client_factory()
            test_client = httpx_factory(headers={"test": "header"})

            assert test_client is not None
            assert isinstance(test_client, httpx2.AsyncClient)
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

    def test_get_auth_headers_strips_static_header_whitespace(self):
        """
        Static header names/values must be stripped of surrounding whitespace.

        h11 rejects header values with leading/trailing whitespace as an
        "Illegal header value", which silently aborts the MCP connection. A
        stray space in a configured static header value would otherwise make
        every request to that server fail with an opaque error.
        """
        client = MCPClient(
            server_url="http://example.com/mcp",
            transport_type="http",
            extra_headers={"X-Db-Url": " mew://host ", "  X-Pad  ": "v"},
        )

        headers = client._get_auth_headers()

        assert headers["X-Db-Url"] == "mew://host"
        assert headers["X-Pad"] == "v"

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
    @patch("litellm.experimental_mcp_client.client.ClientSession")
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
    @patch("litellm.experimental_mcp_client.client.ClientSession")
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


# ---------------------------------------------------------------------------
# Transport error surfacing
# ---------------------------------------------------------------------------


class TestFirstNonCancelledCause:
    """Unwrapping the real cause out of a (possibly nested) exception group."""

    def test_returns_plain_non_cancelled(self):
        err = ValueError("boom")
        assert _first_non_cancelled_cause(err) is err

    def test_returns_none_for_plain_cancelled(self):
        assert _first_non_cancelled_cause(asyncio.CancelledError()) is None

    def test_unwraps_group_to_non_cancelled_leaf(self):
        target = httpx2.ConnectError("refused")
        group = _FakeExceptionGroup("g", [asyncio.CancelledError(), target])
        assert _first_non_cancelled_cause(group) is target

    def test_unwraps_nested_group(self):
        target = httpx2.LocalProtocolError("Illegal header value")
        inner = _FakeExceptionGroup("inner", [asyncio.CancelledError(), target])
        outer = _FakeExceptionGroup("outer", [asyncio.CancelledError(), inner])
        assert _first_non_cancelled_cause(outer) is target

    def test_all_cancelled_returns_none(self):
        group = _FakeExceptionGroup("g", [asyncio.CancelledError(), asyncio.CancelledError()])
        assert _first_non_cancelled_cause(group) is None

    @pytest.mark.skipif(sys.version_info < (3, 11), reason="builtin ExceptionGroup requires 3.11+")
    def test_unwraps_builtin_exception_group(self):
        target = httpx2.ConnectError("refused")
        group = ExceptionGroup("transport failed", [target])  # noqa: F821
        assert _first_non_cancelled_cause(group) is target


class TestExecuteSessionOperationSurfacesTransportError:
    """_execute_session_operation should surface the real transport failure.

    When the upstream transport's task group fails (illegal header, connection
    refused, ...), the in-flight ``session.initialize()`` is cancelled and the
    real error only appears when the transport context exits. The opaque
    ``CancelledError`` must be replaced with that real cause.
    """

    def _make_session(self, mock_session_cls, initialize):
        mock_session = AsyncMock()
        mock_session.initialize = initialize
        session_ctx = MagicMock()
        session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session_cls.return_value = session_ctx

    def _make_transport(self, aexit_side_effect):
        transport_ctx = MagicMock()
        transport_ctx.__aenter__ = AsyncMock(return_value=(MagicMock(), MagicMock()))
        transport_ctx.__aexit__ = AsyncMock(side_effect=aexit_side_effect)
        return transport_ctx

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_surfaces_connect_error_over_cancelled(self, mock_session_cls):
        client = MCPClient(server_url="http://example.com/mcp", transport_type="http")
        self._make_session(
            mock_session_cls,
            AsyncMock(side_effect=asyncio.CancelledError("cancelled by group")),
        )
        connect_error = httpx2.ConnectError("All connection attempts failed")
        transport_ctx = self._make_transport(_FakeExceptionGroup("transport", [connect_error]))

        async def _op(session):
            return "done"

        with pytest.raises(httpx2.ConnectError):
            await client._execute_session_operation(transport_ctx, _op)

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_genuine_cancellation_is_not_replaced(self, mock_session_cls):
        client = MCPClient(server_url="http://example.com/mcp", transport_type="http")
        self._make_session(mock_session_cls, AsyncMock(side_effect=asyncio.CancelledError()))
        transport_ctx = self._make_transport(_FakeExceptionGroup("teardown", [asyncio.CancelledError()]))

        async def _op(session):
            return "done"

        with pytest.raises(asyncio.CancelledError):
            await client._execute_session_operation(transport_ctx, _op)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_phase", ("early", "late", "mixed"))
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_response_close_preserves_cancellation_and_original_errors(self, session_class, failure_phase):
        closed: Final = asyncio.Event()
        close_error: Final = httpx2.ReadError("response close failed")
        connect_error: Final = httpx2.ConnectError("another request failed before cancellation")
        cancelled: Final = asyncio.CancelledError("caller cancelled")

        class FailingCloseStream(httpx2.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                yield b"pending"

            async def aclose(self) -> None:
                closed.set()
                raise close_error

        client: Final = MCPClient(server_url="https://example.com/mcp")
        async with client._create_httpx_client_factory(
            transport=httpx2.MockTransport(lambda _: httpx2.Response(200, stream=FailingCloseStream()))
        )() as http_client:
            response: Final = await http_client.send(http_client.build_request("POST", client.server_url), stream=True)

            async def initialize():
                if failure_phase == "early":
                    await response.aclose()
                raise cancelled

            async def close_transport(*args):
                if failure_phase == "early":
                    return
                try:
                    await response.aclose()
                except httpx2.ReadError as error:
                    failures: Final = [error, connect_error] if failure_phase == "mixed" else [error]
                    raise _FakeExceptionGroup("transport", [_FakeExceptionGroup("reader", failures)])

            self._make_session(session_class, initialize)
            expected: Final = close_error if failure_phase == "early" else connect_error if failure_phase == "mixed" else cancelled
            with pytest.raises(type(expected)) as caught:
                await client._execute_session_operation(self._make_transport(close_transport), AsyncMock(), http_client)
            assert caught.value is expected
            assert closed.is_set()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_cleanup_error_after_success_is_swallowed(self, mock_session_cls):
        client = MCPClient(server_url="http://example.com/mcp", transport_type="http")
        init_result = MagicMock()
        init_result.instructions = None
        self._make_session(mock_session_cls, AsyncMock(return_value=init_result))
        transport_ctx = self._make_transport(_FakeExceptionGroup("late", [httpx2.ConnectError("late cleanup error")]))

        async def _op(session):
            return "done"

        result = await client._execute_session_operation(transport_ctx, _op)
        assert result == "done"


    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_session_entry_failure_still_closes_transport(self, session_class):
        failure: Final = RuntimeError("session dispatcher did not start")
        session_class.return_value.__aenter__ = AsyncMock(side_effect=failure)
        closed: Final = asyncio.Event()

        async def close_transport(*args):
            await anyio.lowlevel.checkpoint()
            closed.set()

        transport: Final = self._make_transport(close_transport)
        client: Final = MCPClient(server_url="https://example.com/mcp")
        with pytest.raises(RuntimeError) as caught:
            await client._execute_session_operation(transport, AsyncMock())
        assert caught.value is failure
        assert closed.is_set()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("original_error", (False, True))
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_session_exit_cancellation_preserves_original_failure(self, session_class, original_error):
        self._make_session(session_class, AsyncMock(return_value=None))
        cancelled: Final = asyncio.CancelledError("cancelled while closing session")
        session_class.return_value.__aexit__ = AsyncMock(side_effect=cancelled)
        original: Final = RuntimeError("operation failed")
        transport: Final = self._make_transport(None)
        client: Final = MCPClient(server_url="https://example.com/mcp")

        async def operation(session):
            if original_error:
                raise original
            return "done"

        with pytest.raises(RuntimeError if original_error else asyncio.CancelledError) as caught:
            await client._execute_session_operation(transport, operation)
        assert caught.value is (original if original_error else cancelled)
        transport.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("signal_type", (SystemExit, KeyboardInterrupt))
    @pytest.mark.parametrize("phase", ("session", "transport"))
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_cleanup_preserves_process_exit(self, session_class, phase, signal_type):
        self._make_session(session_class, AsyncMock(return_value=None))
        signal: Final = signal_type("process stopping")
        if phase == "session":
            session_class.return_value.__aexit__ = AsyncMock(side_effect=signal)
        transport: Final = self._make_transport(signal if phase == "transport" else None)
        client: Final = MCPClient(server_url="https://example.com/mcp")
        with pytest.raises(signal_type) as caught:
            await client._execute_session_operation(transport, AsyncMock(return_value="done"))
        assert caught.value is signal
        transport.__aexit__.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_session_and_termination_share_one_cleanup_deadline(self, session_class):
        self._make_session(session_class, AsyncMock(return_value=None))
        deleting: Final = asyncio.Event()

        async def close_session(*args):
            await anyio.sleep(1)

        async def respond(request: httpx2.Request) -> httpx2.Response:
            deleting.set()
            await anyio.sleep_forever()
            raise AssertionError("termination unexpectedly resumed")

        client: Final = MCPClient(server_url="https://example.com/mcp")
        http_client: Final = client._create_httpx_client_factory(transport=httpx2.MockTransport(respond))()
        session_class.return_value.__aexit__ = AsyncMock(side_effect=close_session)

        async def close_transport(*args):
            await http_client.delete(client.server_url)

        before: Final = anyio.current_time()
        try:
            with pytest.raises(asyncio.CancelledError):
                await client._execute_session_operation(
                    self._make_transport(close_transport), AsyncMock(return_value="completed"), http_client=http_client
                )
            assert deleting.is_set()
            assert 4.8 <= anyio.current_time() - before < 5.8
        finally:
            await http_client.aclose()


class TestMCPClientResolvedAuth:
    """A pre-resolved httpx2.Auth is attached to the upstream client's auth= slot."""

    @pytest.mark.asyncio
    async def test_resolved_auth_feeds_the_auth_slot(self):
        resolved = httpx2.Auth()
        client = MCPClient(server_url="https://upstream.example.com", resolved_auth=resolved)
        http_client = client._create_httpx_client_factory()()
        try:
            assert http_client.auth is resolved
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_resolved_auth_takes_precedence_over_aws_auth(self):
        resolved = httpx2.Auth()
        client = MCPClient(
            server_url="https://upstream.example.com",
            resolved_auth=resolved,
            aws_auth=httpx2.Auth(),
        )
        http_client = client._create_httpx_client_factory()()
        try:
            assert http_client.auth is resolved
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_without_resolved_auth_falls_back_to_aws_auth(self):
        aws = httpx2.Auth()
        client = MCPClient(server_url="https://upstream.example.com", aws_auth=aws)
        http_client = client._create_httpx_client_factory()()
        try:
            assert http_client.auth is aws
        finally:
            await http_client.aclose()


def _rendered_log_message(call):
    message = str(call.args[0])
    values = call.args[1:]
    return message % values if values else message


def _all_logged_messages(mock_logger):
    return " ".join(
        _rendered_log_message(call)
        for level in ("info", "debug", "warning", "error", "exception")
        for call in getattr(mock_logger, level).call_args_list
        if call.args
    )


@pytest.mark.asyncio
async def test_call_tool_does_not_log_arguments():
    from mcp.types import CallToolRequestParams

    secret = "ssn-123-45-6789"
    client = MCPClient(server_url="http://test-server")
    client.run_with_session = AsyncMock(return_value=MagicMock())
    params = CallToolRequestParams(name="search_tool", arguments={"input": secret, "model": "gpt-5-mini"})

    with patch.object(mcp_client_module, "verbose_logger") as mock_logger:
        await client.call_tool(params)

    logged = _all_logged_messages(mock_logger)
    assert "search_tool" in logged
    assert secret not in logged
    assert "gpt-5-mini" not in logged


@pytest.mark.asyncio
async def test_get_prompt_does_not_log_arguments():
    from mcp.types import GetPromptRequestParams

    secret = "ssn-987-65-4321"
    client = MCPClient(server_url="http://test-server")
    client.run_with_session = AsyncMock(return_value=MagicMock())
    params = GetPromptRequestParams(name="my_prompt", arguments={"input": secret})

    with patch.object(mcp_client_module, "verbose_logger") as mock_logger:
        await client.get_prompt(params)

    logged = _all_logged_messages(mock_logger)
    assert "my_prompt" in logged
    assert secret not in logged


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.mark.asyncio
async def test_call_tool_raise_on_error_logs_at_debug_not_error():
    """When the caller opts into raise_on_error it owns the exception and logs it at the fitting
    level (an expected pass-through re-auth 401 is info, not error). call_tool must therefore not emit
    its own error-level line in that mode, so error-rate alerts do not trip on the expected signal;
    the swallow path (raise_on_error=False) still logs at error since nothing downstream will."""
    from mcp.types import CallToolRequestParams

    client = MCPClient(transport_type=MCPTransport.stdio)
    boom = RuntimeError("upstream boom")

    async def _raise(_operation, **_kwargs):
        raise boom

    params = CallToolRequestParams(name="t", arguments={})

    with patch.object(client, "run_with_session", side_effect=_raise) as mock_rws:
        with patch.object(mcp_client_module, "verbose_logger") as mock_log:
            with pytest.raises(RuntimeError):
                await client.call_tool(params, raise_on_error=True)
            assert not mock_log.error.called, "raise_on_error path must not log at error"
            debug_msgs = [str(c.args[0]) for c in mock_log.debug.call_args_list if c.args]
            assert any("call_tool failed" in m for m in debug_msgs), "the demoted failure line must go to debug"
            assert mock_rws.call_args.kwargs.get("quiet_on_error") is True, (
                "call_tool must forward quiet_on_error so run_with_session also demotes its own failure line"
            )

    with patch.object(client, "run_with_session", side_effect=_raise):
        with patch.object(mcp_client_module, "verbose_logger") as mock_log:
            result = await client.call_tool(params, raise_on_error=False)
            assert result.is_error is True
            assert mock_log.error.called, "swallow path must keep error-level visibility"


@pytest.mark.asyncio
async def test_list_tools_raise_on_error_logs_at_debug_not_error():
    """list_tools must mirror call_tool: when the caller opts into raise_on_error it owns the
    exception, so an expected pass-through re-auth 401 does not emit an error/exception line that
    would trip error-rate alerts. The swallow path still logs the full exception."""
    client = MCPClient(transport_type=MCPTransport.stdio)
    boom = RuntimeError("upstream boom")

    async def _raise(_operation, **_kwargs):
        raise boom

    with patch.object(client, "run_with_session", side_effect=_raise) as mock_rws:
        with patch.object(mcp_client_module, "verbose_logger") as mock_log:
            with pytest.raises(RuntimeError):
                await client.list_tools(raise_on_error=True)
            assert not mock_log.error.called, "raise_on_error path must not log at error"
            assert not mock_log.exception.called, "raise_on_error path must not log a traceback"
            debug_msgs = [str(c.args[0]) for c in mock_log.debug.call_args_list if c.args]
            assert any("list_tools failed" in m for m in debug_msgs), "the demoted failure line must go to debug"
            assert mock_rws.call_args.kwargs.get("quiet_on_error") is True, (
                "list_tools must forward quiet_on_error so run_with_session also demotes its own failure line"
            )

    with patch.object(client, "run_with_session", side_effect=_raise):
        with patch.object(mcp_client_module, "verbose_logger") as mock_log:
            result = await client.list_tools(raise_on_error=False)
            assert result == []
            assert mock_log.exception.called, "swallow path must keep full exception visibility"


@pytest.mark.asyncio
async def test_run_with_session_quiet_on_error_demotes_warning_to_debug():
    """run_with_session logs its failure at warning by default (an operator signal for an unexpected
    outage), but when the caller owns the exception (quiet_on_error=True, set by call_tool / list_tools
    under raise_on_error) it must demote that line to debug so an expected pass-through re-auth does not
    emit a warning per call."""
    client = MCPClient(transport_type=MCPTransport.stdio)
    boom = RuntimeError("session boom")

    async def _op(_session):
        raise boom

    async def _fake_exec(_transport_ctx, _operation, http_client=None):
        raise boom

    with patch.object(client, "_create_transport_context", return_value=(object(), None)):
        with patch.object(client, "_execute_session_operation", side_effect=_fake_exec):
            with patch.object(mcp_client_module, "verbose_logger") as mock_log:
                with pytest.raises(RuntimeError):
                    await client.run_with_session(_op, quiet_on_error=True)
                assert not mock_log.warning.called, "quiet_on_error must not emit a warning"
                debug_msgs = [str(c.args[0]) for c in mock_log.debug.call_args_list if c.args]
                assert any("run_with_session failed" in m for m in debug_msgs), "the failure line must go to debug"

            with patch.object(mcp_client_module, "verbose_logger") as mock_log:
                with pytest.raises(RuntimeError):
                    await client.run_with_session(_op)
                warning_msgs = [str(c.args[0]) for c in mock_log.warning.call_args_list if c.args]
                assert any("run_with_session failed" in m for m in warning_msgs), (
                    "the default path must keep the operator-visible warning"
                )


class _ScriptedUpstream:
    """An in-memory MCP upstream that answers ``initialize`` and then follows one script for
    ``tools/list``.

    ``answer=None`` ends the response stream without a JSON-RPC reply, which is what a
    streamable-HTTP upstream does when its SSE stream closes early: the SDK drops the message and
    the request is never resolved and never fails. Anything else is sent back as that JSON-RPC
    error, the shape an upstream application uses to report its own failure.
    """

    def __init__(self, tools_list_error: ErrorData | None = None):
        self._tools_list_error = tools_list_error
        self._to_client_tx, self._to_client_rx = anyio.create_memory_object_stream(10)
        self._from_client_tx, self._from_client_rx = anyio.create_memory_object_stream(10)
        self._task_group = None

    async def __aenter__(self):
        self._task_group = anyio.create_task_group()
        await self._task_group.__aenter__()
        self._task_group.start_soon(self._serve)
        return self._to_client_rx, self._from_client_tx

    async def __aexit__(self, *_exc_info):
        self._task_group.cancel_scope.cancel()
        return await self._task_group.__aexit__(None, None, None)

    async def _send(self, message):
        await self._to_client_tx.send(SessionMessage(message))

    async def _serve(self):
        async for session_message in self._from_client_rx:
            request = session_message.message
            method = getattr(request, "method", None)
            if method == "initialize":
                result = InitializeResult(
                    protocolVersion=LATEST_HANDSHAKE_VERSION,
                    capabilities=ServerCapabilities(),
                    serverInfo=Implementation(name="scripted-upstream", version="1.0.0"),
                )
                await self._send(
                    JSONRPCResponse(
                        jsonrpc="2.0",
                        id=request.id,
                        result=result.model_dump(by_alias=True, mode="json", exclude_none=True),
                    )
                )
            elif method == "tools/list" and self._tools_list_error is not None:
                await self._send(JSONRPCError(jsonrpc="2.0", id=request.id, error=self._tools_list_error))


class _ScriptedClient(MCPClient):
    """An MCPClient whose transport is a scripted in-memory upstream instead of a real connection,
    so the real ``ClientSession`` and its real timeout machinery are what run."""

    def __init__(self, *, timeout: float, tools_list_error: ErrorData | None = None):
        super().__init__(server_url="http://upstream.local/mcp", timeout=timeout)
        self._upstream = _ScriptedUpstream(tools_list_error=tools_list_error)

    def _create_transport_context(self):
        return self._upstream, None


@pytest.mark.asyncio
async def test_list_tools_fails_on_its_own_timeout_when_the_upstream_never_answers():
    """An upstream that accepts the request and never answers must fail the client's own timeout.

    Without a session read timeout the request waits forever, so discovery only ends when an outer
    cancel scope kills it. That is the reported symptom: a cancelled list_tools, no tools, and a
    fault that blames the gateway. The outer guard here is 20x the client timeout, so a run that
    reaches it proves nothing bounded the request.

    The classification is asserted here, off a real ``ClientSession`` running its real read timeout,
    rather than off a hand-built exception. A hand-built fixture encodes what we currently believe
    the SDK raises and would keep passing after the SDK stopped raising it, at which point the
    translation would quietly stop matching and the fault would silently downgrade to ``internal``.
    Driving the real path makes an SDK bump that breaks the discriminator fail loudly instead.
    """
    client = _ScriptedClient(timeout=0.5)

    started = asyncio.get_running_loop().time()
    with pytest.raises(TimeoutError) as exc_info:
        await asyncio.wait_for(client.list_tools(raise_on_error=True), timeout=10)
    elapsed = asyncio.get_running_loop().time() - started

    assert elapsed < 5, f"the request must end on the client's own 0.5s timeout, took {elapsed:.2f}s"

    fault = classify_list_exception(exc_info.value)
    assert fault.tag == "timeout", "an upstream that stopped answering must not be classified as the gateway's fault"
    assert list_fault_http_status(fault) == 504


@pytest.mark.asyncio
async def test_upstream_json_rpc_error_408_is_not_reported_as_a_client_timeout():
    """The SDK reports its own elapsed read timeout and relays an upstream JSON-RPC error through
    the same exception class and the same numeric field, and JSON-RPC error codes are a different
    namespace from HTTP status codes. An upstream answering with application code 408 must keep
    travelling as ``MCPError`` so it is never blamed on the gateway as a 504.

    This is the other half of the pair: the same real transport and the same real session, so one
    mechanism pins both directions.
    """
    client = _ScriptedClient(
        timeout=30,
        tools_list_error=ErrorData(code=REQUEST_TIMEOUT, message="re-authenticate and retry"),
    )

    with pytest.raises(MCPError) as exc_info:
        await asyncio.wait_for(client.list_tools(raise_on_error=True), timeout=10)

    assert not isinstance(exc_info.value, TimeoutError), "an upstream application error is not a gateway timeout"
    assert exc_info.value.error.code == REQUEST_TIMEOUT

    fault = classify_list_exception(exc_info.value)
    assert fault.tag != "timeout", "an upstream's own application error must never be reported as a gateway timeout"
    assert list_fault_http_status(fault) != 504


def _raise_mcp_error_while_handling_a_timeout(code: int, message: str) -> MCPError:
    """An ``MCPError`` carrying the context chain it would have if it were raised while a
    ``TimeoutError`` was in flight, which is how the SDK raises its own read timeout."""
    try:
        try:
            raise TimeoutError()
        except TimeoutError:
            raise MCPError(code=code, message=message)
    except MCPError as raised:
        return raised


def test_as_mcp_read_timeout_separates_the_sdk_timeout_from_a_relayed_upstream_error():
    """Neither signal alone is enough. The code alone cannot separate the SDK's own timeout from an
    upstream JSON-RPC error that happens to use 408, and the context chain alone cannot separate it
    from any other relayed error that surfaces while a timeout is being handled, so both must hold.
    """
    timeout_code = REQUEST_TIMEOUT

    translated = as_mcp_read_timeout(_raise_mcp_error_while_handling_a_timeout(timeout_code, "Timed out while waiting"))
    assert isinstance(translated, TimeoutError)
    assert str(translated) == "Timed out while waiting"

    relayed_408 = MCPError(code=timeout_code, message="upstream said 408")
    assert as_mcp_read_timeout(relayed_408) is None, "an upstream 408 with no elapsed timeout is not our timeout"

    relayed_other = _raise_mcp_error_while_handling_a_timeout(-32603, "upstream internal error")
    assert as_mcp_read_timeout(relayed_other) is None, "a non-timeout code is not our timeout, whatever the chain"

    assert as_mcp_read_timeout(MCPError(code=-32603, message="boom")) is None
    assert as_mcp_read_timeout(RuntimeError("not an MCPError")) is None


@pytest.mark.asyncio
async def test_read_timeout_logs_an_actionable_line_that_quiet_on_error_cannot_demote():
    """The reported failure surfaced only as "MCP Client list_tools was cancelled", which names
    neither the server nor the elapsed budget. An upstream that stops answering is always
    operator-actionable, so this line stays at warning even for callers that own the exception."""
    client = _ScriptedClient(timeout=0.5)

    with patch.object(mcp_client_module, "verbose_logger") as mock_log:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(client.list_tools(raise_on_error=True), timeout=10)

    warnings = [str(call.args[0]) % tuple(call.args[1:]) for call in mock_log.warning.call_args_list if call.args]
    timeout_lines = [line for line in warnings if "timed out after" in line]
    assert timeout_lines, f"expected an actionable timeout warning, got {warnings}"
    assert "http://upstream.local/mcp" in timeout_lines[0], "the line must name the server that stopped answering"
    assert "0.5s" in timeout_lines[0], "the line must name the budget that elapsed"


class TestAuthSchemeNormalization:
    """MCP egress must emit exactly one authorization scheme.

    Callers supply both a bare credential and a complete header value (the latter whenever it is
    passed through from ``x-mcp-auth`` / ``Authorization``), and the second shape used to be given
    a second scheme, which upstream servers reject as a malformed token.
    """

    @pytest.mark.parametrize(
        "auth_type, auth_value",
        [
            (MCPAuth.bearer_token, "bare-token"),
            (MCPAuth.bearer_token, "Bearer bare-token"),
            (MCPAuth.bearer_token, "bearer bare-token"),
            (MCPAuth.bearer_token, "  BEARER   bare-token"),
            (MCPAuth.oauth2, "bare-token"),
            (MCPAuth.oauth2, "Bearer bare-token"),
            (MCPAuth.oauth2_token_exchange, "bare-token"),
            (MCPAuth.oauth2_token_exchange, "Bearer bare-token"),
        ],
    )
    def test_bearer_family_emits_exactly_one_scheme(self, auth_type, auth_value):
        client = MCPClient(server_url="http://example.com/mcp", auth_type=auth_type, auth_value=auth_value)

        assert client._get_auth_headers()["Authorization"] == "Bearer bare-token"

    @pytest.mark.parametrize("auth_value", ["bare-token", "token bare-token", "TOKEN bare-token"])
    def test_token_scheme_emits_exactly_one_scheme(self, auth_value):
        client = MCPClient(server_url="http://example.com/mcp", auth_type=MCPAuth.token, auth_value=auth_value)

        assert client._get_auth_headers()["Authorization"] == "token bare-token"

    @pytest.mark.parametrize(
        "auth_type, auth_value",
        [
            (MCPAuth.bearer_token, "Bearertoken"),
            (MCPAuth.oauth2, "Bearer.eyJzdWIiOiJhYmMifQ.sig"),
            (MCPAuth.token, "tokenish"),
        ],
    )
    def test_a_credential_merely_starting_with_the_scheme_text_is_left_intact(self, auth_type, auth_value):
        """RFC 7235 requires whitespace between scheme and credential, so a token whose first
        characters happen to spell the scheme is a credential, not a schemed value."""
        client = MCPClient(server_url="http://example.com/mcp", auth_type=auth_type, auth_value=auth_value)

        scheme = "token" if auth_type == MCPAuth.token else "Bearer"
        assert client._get_auth_headers()["Authorization"] == f"{scheme} {auth_value}"

    @pytest.mark.parametrize(
        "auth_type, auth_value, expected",
        [
            (MCPAuth.bearer_token, "Bearer ", "Bearer Bearer"),
            (MCPAuth.bearer_token, "Bearer    ", "Bearer Bearer"),
        ],
    )
    def test_a_scheme_with_no_credential_behind_it_still_produces_a_header(self, auth_type, auth_value, expected):
        """Treating this as a schemed value would leave nothing to send, and a request with no
        Authorization at all is harder to diagnose upstream than a visibly wrong one."""
        client = MCPClient(server_url="http://example.com/mcp", auth_type=auth_type, auth_value=auth_value)

        assert client._get_auth_headers()["Authorization"] == expected

    def test_basic_with_a_scheme_and_no_credential_still_produces_a_header(self):
        client = MCPClient(server_url="http://example.com/mcp", auth_type=MCPAuth.basic, auth_value="Basic ")

        assert "Authorization" in client._get_auth_headers()

    def test_basic_accepts_an_already_encoded_schemed_value_without_re_encoding_it(self):
        """Stripping the scheme at header-build time cannot fix this shape: ``to_basic_auth`` has by
        then encoded the whole ``Basic ...`` string, leaving no prefix to find."""
        encoded = base64.b64encode(b"user:pass").decode()

        client = MCPClient(
            server_url="http://example.com/mcp",
            auth_type=MCPAuth.basic,
            auth_value=f"Basic {encoded}",
        )

        header = client._get_auth_headers()["Authorization"]
        assert header == f"Basic {encoded}"
        assert base64.b64decode(header.split(" ", 1)[1]) == b"user:pass"

    @pytest.mark.parametrize("auth_value", ["user:pass", "Basic user:pass", "basic user:pass"])
    def test_basic_always_emits_encoded_credentials(self, auth_value):
        """A schemed value whose remainder is raw rather than encoded is still a username/password
        pair, so it is encoded rather than forwarded as an invalid RFC 7617 header."""
        client = MCPClient(server_url="http://example.com/mcp", auth_type=MCPAuth.basic, auth_value=auth_value)

        header = client._get_auth_headers()["Authorization"]
        assert base64.b64decode(header.split(" ", 1)[1]) == b"user:pass"

    def test_authorization_auth_type_is_passed_through_verbatim(self):
        """``MCPAuth.authorization`` means the caller owns the whole header value."""
        client = MCPClient(
            server_url="http://example.com/mcp",
            auth_type=MCPAuth.authorization,
            auth_value="Bearer Bearer deliberately-doubled",
        )

        assert client._get_auth_headers()["Authorization"] == "Bearer Bearer deliberately-doubled"

    def test_api_key_credential_is_not_treated_as_a_schemed_value(self):
        client = MCPClient(
            server_url="http://example.com/mcp",
            auth_type=MCPAuth.api_key,
            auth_value="Bearer looks-schemed",
        )

        assert client._get_auth_headers()["X-API-Key"] == "Bearer looks-schemed"


@pytest.mark.parametrize(
    "auth_value, scheme, expected",
    [
        ("Bearer abc", "Bearer", "abc"),
        ("bearer abc", "Bearer", "abc"),
        ("Bearer\tabc", "Bearer", "abc"),
        ("bearer\t\tabc", "Bearer", "abc"),
        ("  Bearer   abc  ", "Bearer", "abc  "),
        ("abc", "Bearer", "abc"),
        ("Bearerabc", "Bearer", "Bearerabc"),
        ("Basic abc", "Bearer", "Basic abc"),
        ("token abc", "token", "abc"),
        ("Basic abc", "Basic", "abc"),
        ("Bearer ", "Bearer", "Bearer "),
        ("Bearer   ", "Bearer", "Bearer   "),
    ],
)
def test_strip_auth_scheme(auth_value, scheme, expected):
    assert strip_auth_scheme(auth_value, scheme) == expected


@pytest.mark.parametrize(
    "auth_type, auth_value, expected",
    [
        (MCPAuth.bearer_token, "Bearer jwt", "Bearer jwt"),
        (MCPAuth.bearer_token, "jwt", "Bearer jwt"),
        (MCPAuth.api_key, "ApiKey secret", "ApiKey secret"),
        (MCPAuth.api_key, "secret", "ApiKey secret"),
        (MCPAuth.basic, "Basic dXNlcjpwYXNz", "Basic dXNlcjpwYXNz"),
    ],
)
def test_openapi_byok_auth_header_emits_exactly_one_scheme(auth_type, auth_value, expected):
    """A non-BYOK server short-circuits ``_resolve_byok_mcp_auth_header``, so this formatter also
    receives the deprecated global ``x-mcp-auth``, which is already a complete header value."""
    server = MCPServer(
        server_id="s1",
        name="openapi-server",
        url="http://example.com/mcp",
        transport=MCPTransport.http,
        auth_type=auth_type,
        spec_path="/tmp/spec.json",
    )

    assert server.is_byok is False
    assert _format_byok_openapi_auth_header(server, auth_value) == expected


def test_mcp_extra_matches_proxy_extra_and_supports_streamable_http():
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    from packaging.requirements import Requirement

    pyproject_path = Path(__file__).parents[3] / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        project = tomllib.load(f)
    extras = project["project"]["optional-dependencies"]

    sdk2_names: Final = frozenset(("mcp", "httpx2", "pydantic"))
    mcp_extra: Final = {Requirement(req).name: req for req in extras["mcp"]}
    assert mcp_extra == {
        name: req
        for req in extras["proxy"]
        if (name := Requirement(req).name) in sdk2_names
    }

    specifier: Final = Requirement(mcp_extra["mcp"]).specifier
    assert not specifier.contains("1.28.1")
    assert specifier.contains("2.2.0")
    with (pyproject_path.parent / "uv.lock").open("rb") as f:
        locked = tomllib.load(f)
    mcp_versions: Final = [package["version"] for package in locked["package"] if package["name"] == "mcp"]
    assert len(mcp_versions) == 1
    assert specifier.contains(mcp_versions[0])


@pytest.mark.parametrize("module", ["mcp", "mcp_types", "httpx2", "httpcore2"])
def test_base_sdk_guard_rejects_mcp_dependencies(tmp_path: Path, module: str) -> None:
    import subprocess
    import sys

    (tmp_path / f"{module}.py").write_text("")
    checker = Path(__file__).parents[2] / "base_sdk_tests" / "check_base_sdk_install.py"
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import runpy, sys; sys.path.insert(0, sys.argv[2]); "
            "runpy.run_path(sys.argv[1])['check_environment_is_base_only']()",
            str(checker),
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0, f"base-only guard accepted installed {module}"
    assert f"{module} installed" in result.stderr


@pytest.mark.parametrize(
    "auth_type, default_header",
    [
        (MCPAuth.oauth2, "Authorization"),
        (MCPAuth.bearer_token, "Authorization"),
        (MCPAuth.api_key, "X-API-Key"),
    ],
)
def test_v1_auth_headers_default_to_the_auth_type_slot(auth_type: MCPAuth, default_header: str) -> None:
    client = MCPClient(server_url="http://up.example.com/mcp", auth_type=auth_type)
    client.update_auth_value("tok")
    assert default_header in client._get_auth_headers()


@pytest.mark.parametrize("auth_type", [MCPAuth.oauth2, MCPAuth.bearer_token, MCPAuth.api_key])
def test_v1_auth_headers_honor_the_configured_slot(auth_type: MCPAuth) -> None:
    """The v1 stack mints its own client_credentials token (oauth2_token_cache) and writes it here,
    so leaving this table hardcoded makes the knob a silent no-op for every server that resolves
    through v1 rather than the v2 resolver."""
    client = MCPClient(
        server_url="http://up.example.com/mcp",
        auth_type=auth_type,
        auth_header_name="esb-oauth",
    )
    client.update_auth_value("tok")
    headers = client._get_auth_headers()
    assert "esb-oauth" in headers
    assert "Authorization" not in headers
    assert "X-API-Key" not in headers


def test_v1_static_headers_still_win_their_own_slot():
    # extra_headers (which carries static_headers) is applied last on the v1 path, so a static
    # Authorization survives untouched while the resolved credential sits on its own header.
    client = MCPClient(
        server_url="http://up.example.com/mcp",
        auth_type=MCPAuth.oauth2,
        auth_header_name="esb-oauth",
        extra_headers={"Authorization": "Bearer static-upstream-mcp-token"},
    )
    client.update_auth_value("minted")
    headers = client._get_auth_headers()
    assert headers["esb-oauth"] == "Bearer minted"
    assert headers["Authorization"] == "Bearer static-upstream-mcp-token"


@pytest.mark.asyncio
async def test_sdk_same_origin_redirect_lists_and_calls_tools() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.url.path == "/mcp":
            return httpx2.Response(307, headers={"Location": "/final/mcp"})
        assert request.url == "https://upstream.example.com/final/mcp"
        assert request.headers["x-upstream-token"] == "Bearer synthetic-token"
        if request.method != "POST":
            return httpx2.Response(405)
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        match payload.method:
            case "initialize":
                return httpx2.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload.id,
                        "result": {
                            "protocolVersion": LATEST_HANDSHAKE_VERSION,
                            "capabilities": {"tools": {}},
                            "serverInfo": {"name": "redirect-test", "version": "1"},
                        },
                    },
                )
            case "tools/list":
                return httpx2.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload.id,
                        "result": {"tools": [{"name": "add", "inputSchema": {"type": "object"}}]},
                    },
                )
            case "tools/call":
                assert payload.params is not None
                assert payload.params["name"] == "add"
                assert payload.params["arguments"] == {"a": 2, "b": 3}
                return httpx2.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload.id,
                        "result": {"content": [{"type": "text", "text": "5"}], "isError": False},
                    },
                )
            case _:
                pytest.fail(f"Unexpected MCP request: {payload.method}")

    responder: Final = Mock(side_effect=respond)
    client: Final = _MockTransportClient(
        responder,
        server_url="https://upstream.example.com/mcp",
        auth_type=MCPAuth.bearer_token,
        auth_value="synthetic-token",
        auth_header_name="x-upstream-token",
        timeout=5,
    )
    with anyio.fail_after(10):
        tools: Final = await client.list_tools(raise_on_error=True)
        result: Final = await client.call_tool(
            CallToolRequestParams(name="add", arguments={"a": 2, "b": 3}), raise_on_error=True
        )
    assert [tool.name for tool in tools] == ["add"]
    assert result.is_error is False
    assert len(result.content) == 1
    assert result.content[0].type == "text"
    assert result.content[0].text == "5"
    assert any(call.args[0].url.path == "/mcp" for call in responder.call_args_list)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("list", "call"))
async def test_sdk_cross_origin_redirect_never_contacts_destination(operation: str) -> None:
    responder: Final = Mock(
        return_value=httpx2.Response(307, headers={"Location": "https://destination.example.com/mcp"})
    )
    client: Final = _MockTransportClient(
        responder,
        server_url="https://upstream.example.com/mcp",
        auth_type=MCPAuth.bearer_token,
        auth_value="synthetic-token",
        auth_header_name="x-upstream-token",
        timeout=5,
    )
    pending_operation: Final = (
        client.list_tools(raise_on_error=True)
        if operation == "list"
        else client.call_tool(CallToolRequestParams(name="add", arguments={"a": 2, "b": 3}), raise_on_error=True)
    )
    with anyio.fail_after(10), pytest.raises(MCPError):
        await pending_operation
    assert responder.call_count == 1
    request: Final = responder.call_args.args[0]
    assert request.method == "POST"
    assert request.url == "https://upstream.example.com/mcp"
    assert request.headers["x-upstream-token"] == "Bearer synthetic-token"
    assert all(call.args[0].url.host != "destination.example.com" for call in responder.call_args_list)


@pytest.mark.asyncio
async def test_a_custom_credential_header_is_stripped_when_a_redirect_crosses_origin():
    """httpx drops Authorization across origins but keeps every other header, so a credential the
    operator moved to its own slot would be replayed to whatever host the upstream redirects to.
    Verified against real httpx redirect handling, not a hand-built request.
    """
    seen: list[tuple[str, str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append((request.url.host, request.headers.get("esb-oauth", "<stripped>")))
        if request.url.host == "upstream.example.com":
            return httpx2.Response(302, headers={"Location": "https://attacker.example.com/collect"})
        return httpx2.Response(200)

    client = MCPClient(
        server_url="https://upstream.example.com/mcp",
        auth_type=MCPAuth.oauth2,
        auth_header_name="esb-oauth",
    )
    client.update_auth_value("minted-token")
    factory = client._create_httpx_client_factory()
    async with factory(headers=client._get_auth_headers(), timeout=None) as http_client:
        http_client._transport = httpx2.MockTransport(handler)
        await http_client.get("https://upstream.example.com/mcp")

    assert seen[0] == ("upstream.example.com", "Bearer minted-token")
    assert seen[1] == ("attacker.example.com", "<stripped>")


@pytest.mark.asyncio
async def test_authorization_is_left_to_httpx_and_needs_no_guard():
    # The default slot is already protected by httpx, so the client must not install a guard for it
    # and must not interfere with the ordinary Authorization path.
    url = "https://upstream.example.com/mcp"
    from litellm.types.mcp import credential_redirect_hook

    def guard_for(client: MCPClient):
        return credential_redirect_hook(client.server_url, client._credential_slot)

    assert guard_for(MCPClient(server_url=url, auth_type=MCPAuth.oauth2)) is None
    assert guard_for(MCPClient(server_url=url, resolved_auth=StaticHeaderAuth("Bearer x"))) is None
    # a v2 resolver slot is discovered from the auth object, without the caller naming it again
    custom = MCPClient(server_url=url, resolved_auth=StaticHeaderAuth("Bearer x", header_name="esb-oauth"))
    assert guard_for(custom) is not None
    # and the same answer arrives via the v1 configured slot
    assert guard_for(MCPClient(server_url=url, auth_header_name="ESB-OAuth")) is not None


def test_an_injected_header_cannot_shadow_the_configured_credential_slot():
    """The v2 path drops a colliding injected header so the resolved credential wins its slot. The
    v1 path applies extra_headers last, so without this it silently sends the injected value and the
    upstream rejects a credential the gateway thought it had sent.
    """
    client = MCPClient(
        server_url="https://upstream.example.com/mcp",
        auth_type=MCPAuth.oauth2,
        auth_header_name="esb-oauth",
        extra_headers={"esb-oauth": "Bearer injected", "X-Trace": "keep"},
    )
    client.update_auth_value("minted-token")
    headers = client._get_auth_headers()
    assert headers["esb-oauth"] == "Bearer minted-token"
    assert headers["X-Trace"] == "keep"


def test_without_a_configured_slot_the_existing_precedence_is_unchanged():
    # extra_headers winning over authentication_token is long-standing v1 behavior; the fix above
    # must apply only to the slot the operator explicitly named.
    client = MCPClient(
        server_url="https://upstream.example.com/mcp",
        auth_type=MCPAuth.oauth2,
        extra_headers={"Authorization": "Bearer injected"},
    )
    client.update_auth_value("minted-token")
    assert client._get_auth_headers()["Authorization"] == "Bearer injected"


_REDIRECT_CASES = [
    ("https://upstream.example.com/mcp", "https://upstream.example.com/other"),  # same origin
    ("https://upstream.example.com/mcp", "https://upstream.example.com:443/other"),  # explicit default port
    ("https://upstream.example.com/mcp", "https://attacker.example.com/collect"),  # different host
    ("https://upstream.example.com/mcp", "http://upstream.example.com/collect"),  # scheme downgrade
    ("https://upstream.example.com/mcp", "https://upstream.example.com:8443/other"),  # different port
    ("https://upstream.example.com/mcp", "https://sub.upstream.example.com/x"),  # different host
    ("http://upstream.example.com/mcp", "https://upstream.example.com/other"),  # http -> https upgrade
    ("http://upstream.example.com/mcp", "http://upstream.example.com/other"),  # same origin, plain http
]


@pytest.mark.parametrize("start,target", _REDIRECT_CASES)
@pytest.mark.asyncio
async def test_the_guard_agrees_with_httpx_about_authorization(start: str, target: str) -> None:
    """Our custom slot must be dropped on exactly the redirects where httpx drops Authorization.

    The rule is mirrored rather than imported, so this drives real httpx and compares the two
    outcomes. A future httpx that changes its redirect rule reds here instead of silently leaving
    the custom slot forwarded where Authorization is not (or stripped where it is not needed).
    """
    seen: list[tuple[str, str, str]] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(
            (
                str(request.url),
                request.headers.get("authorization", "<stripped>"),
                request.headers.get("esb-oauth", "<stripped>"),
            )
        )
        if str(request.url) == start:
            return httpx2.Response(302, headers={"Location": target})
        return httpx2.Response(200)

    client = MCPClient(server_url=start, auth_type=MCPAuth.oauth2, auth_header_name="esb-oauth")
    factory = client._create_httpx_client_factory()
    async with factory(headers={"Authorization": "Bearer AUTH", "esb-oauth": "Bearer ESB"}, timeout=None) as http:
        http._transport = httpx2.MockTransport(handler)
        await http.get(start)

    _url, authorization, esb = seen[-1]
    assert (authorization == "<stripped>") == (esb == "<stripped>"), (
        f"httpx and the guard disagree for {target}: authorization={authorization!r} esb-oauth={esb!r}"
    )


def test_a_differently_cased_injected_header_cannot_shadow_the_slot() -> None:
    # HTTP header names are case-insensitive and v2 drops the collision case-insensitively, so an
    # exact-key check here would leave both spellings in the dict and let the injected value win.
    client = MCPClient(
        server_url="https://upstream.example.com/mcp",
        auth_type=MCPAuth.oauth2,
        auth_header_name="esb-oauth",
        extra_headers={"ESB-OAuth": "Bearer injected", "X-Trace": "keep"},
    )
    client.update_auth_value("minted-token")
    headers = client._get_auth_headers()
    assert [v for k, v in headers.items() if k.lower() == "esb-oauth"] == ["Bearer minted-token"]
    assert headers["X-Trace"] == "keep"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "body", "expected_type"),
    [
        ("text/html", b"<html>secret-page</html>", MCPError),
        ("application/json", b"secret-invalid-json", MCPError),
        ("application/json", b"", MCPError),
        ("application/json", b'{"secret":"invalid-rpc"}', MCPError),
        ("application/json", b'{"jsonrpc":"2.0","id":0}', MCPError),
        ("application/json", b'{"jsonrpc":"2.0","id":0,"result":{"secret":"bad-schema"}}', ValidationError),
    ],
)
async def test_invalid_http_response_surfaces_without_waiting_for_timeout(
    content_type: str, body: bytes, expected_type: type[Exception]
) -> None:
    from litellm.proxy._experimental.mcp_server.rest_endpoints import _connection_error_message

    def respond(request: httpx2.Request) -> httpx2.Response:
        if expected_type is ValidationError:
            return httpx2.Response(200, json={**json.loads(body), "id": json.loads(request.content)["id"]})
        return httpx2.Response(200, headers={"Content-Type": content_type}, content=body)

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http_client:
        client: Final = MCPClient(server_url="https://example.com/mcp", timeout=30)
        with pytest.raises(expected_type) as caught:
            await asyncio.wait_for(
                client._execute_session_operation(
                    streamable_http_client(client.server_url, http_client=http_client),
                    lambda session: session.list_tools(),
                ),
                timeout=3,
            )

    message: Final = _connection_error_message(caught.value, client.server_url, 30)
    assert "unsupported content type" in message or "invalid MCP response" in message
    assert "secret" not in message
    assert "timed out" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [200, 401, 403, 429, 503])
async def test_http_response_handler_preserves_success_and_http_errors(status_code: int) -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "DELETE":
            return httpx2.Response(200)
        payload: Final = json.loads(request.content)
        if "id" not in payload:
            return httpx2.Response(202)
        result: Final = (
            {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {},
                "serverInfo": {"name": "test", "version": "1"},
            }
            if payload["method"] == "initialize"
            else {"tools": []}
        )
        return httpx2.Response(status_code, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    client: Final = MCPClient(server_url="https://example.com/mcp", timeout=30)
    async with client._create_httpx_client_factory(transport=httpx2.MockTransport(respond))() as http_client:
        operation: Final = client._execute_session_operation(
            streamable_http_client(client.server_url, http_client=http_client),
            lambda session: session.list_tools(),
            http_client=http_client,
        )
        if status_code == 200:
            result: Final = await asyncio.wait_for(operation, timeout=3)
            assert result.tools == []
        else:
            with pytest.raises(httpx2.HTTPStatusError) as caught:
                await asyncio.wait_for(operation, timeout=3)
            assert caught.value.response.status_code == status_code


@pytest.mark.asyncio
async def test_http_status_check_allows_auth_refresh_before_rejecting() -> None:
    from litellm.proxy._experimental.mcp_server.outbound_credentials.client_credentials import ClientCredentialsBearerAuth

    seen = []

    async def refresh(failed):
        assert failed == "stale"
        return "fresh"

    def respond(request):
        seen.append(request.headers["authorization"])
        return httpx2.Response(401 if len(seen) == 1 else 200, json={"ok": True})

    from litellm.proxy._experimental.mcp_server.outbound_credentials.types import ClientCredentialsConfig

    auth = ClientCredentialsBearerAuth("stale", refresh, ClientCredentialsConfig())
    client = MCPClient(server_url="https://example.com/mcp", resolved_auth=auth)
    async with client._create_httpx_client_factory(transport=httpx2.MockTransport(respond))() as http_client:
        response = await http_client.post(client.server_url, json={"method": "tools/list"})
        assert response.status_code == 200
    assert seen == ["Bearer stale", "Bearer fresh"]


@pytest.mark.asyncio
async def test_http_response_handler_preserves_notifications_and_tool_listing() -> None:
    notification: Final = {
        "jsonrpc": "2.0",
        "method": "notifications/message",
        "params": {"level": "info", "data": "Listing tools"},
    }
    logging_callback: Final = AsyncMock()

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "DELETE":
            return httpx2.Response(200)
        payload: Final = json.loads(request.content)
        if "id" not in payload:
            return httpx2.Response(202)
        if payload["method"] == "initialize":
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload["id"],
                    "result": {
                        "protocolVersion": payload["params"]["protocolVersion"],
                        "capabilities": {"logging": {}, "tools": {}},
                        "serverInfo": {"name": "test", "version": "1"},
                    },
                },
            )
        response: Final = {
            "jsonrpc": "2.0",
            "id": payload["id"],
            "result": {"tools": [{"name": "search", "inputSchema": {"type": "object"}}]},
        }
        return httpx2.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            content="".join(f"event: message\ndata: {json.dumps(message)}\n\n" for message in (notification, response)),
        )

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http_client:
        client: Final = MCPClient(server_url="https://example.com/mcp", timeout=30, logging_callback=logging_callback)
        result: Final = await asyncio.wait_for(
            client._execute_session_operation(
                streamable_http_client(client.server_url, http_client=http_client), lambda session: session.list_tools()
            ),
            timeout=3,
        )

    assert [tool.name for tool in result.tools] == ["search"]
    logging_callback.assert_awaited_once_with(LoggingMessageNotificationParams(level="info", data="Listing tools"))


@pytest.mark.asyncio
async def test_invalid_tool_list_schema_is_identified_as_an_upstream_response() -> None:
    from litellm.proxy._experimental.mcp_server.rest_endpoints import _connection_error_message

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "DELETE":
            return httpx2.Response(200)
        payload: Final = json.loads(request.content)
        if "id" not in payload:
            return httpx2.Response(202)
        result: Final = (
            {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {},
                "serverInfo": {"name": "test", "version": "1"},
            }
            if payload["method"] == "initialize"
            else {"tools": "secret-invalid-tools"}
        )
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http_client:
        client: Final = MCPClient(server_url="https://example.com/mcp", timeout=30)
        with pytest.raises(ValidationError) as caught:
            await asyncio.wait_for(
                client._execute_session_operation(
                    streamable_http_client(client.server_url, http_client=http_client),
                    lambda session: session.list_tools(),
                ),
                timeout=3,
            )

    message: Final = _connection_error_message(caught.value, client.server_url, 30)
    assert "invalid MCP response" in message
    assert "secret" not in message


class _DiagnosticSSEStream(httpx2.AsyncByteStream):
    def __init__(self, messages: asyncio.Queue[bytes | Exception | None]) -> None:
        self.messages = messages

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"event: endpoint\ndata: /messages\n\n"
        while True:
            message: Final = await self.messages.get()
            if message is None:
                return
            if isinstance(message, Exception):
                raise message
            yield b"event: message\ndata: " + message + b"\n\n"


_DIAGNOSTIC_STDIO_SERVER: Final = """
import json, sys
mode, failure_method = sys.argv[1:]
for line in sys.stdin:
    request = json.loads(line)
    if "method" not in request or "id" not in request:
        continue
    if request["method"] == failure_method:
        if mode == "bad-json":
            print("secret-invalid-json", flush=True)
            continue
        if mode == "closed":
            sys.exit(0)
        if mode == "silent":
            print(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "Waiting"}}), flush=True)
            continue
    if request["method"] == "initialize":
        result = {"protocolVersion": request["params"]["protocolVersion"], "capabilities": {"tools": {}, "logging": {}}, "serverInfo": {"name": "diagnostic", "version": "1"}}
    elif request["method"] == "tools/list":
        print(json.dumps({"jsonrpc": "2.0", "method": "notifications/message", "params": {"level": "info", "data": "Listing tools"}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": "unmatched", "result": {}}), flush=True)
        print(json.dumps({"jsonrpc": "2.0", "id": "server-ping", "method": "ping"}), flush=True)
        result = {"tools": [{"name": "ping", "inputSchema": {"type": "object"}}]}
    else:
        result = {"content": [{"type": "text", "text": "pong"}], "isError": False}
    print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
"""


def _diagnostic_transport(transport: MCPTransport, mode: str, failure_method: str) -> _TransportContext:
    from mcp import StdioServerParameters
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client

    if transport == MCPTransport.stdio:
        return stdio_client(
            StdioServerParameters(
                command=sys.executable, args=["-u", "-c", _DIAGNOSTIC_STDIO_SERVER, mode, failure_method]
            )
        )
    messages: Final[asyncio.Queue[bytes | Exception | None]] = asyncio.Queue()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            return httpx2.Response(
                200, headers={"Content-Type": "text/event-stream"}, stream=_DiagnosticSSEStream(messages)
            )
        payload: Final = json.loads(request.content)
        if "method" not in payload or "id" not in payload:
            return httpx2.Response(202)
        if payload["method"] == failure_method and mode != "ok":
            if mode == "bad-json":
                await messages.put(b"secret-invalid-json")
            elif mode == "io-error":
                await messages.put(httpx2.ReadError("secret-read-error"))
            elif mode == "closed":
                await messages.put(None)
            elif mode == "silent":
                await messages.put(
                    b'{"jsonrpc":"2.0","method":"notifications/message","params":{"level":"info","data":"Waiting"}}'
                )
            return httpx2.Response(202)
        if payload["method"] == "tools/list":
            for message in (
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/message",
                    "params": {"level": "info", "data": "Listing tools"},
                },
                {"jsonrpc": "2.0", "id": "unmatched", "result": {}},
                {"jsonrpc": "2.0", "id": "server-ping", "method": "ping"},
            ):
                await messages.put(json.dumps(message).encode())
        result: Final = (
            {
                "protocolVersion": payload["params"]["protocolVersion"],
                "capabilities": {"tools": {}, "logging": {}},
                "serverInfo": {"name": "diagnostic", "version": "1"},
            }
            if payload["method"] == "initialize"
            else {"tools": [{"name": "ping", "inputSchema": {"type": "object"}}]}
            if payload["method"] == "tools/list"
            else {"content": [{"type": "text", "text": "pong"}], "isError": False}
        )
        await messages.put(json.dumps({"jsonrpc": "2.0", "id": payload["id"], "result": result}).encode())
        return httpx2.Response(202)

    def factory(
        headers: dict[str, str] | None = None,
        timeout: httpx2.Timeout | None = None,
        auth: httpx2.Auth | None = None,
    ) -> httpx2.AsyncClient:
        return httpx2.AsyncClient(transport=httpx2.MockTransport(respond), headers=headers, timeout=timeout, auth=auth)

    return sse_client("https://example.com/sse", httpx_client_factory=factory)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [MCPTransport.sse, MCPTransport.stdio])
@pytest.mark.parametrize("failure_method", ["initialize", "tools/list"])
async def test_transport_parsing_failure_is_preserved(transport: MCPTransport, failure_method: str) -> None:
    client: Final = MCPClient(server_url="https://example.com/sse", transport_type=transport, timeout=0.2)
    with pytest.raises(ValidationError):
        await asyncio.wait_for(
            client._execute_session_operation(
                _diagnostic_transport(transport, "bad-json", failure_method), lambda session: session.list_tools()
            ),
            timeout=3,
        )


@pytest.mark.asyncio
async def test_sse_read_failure_is_preserved() -> None:
    client: Final = MCPClient(server_url="https://example.com/sse", transport_type=MCPTransport.sse, timeout=0.2)
    with pytest.raises(httpx2.ReadError, match="secret-read-error"):
        await asyncio.wait_for(
            client._execute_session_operation(
                _diagnostic_transport(MCPTransport.sse, "io-error", "tools/list"), lambda session: session.list_tools()
            ),
            timeout=3,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [MCPTransport.sse, MCPTransport.stdio])
@pytest.mark.parametrize("mode", ["ok", "closed", "silent"])
async def test_transport_completion_and_normal_messages(transport: MCPTransport, mode: str) -> None:
    from mcp import ClientSession
    from litellm.proxy._experimental.mcp_server.rest_endpoints import _connection_error_message

    logging_callback: Final = AsyncMock()
    read_timeout: Final = 0.2 if mode == "silent" else 30
    client: Final = MCPClient(
        server_url="https://example.com/sse", transport_type=transport, timeout=read_timeout, logging_callback=logging_callback
    )

    async def operation(session: ClientSession) -> CallToolResult:
        tools: Final = await session.list_tools()
        assert [tool.name for tool in tools.tools] == ["ping"]
        return await session.call_tool("ping", {})

    pending: Final = client._execute_session_operation(_diagnostic_transport(transport, mode, "tools/list"), operation)
    if mode == "ok":
        result: Final = await asyncio.wait_for(pending, timeout=3)
        assert result.is_error is False
        assert result.content[0].text == "pong"
        logging_callback.assert_awaited_once_with(LoggingMessageNotificationParams(level="info", data="Listing tools"))
    else:
        with pytest.raises(MCPError) as caught:
            await asyncio.wait_for(pending, timeout=3)
        if mode == "closed":
            assert "connection was closed" in _connection_error_message(caught.value, client.server_url, read_timeout)
        else:
            assert isinstance(as_mcp_read_timeout(caught.value), TimeoutError)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", [MCPTransport.sse, MCPTransport.stdio])
async def test_transport_cancellation_cleans_up_a_pending_request(transport: MCPTransport) -> None:
    ready: Final = asyncio.Event()

    async def on_log(message: LoggingMessageNotificationParams) -> None:
        if message.data == "Waiting":
            ready.set()

    client: Final = MCPClient(
        server_url="https://example.com/sse", transport_type=transport, timeout=30, logging_callback=on_log
    )
    task: Final = asyncio.create_task(
        client._execute_session_operation(
            _diagnostic_transport(transport, "silent", "tools/list"), lambda session: session.list_tools()
        )
    )
    try:
        await asyncio.wait_for(ready.wait(), timeout=3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)


class _InterruptedHTTPBody(httpx2.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'{"jsonrpc":'
        raise httpx2.RemoteProtocolError("secret-incomplete-response")


@pytest.mark.asyncio
async def test_interrupted_http_response_preserves_the_transport_failure() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"Content-Type": "application/json"}, stream=_InterruptedHTTPBody())

    client: Final = MCPClient(server_url="https://example.com/mcp", timeout=30)
    async with client._create_httpx_client_factory(transport=httpx2.MockTransport(respond))() as http_client:
        with pytest.raises(httpx2.RemoteProtocolError, match="secret-incomplete-response"):
            await asyncio.wait_for(
                client._execute_session_operation(
                    streamable_http_client(client.server_url, http_client=http_client),
                    lambda session: session.list_tools(),
                    http_client=http_client,
                ),
                timeout=3,
            )


@pytest.mark.asyncio
async def test_empty_http_event_stream_uses_the_existing_request_deadline() -> None:
    def respond(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, headers={"Content-Type": "text/event-stream"}, content=b"")

    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as http_client:
        client: Final = MCPClient(server_url="https://example.com/mcp", timeout=0.2)
        with pytest.raises(MCPError) as caught:
            await asyncio.wait_for(
                client._execute_session_operation(
                    streamable_http_client(client.server_url, http_client=http_client),
                    lambda session: session.list_tools(),
                ),
                timeout=3,
            )
    assert caught.value.error.code == CONNECTION_CLOSED
    assert "SSE stream ended" in caught.value.error.message


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("prompts/list", "resources/list", "resources/templates/list"))
@pytest.mark.parametrize(
    "outcome",
    (
        "absent",
        "other_capability",
        "supported",
        "method_not_found",
        "internal_error",
        "unauthorized",
        "timeout",
        "initialize_not_found",
    ),
)
@pytest.mark.parametrize("raise_on_error", (False, True))
async def test_optional_discovery_capabilities_and_errors(
    method: str, outcome: str, caplog: pytest.LogCaptureFixture, raise_on_error: bool
) -> None:
    import logging
    from unittest.mock import Mock

    from mcp.types import JSONRPCRequest

    capability: Final = "prompts" if method == "prompts/list" else "resources"
    field: Final = {
        "prompts/list": "prompts",
        "resources/list": "resources",
        "resources/templates/list": "resourceTemplates",
    }[method]
    advertised: Final = "resources" if capability == "prompts" else "prompts"
    entry: Final = {
        "prompts/list": {"name": "example"},
        "resources/list": {"name": "example", "uri": "test://example"},
        "resources/templates/list": {"name": "example", "uriTemplate": "test://{name}"},
    }[method]

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "DELETE":
            return httpx2.Response(200)
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        if outcome == "initialize_not_found":
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload.id,
                    "error": {"code": -32601, "message": "Initialization rejected"},
                },
            )
        if payload.method == "initialize":
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload.id,
                    "result": {
                        "protocolVersion": (payload.params or {})["protocolVersion"],
                        "capabilities": {}
                        if outcome == "absent"
                        else {advertised if outcome == "other_capability" else capability: {}},
                        "serverInfo": {"name": "discovery", "version": "1"},
                    },
                },
            )
        if outcome == "timeout":
            raise httpx2.ReadTimeout("Optional list timed out", request=request)
        if outcome == "unauthorized":
            return httpx2.Response(401)
        if outcome in ("method_not_found", "internal_error", "absent", "other_capability"):
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload.id,
                    "error": {
                        "code": -32603 if outcome == "internal_error" else -32601,
                        "message": "Optional list rejected",
                    },
                },
            )
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload.id, "result": {field: [entry]}})

    responder: Final = Mock(side_effect=respond)
    caplog.set_level(logging.DEBUG, logger="LiteLLM")
    client: Final = _MockTransportClient(responder, server_url="https://example.com/mcp")
    operation: Final = {
        "prompts/list": client.list_prompts,
        "resources/list": client.list_resources,
        "resources/templates/list": client.list_resource_templates,
    }[method]
    if raise_on_error and outcome in ("internal_error", "unauthorized", "timeout", "initialize_not_found"):
        with pytest.raises((MCPError, httpx2.HTTPError)):
            await operation(raise_on_error=True)
        return
    result: Final = await operation(raise_on_error=raise_on_error)

    requests: Final = tuple(
        _JSONRPC_MESSAGE_ADAPTER.validate_json(call.args[0].content)
        for call in responder.call_args_list
        if call.args[0].method == "POST"
    )
    assert sum(isinstance(request, JSONRPCRequest) and request.method == method for request in requests) == (
        0 if outcome in ("absent", "other_capability", "initialize_not_found") else 1
    )
    assert [item.name for item in result] == (["example"] if outcome == "supported" else [])
    failures: Final = tuple(
        record for record in caplog.records if record.name == "LiteLLM" and record.levelno >= logging.WARNING
    )
    if outcome in ("internal_error", "unauthorized", "timeout", "initialize_not_found"):
        assert any(record.levelno == logging.ERROR and "failed" in record.message for record in failures)
    else:
        assert failures == ()
    if outcome == "method_not_found":
        assert any(
            record.levelno == logging.DEBUG and "Optional list rejected" in record.message for record in caplog.records
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("supports_first", (True, False))
async def test_optional_discovery_uses_each_sessions_capabilities(supports_first: bool) -> None:
    from unittest.mock import Mock

    from mcp.types import JSONRPCRequest

    capabilities: Final = iter(({"resources": {}}, {}) if supports_first else ({}, {"resources": {}}))

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "DELETE":
            return httpx2.Response(200)
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        result: Final = (
            {
                "protocolVersion": (payload.params or {})["protocolVersion"],
                "capabilities": next(capabilities),
                "serverInfo": {"name": "changing", "version": "1"},
            }
            if payload.method == "initialize"
            else {"resources": [{"name": "example", "uri": "test://example"}]}
        )
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload.id, "result": result})

    responder: Final = Mock(side_effect=respond)
    client: Final = _MockTransportClient(responder, server_url="https://example.com/mcp")
    first: Final = await client.list_resources()
    second: Final = await client.list_resources()

    assert [item.name for item in first] == (["example"] if supports_first else [])
    assert [item.name for item in second] == ([] if supports_first else ["example"])
    requests: Final = tuple(
        _JSONRPC_MESSAGE_ADAPTER.validate_json(call.args[0].content)
        for call in responder.call_args_list
        if call.args[0].method == "POST"
    )
    assert sum(isinstance(request, JSONRPCRequest) and request.method == "resources/list" for request in requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("prompts/list", "resources/list", "resources/templates/list"))
async def test_optional_discovery_preserves_cancellation(method: str) -> None:
    from mcp.types import JSONRPCRequest

    ready: Final = asyncio.Event()
    pending: Final = asyncio.Event()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "DELETE":
            return httpx2.Response(200)
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        if payload.method == "initialize":
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload.id,
                    "result": {
                        "protocolVersion": (payload.params or {})["protocolVersion"],
                        "capabilities": {"resources": {}, "prompts": {}},
                        "serverInfo": {"name": "pending", "version": "1"},
                    },
                },
            )
        if not (payload.params or {}).get("cursor"):
            field: Final = {
                "prompts/list": "prompts",
                "resources/list": "resources",
                "resources/templates/list": "resourceTemplates",
            }[method]
            return httpx2.Response(
                200, json={"jsonrpc": "2.0", "id": payload.id, "result": {field: [], "nextCursor": "pending-page"}}
            )
        ready.set()
        await pending.wait()
        return httpx2.Response(202)

    client: Final = _MockTransportClient(respond, server_url="https://example.com/mcp")
    operation: Final = {
        "prompts/list": client.list_prompts,
        "resources/list": client.list_resources,
        "resources/templates/list": client.list_resource_templates,
    }[method]
    task: Final = asyncio.create_task(operation())
    try:
        await asyncio.wait_for(ready.wait(), timeout=3)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("prompts/list", "resources/list", "resources/templates/list"))
@pytest.mark.parametrize("session_id", (None, "pagination-session"))
@pytest.mark.parametrize("empty_middle", (False, True))
async def test_optional_discovery_collects_all_pages(method: str, session_id: str | None, empty_middle: bool) -> None:
    from mcp.types import Prompt, PromptArgument, Resource, ResourceTemplate

    field: Final = {
        "prompts/list": "prompts",
        "resources/list": "resources",
        "resources/templates/list": "resourceTemplates",
    }[method]
    entries: Final = tuple(
        {
            "prompts/list": Prompt(
                name=f"item-{index}",
                description="prompt description",
                arguments=[PromptArgument(name="query", required=True)],
            ),
            "resources/list": Resource(
                name=f"item-{index}",
                uri=f"test://item/{index}",
                mime_type="text/plain",
                description="resource description",
            ),
            "resources/templates/list": ResourceTemplate(
                name=f"item-{index}", uri_template=f"test://item/{index}/{{query}}", mime_type="text/plain"
            ),
        }[method]
        for index in range(5)
    )

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "GET":
            return httpx2.Response(405)
        if request.method == "DELETE":
            return httpx2.Response(200)
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        if payload.method == "initialize":
            return httpx2.Response(
                200,
                headers={"mcp-session-id": session_id} if session_id else {},
                json={
                    "jsonrpc": "2.0",
                    "id": payload.id,
                    "result": {
                        "protocolVersion": (payload.params or {})["protocolVersion"],
                        "capabilities": {"prompts": {}, "resources": {}},
                        "serverInfo": {"name": "paged", "version": "1"},
                    },
                },
            )
        assert payload.method == method
        assert request.headers.get("mcp-session-id") == session_id
        cursor: Final = (payload.params or {}).get("cursor")
        assert cursor in (None, "opaque:/second+page", "opaque:/last+page")
        page: Final = (
            entries[:3] if cursor is None else (() if empty_middle and cursor == "opaque:/second+page" else entries[3:])
        )
        next_cursor: Final = (
            "opaque:/second+page"
            if cursor is None
            else "opaque:/last+page"
            if empty_middle and cursor == "opaque:/second+page"
            else ""
        )
        return httpx2.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload.id,
                "result": {
                    field: [item.model_dump(mode="json", by_alias=True) for item in page],
                    "nextCursor": next_cursor,
                },
            },
        )

    responder: Final = Mock(side_effect=respond)
    client: Final = _MockTransportClient(responder, server_url="https://example.com/mcp")
    operation: Final = {
        "prompts/list": client.list_prompts,
        "resources/list": client.list_resources,
        "resources/templates/list": client.list_resource_templates,
    }[method]
    assert await operation(raise_on_error=True) == list(entries)
    requests: Final = tuple(
        _JSONRPC_MESSAGE_ADAPTER.validate_json(call.args[0].content)
        for call in responder.call_args_list
        if call.args[0].method == "POST"
    )
    assert sum(isinstance(request, JSONRPCRequest) and request.method == "initialize" for request in requests) == 1
    assert tuple(
        (request.params or {}).get("cursor")
        for request in requests
        if isinstance(request, JSONRPCRequest) and request.method == method
    ) == ((None, "opaque:/second+page", "opaque:/last+page") if empty_middle else (None, "opaque:/second+page"))
    assert sum(call.args[0].method == "DELETE" for call in responder.call_args_list) == (1 if session_id else 0)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("prompts/list", "resources/list", "resources/templates/list"))
@pytest.mark.parametrize(
    "failure", ("repeat", "cycle", "cap", "method_not_found", "internal_error", "unauthorized", "deadline")
)
@pytest.mark.parametrize("strict", (False, True))
async def test_optional_discovery_rejects_incomplete_walks(
    method: str, failure: str, strict: bool, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(mcp_client_module, "MCP_TOOL_LISTING_MAX_PAGES", 3 if failure == "cycle" else 2, raising=False)
    monkeypatch.setattr(mcp_client_module, "MCP_TOOL_LISTING_TIMEOUT", 0.05)
    field: Final = {
        "prompts/list": "prompts",
        "resources/list": "resources",
        "resources/templates/list": "resourceTemplates",
    }[method]
    entry: Final = {
        "prompts/list": {"name": "first"},
        "resources/list": {"name": "first", "uri": "test://first"},
        "resources/templates/list": {"name": "first", "uriTemplate": "test://{name}"},
    }[method]
    cancelled: Final = asyncio.Event()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        if payload.method == "initialize":
            return httpx2.Response(
                200,
                json={
                    "jsonrpc": "2.0",
                    "id": payload.id,
                    "result": {
                        "protocolVersion": (payload.params or {})["protocolVersion"],
                        "capabilities": {"prompts": {}, "resources": {}},
                        "serverInfo": {"name": "interrupted", "version": "1"},
                    },
                },
            )
        assert payload.method == method
        cursor: Final = (payload.params or {}).get("cursor")
        if cursor is not None:
            if failure == "deadline":
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()
            if failure == "unauthorized":
                return httpx2.Response(401)
            if failure in ("method_not_found", "internal_error"):
                return httpx2.Response(
                    200,
                    json={
                        "jsonrpc": "2.0",
                        "id": payload.id,
                        "error": {
                            "code": -32601 if failure == "method_not_found" else -32603,
                            "message": "Later page unavailable",
                        },
                    },
                )
        next_cursor: Final = (
            "private-cursor-2" if cursor == "private-cursor-1" and failure != "repeat" else "private-cursor-1"
        )
        return httpx2.Response(
            200, json={"jsonrpc": "2.0", "id": payload.id, "result": {field: [entry], "nextCursor": next_cursor}}
        )

    responder: Final = AsyncMock(side_effect=respond)
    client: Final = _MockTransportClient(responder, server_url="https://example.com/mcp", timeout=0.2)
    operation: Final = {
        "prompts/list": client.list_prompts,
        "resources/list": client.list_resources,
        "resources/templates/list": client.list_resource_templates,
    }[method]
    if strict:
        error_type: Final = {
            "internal_error": MCPError,
            "unauthorized": httpx2.HTTPStatusError,
            "deadline": TimeoutError,
        }.get(failure, RuntimeError)
        with pytest.raises(error_type):
            await operation(raise_on_error=True)
    else:
        assert await operation() == []
    assert len(
        tuple(
            payload
            for call in responder.call_args_list
            if isinstance(payload := _JSONRPC_MESSAGE_ADAPTER.validate_json(call.args[0].content), JSONRPCRequest)
            and payload.method == method
        )
    ) == (3 if failure == "cycle" else 2)
    assert "private-cursor" not in caplog.text
    if failure == "deadline":
        assert cancelled.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ("prompts/list", "resources/list", "resources/templates/list"))
async def test_optional_discovery_allows_exhaustion_at_page_cap(method: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_client_module, "MCP_TOOL_LISTING_MAX_PAGES", 2, raising=False)
    field: Final = {
        "prompts/list": "prompts",
        "resources/list": "resources",
        "resources/templates/list": "resourceTemplates",
    }[method]

    def respond(request: httpx2.Request) -> httpx2.Response:
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        if payload.method == "initialize":
            result: Final = {
                "protocolVersion": (payload.params or {})["protocolVersion"],
                "capabilities": {"prompts": {}, "resources": {}},
                "serverInfo": {"name": "empty-pages", "version": "1"},
            }
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload.id, "result": result})
        assert payload.method == method
        return httpx2.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": payload.id,
                "result": {field: [], "nextCursor": None if (payload.params or {}).get("cursor") else "last-page"},
            },
        )

    responder: Final = Mock(side_effect=respond)
    client: Final = _MockTransportClient(responder, server_url="https://example.com/mcp")
    operation: Final = {
        "prompts/list": client.list_prompts,
        "resources/list": client.list_resources,
        "resources/templates/list": client.list_resource_templates,
    }[method]
    assert await operation(raise_on_error=True) == []
    assert (
        sum(
            isinstance(payload := _JSONRPC_MESSAGE_ADAPTER.validate_json(call.args[0].content), JSONRPCRequest)
            and payload.method == method
            for call in responder.call_args_list
        )
        == 2
    )


def test_client_import_before_proxy_credentials_succeeds_in_fresh_process():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", "import litellm.experimental_mcp_client.client; from litellm.proxy._experimental.mcp_server.mcp_server_manager import MCPServerManager; print(MCPServerManager.__name__)"],
        capture_output=True, text=True, timeout=60, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "MCPServerManager"


@pytest.mark.asyncio
@pytest.mark.parametrize("resolved", (False, True))
async def test_discovery_auth_fingerprint_tracks_effective_credentials(resolved: bool) -> None:
    from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import StaticHeaderAuth

    def client(token: str) -> MCPClient:
        return MCPClient(
            server_url="https://example.com/mcp",
            auth_type=MCPAuth.api_key,
            auth_value=None if resolved else token,
            resolved_auth=StaticHeaderAuth(token) if resolved else None,
        )

    original: Final = await client("private-original-credential").discovery_auth_fingerprint()
    repeated: Final = await client("private-original-credential").discovery_auth_fingerprint()
    replaced: Final = await client("private-replaced-credential").discovery_auth_fingerprint()
    assert original == repeated
    assert original != replaced
    assert len(original) == 64
    assert "private-original-credential" not in original


@pytest.mark.asyncio
async def test_request_auth_preview_uses_the_same_effective_headers_as_egress() -> None:
    from litellm.proxy._experimental.mcp_server.outbound_credentials.httpx_auth import StaticHeaderAuth

    client: Final = MCPClient(
        server_url="https://upstream.example/mcp", auth_type=MCPAuth.bearer_token,
        resolved_auth=StaticHeaderAuth("Bearer resolved"), extra_headers={"X-Trace": "trace"},
    )
    request: Final = await client.prepare_request_auth()
    assert request.method == "POST"
    assert str(request.url) == "https://upstream.example/mcp"
    assert request.headers["Authorization"] == "Bearer resolved"
    assert request.headers["X-Trace"] == "trace"


@pytest.mark.asyncio
@pytest.mark.parametrize("rpc_error", [False, True])
async def test_expired_session_preserves_sdk_error_and_next_operation_reinitializes(rpc_error: bool) -> None:
    from mcp.types import INVALID_REQUEST, METHOD_NOT_FOUND

    requests = []

    def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method != "POST":
            return httpx2.Response(405)
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx2.Response(202)
        requests.append((payload["method"], request.headers.get("mcp-session-id")))
        if payload["method"] == "initialize":
            return httpx2.Response(200, headers={"mcp-session-id": f"session-{len(requests)}"}, json={
                "jsonrpc": "2.0", "id": payload["id"], "result": {
                    "protocolVersion": "2025-06-18", "capabilities": {},
                    "serverInfo": {"name": "expiry-test", "version": "1"},
                },
            })
        if len(requests) == 2:
            if rpc_error:
                return httpx2.Response(404, json={
                    "jsonrpc": "2.0", "id": payload["id"],
                    "error": {"code": METHOD_NOT_FOUND, "message": "Tool catalog unavailable"},
                })
            return httpx2.Response(404)
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": {"tools": []}})

    client = MCPClient(server_url="https://example.com/mcp", timeout=3)
    async with client._create_httpx_client_factory(transport=httpx2.MockTransport(respond))() as http_client:
        with pytest.raises(MCPError) as caught:
            await client._execute_session_operation(
                streamable_http_client(client.server_url, http_client=http_client), lambda session: session.list_tools()
            )
        assert caught.value.error.code == (METHOD_NOT_FOUND if rpc_error else INVALID_REQUEST)
        assert caught.value.error.message == ("Tool catalog unavailable" if rpc_error else "Session terminated")
        result = await client._execute_session_operation(
            streamable_http_client(client.server_url, http_client=http_client), lambda session: session.list_tools()
        )
    assert result.tools == []
    assert requests == [("initialize", None), ("tools/list", "session-1"), ("initialize", None), ("tools/list", "session-3")]


@pytest.mark.asyncio
async def test_404_before_session_initialization_preserves_method_not_found() -> None:
    from mcp.types import METHOD_NOT_FOUND

    client = MCPClient(server_url="https://example.com/mcp", timeout=3)
    transport = httpx2.MockTransport(lambda request: httpx2.Response(404))
    async with client._create_httpx_client_factory(transport=transport)() as http_client:
        with pytest.raises(MCPError) as caught:
            await client._execute_session_operation(
                streamable_http_client(client.server_url, http_client=http_client), lambda session: session.list_tools()
            )
    assert caught.value.error.code == METHOD_NOT_FOUND
    assert caught.value.error.message == "Not Found"


@pytest.mark.parametrize("missing_module", ("mcp", "httpx2", "mcp.types", "openai.types.chat"))
def test_public_mcp_import_missing_dependency(missing_module: str) -> None:
    with patch.dict(sys.modules):
        for name in tuple(sys.modules):
            if name.startswith(("litellm.experimental_mcp_client", "mcp.", "mcp_types.")) or name == "mcp":
                del sys.modules[name]
        with patch.dict(sys.modules, {missing_module: None}):
            with pytest.raises(ImportError) as caught:
                importlib.import_module("litellm.experimental_mcp_client.client")

    if missing_module in ("mcp", "httpx2"):
        assert "pip install 'litellm[mcp]'" in str(caught.value)
        assert isinstance(caught.value.__cause__, ModuleNotFoundError)
        assert caught.value.__cause__.name == missing_module
    else:
        assert isinstance(caught.value, ModuleNotFoundError)
        assert caught.value.name == missing_module
        assert caught.value.__cause__ is None
        assert "litellm[mcp]" not in str(caught.value)


def test_public_mcp_import_preserves_incompatible_sdk_error() -> None:
    with patch.dict(sys.modules):
        for name in tuple(sys.modules):
            if name.startswith("litellm.experimental_mcp_client"):
                del sys.modules[name]
        with patch.dict(sys.modules, {"mcp": ModuleType("mcp")}):
            with pytest.raises(ImportError, match="cannot import name 'ClientSession'") as caught:
                importlib.import_module("litellm.experimental_mcp_client.client")

    assert not isinstance(caught.value, ModuleNotFoundError)
    assert caught.value.__cause__ is None
    assert "litellm[mcp]" not in str(caught.value)


@pytest.mark.asyncio
@pytest.mark.parametrize("grouped", (False, True))
@pytest.mark.parametrize("raise_on_error", (False, True))
@pytest.mark.parametrize("termination", ("ok", "failure", "hang"))
async def test_outer_deadline_delivers_session_termination(termination: str, grouped: bool, raise_on_error: bool) -> None:
    deleted: Final = asyncio.Event()
    started: Final = asyncio.Event()

    async def respond(request: httpx2.Request) -> httpx2.Response:
        await anyio.lowlevel.checkpoint()
        if request.method == "DELETE":
            first_termination: Final = not deleted.is_set()
            deleted.set()
            if termination == "hang" and first_termination:
                await anyio.sleep_forever()
            return httpx2.Response(500 if termination == "failure" else 200)
        if request.method == "GET":
            return httpx2.Response(405)
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        if payload.method == "initialize":
            return httpx2.Response(
                200,
                headers={"mcp-session-id": "cancel-owned-session"},
                json={
                    "jsonrpc": "2.0",
                    "id": payload.id,
                    "result": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "cancellation-peer", "version": "1"},
                    },
                },
            )
        if payload.method == "tools/list":
            return httpx2.Response(200, json={"jsonrpc": "2.0", "id": payload.id, "result": {"tools": []}})
        started.set()
        await anyio.sleep_forever()
        raise AssertionError("cancelled request resumed")

    client: Final = _MockTransportClient(respond, server_url="https://example.com/mcp", timeout=30)

    async def invoke():
        with anyio.fail_after(0.2):
            pending: Final = client.call_tool(CallToolRequestParams(name="slow", arguments={}), raise_on_error=raise_on_error)
            if grouped:
                await asyncio.gather(pending)
            else:
                await pending

    before: Final = anyio.current_time()
    with pytest.raises(TimeoutError):
        await invoke()
    assert started.is_set()
    assert deleted.is_set(), "Cancellation must deliver DELETE before returning to the caller"

    assert anyio.current_time() - before < 6.5
    assert await client.list_tools(raise_on_error=True) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("original_error", (False, True))
async def test_task_cancellation_during_cleanup_preserves_failure(original_error: bool) -> None:
    deleting: Final = asyncio.Event()
    drained: Final = asyncio.Event()
    original: Final = RuntimeError("operation failed before teardown")

    async def respond(request: httpx2.Request) -> httpx2.Response:
        if request.method == "DELETE":
            deleting.set()
            try:
                await anyio.sleep_forever()
            finally:
                drained.set()
        if request.method == "GET":
            return httpx2.Response(405)
        payload: Final = _JSONRPC_MESSAGE_ADAPTER.validate_json(request.content)
        if not isinstance(payload, JSONRPCRequest):
            return httpx2.Response(202)
        return httpx2.Response(
            200,
            headers={"mcp-session-id": "cleanup-session"},
            json={
                "jsonrpc": "2.0",
                "id": payload.id,
                "result": {
                    "protocolVersion": (payload.params or {})["protocolVersion"],
                    "capabilities": {},
                    "serverInfo": {"name": "cleanup-peer", "version": "1"},
                },
            },
        )

    async def operation(session: mcp_client_module.ClientSession) -> str:
        if original_error:
            raise original
        return "completed"

    client: Final = _MockTransportClient(respond, server_url="https://example.com/mcp", timeout=30)
    task: Final = asyncio.create_task(client.run_with_session(operation))
    await asyncio.wait_for(deleting.wait(), 2)
    for _ in range(3):
        task.cancel()
        await asyncio.sleep(0)
    with pytest.raises(RuntimeError if original_error else asyncio.CancelledError) as caught:
        await task
    assert drained.is_set(), "Caller must wait for termination cleanup to finish"
    if original_error:
        assert caught.value is original
    else:
        assert task.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("original_error", (False, True))
@pytest.mark.parametrize("cancel_mode", ("task", "scope"))
async def test_http_close_cancellation_cannot_turn_into_success(original_error: bool, cancel_mode: str) -> None:
    closing: Final = asyncio.Event()
    drained: Final = asyncio.Event()
    original: Final = RuntimeError("failed before HTTP close")

    class ClosingHTTPClient(httpx2.AsyncClient):
        async def aclose(self) -> None:
            closing.set()
            try:
                await anyio.sleep_forever()
            finally:
                drained.set()

    class ClosingMCPClient(MCPClient):
        def _create_transport_context(self):
            http_client: Final = ClosingHTTPClient(transport=httpx2.MockTransport(lambda _: httpx2.Response(200)))
            return streamable_http_client(self.server_url, http_client=http_client), http_client

        async def _execute_session_operation(self, transport_ctx, operation, http_client=None):
            if original_error:
                raise original
            return "completed"

    client: Final = ClosingMCPClient(server_url="https://example.com/mcp")

    async def invoke() -> str:
        with anyio.fail_after(0.05 if cancel_mode == "scope" else None):
            return await client.run_with_session(AsyncMock())

    task: Final = asyncio.create_task(invoke())
    await asyncio.wait_for(closing.wait(), 2)
    if cancel_mode == "task":
        task.cancel()
    cancellation_type: Final = asyncio.CancelledError if cancel_mode == "task" else TimeoutError
    with pytest.raises(RuntimeError if original_error else cancellation_type) as caught:
        await task
    assert drained.is_set(), "Caller must wait for HTTP closure to finish"
    if original_error:
        assert caught.value is original
    elif cancel_mode == "task":
        assert task.cancelled()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_mode", ("scope", "task", "wait_for", "read_timeout"))
@pytest.mark.parametrize("concurrency", (1, 5))
@pytest.mark.parametrize("termination", ("ok", "hang", "hang_body"))
@pytest.mark.parametrize("raise_on_error", (False, True))
async def test_cancellation_delivers_termination_over_tcp(
    cancel_mode: str, concurrency: int, termination: str, raise_on_error: bool
) -> None:
    started: Final = asyncio.Event()
    scope_ready: Final[asyncio.Future[anyio.CancelScope]] = asyncio.get_running_loop().create_future()
    terminations: Final[list[bytes]] = []
    starts: Final[list[bytes]] = []
    stop: Final = asyncio.Event()
    connections: Final[list[asyncio.Task[None]]] = []

    async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        connection: Final = asyncio.current_task()
        assert connection is not None
        connections.append(connection)
        try:
            request_line: Final = await reader.readline()
            if not request_line:
                return
            method: Final = request_line.split()[0]
            headers: Final = await reader.readuntil(b"\r\n\r\n")
            length: Final = next(
                (
                    int(line.split(b":", 1)[1])
                    for line in headers.splitlines()
                    if line.lower().startswith(b"content-length:")
                ),
                0,
            )
            try:
                body: Final = await reader.readexactly(length)
            except asyncio.IncompleteReadError:
                return
            if method == b"DELETE":
                terminations.append(body)
                if termination != "ok":
                    await stop.wait()
                    return
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            elif method == b"GET":
                writer.write(b"HTTP/1.1 405 Method Not Allowed\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            else:
                payload: Final = json.loads(body)
                if payload["method"] == "tools/call":
                    if termination == "hang_body":
                        writer.write(
                            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                            b"Content-Length: 200\r\nConnection: close\r\n\r\n"
                        )
                        await writer.drain()
                    starts.append(body)
                    if len(starts) == concurrency:
                        started.set()
                    await stop.wait()
                    return
                if payload["method"] == "initialize":
                    response: Final = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": payload["id"],
                            "result": {
                                "protocolVersion": "2025-06-18",
                                "capabilities": {"tools": {}},
                                "serverInfo": {"name": "tcp-peer", "version": "1"},
                            },
                        }
                    ).encode()
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nMcp-Session-Id: tcp-session\r\n"
                        + f"Content-Length: {len(response)}\r\nConnection: close\r\n\r\n".encode()
                        + response
                    )
                else:
                    writer.write(b"HTTP/1.1 202 Accepted\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()

    listener: Final = await asyncio.start_server(handle_connection, "127.0.0.1", 0)
    port: Final = listener.sockets[0].getsockname()[1]
    client: Final = MCPClient(
        server_url=f"http://127.0.0.1:{port}/mcp", timeout=2 if cancel_mode == "read_timeout" else 0.5 if termination != "ok" else 30
    )

    async def calls():
        results: Final = await asyncio.gather(
            *(
                client.call_tool(CallToolRequestParams(name="slow", arguments={}), raise_on_error=raise_on_error)
                for _ in range(concurrency)
            ),
            return_exceptions=cancel_mode == "read_timeout",
        )
        if cancel_mode == "read_timeout":
            if raise_on_error:
                assert all(isinstance(result, TimeoutError) for result in results)
            else:
                assert all(isinstance(result, CallToolResult) and result.is_error for result in results)
        return results

    async def invoke():
        if cancel_mode == "scope":
            with anyio.fail_after(None) as scope:
                scope_ready.set_result(scope)
                return await calls()
        return await calls()

    try:
        task: Final = asyncio.create_task(invoke())
        await asyncio.wait_for(started.wait(), 3)
        if cancel_mode == "scope":
            (await scope_ready).deadline = anyio.current_time() + 0.2
        if cancel_mode == "task":
            task.cancel()
        expected_error: Final = (
            TimeoutError
            if cancel_mode == "read_timeout"
            else asyncio.CancelledError
            if cancel_mode == "task"
            else TimeoutError
        )
        if cancel_mode == "read_timeout":
            done, _ = await asyncio.wait((task,), timeout=8)
            assert task in done, "Read timeout and bounded cleanup must complete without external cancellation"
            await task
        elif cancel_mode == "wait_for":
            with pytest.raises(expected_error):
                await asyncio.wait_for(task, 0.2)
        else:
            with pytest.raises(expected_error):
                await task
        assert len(starts) == concurrency
        assert len(terminations) == concurrency, "Each cancelled call must send DELETE over a fresh TCP connection"
    finally:
        stop.set()
        if not task.done():
            task.cancel()
            await asyncio.wait((task,), timeout=8)
        listener.close()
        for connection in connections:
            connection.cancel()
        closed: Final = await asyncio.wait_for(asyncio.gather(*connections, return_exceptions=True), 2)
        assert all(result is None or isinstance(result, asyncio.CancelledError) for result in closed), closed
        await asyncio.wait_for(listener.wait_closed(), 2)
