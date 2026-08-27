import asyncio
import base64
import os
import sys
from importlib import metadata
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import anyio
import httpx
import pytest
from mcp import McpError
from mcp.shared.message import SessionMessage
from mcp.types import (
    LATEST_PROTOCOL_VERSION,
    ErrorData,
    Implementation,
    InitializeResult,
    JSONRPCError,
    JSONRPCMessage,
    JSONRPCResponse,
    ServerCapabilities,
)

# Add the parent directory to the path so we can import litellm

import litellm.experimental_mcp_client.client as mcp_client_module
from litellm.experimental_mcp_client.client import (
    MCP_STREAMABLE_HTTP_REQUIREMENT,
    MCPClient,
    _as_read_timeout,
    _first_non_cancelled_cause,
    missing_streamable_http_client_error,
    strip_auth_scheme,
)
from litellm.proxy._experimental.mcp_server.faults.list_outcomes import (
    classify_list_exception,
    list_fault_http_status,
)
from litellm.proxy._experimental.mcp_server.mcp_server_manager import (
    _format_byok_openapi_auth_header,
)
from litellm.types.mcp_server.mcp_server_manager import MCPServer
from litellm.types.mcp import MCPAuth, MCPStdioConfig, MCPTransport


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
        target = httpx.ConnectError("refused")
        group = _FakeExceptionGroup("g", [asyncio.CancelledError(), target])
        assert _first_non_cancelled_cause(group) is target

    def test_unwraps_nested_group(self):
        target = httpx.LocalProtocolError("Illegal header value")
        inner = _FakeExceptionGroup("inner", [asyncio.CancelledError(), target])
        outer = _FakeExceptionGroup("outer", [asyncio.CancelledError(), inner])
        assert _first_non_cancelled_cause(outer) is target

    def test_all_cancelled_returns_none(self):
        group = _FakeExceptionGroup("g", [asyncio.CancelledError(), asyncio.CancelledError()])
        assert _first_non_cancelled_cause(group) is None

    @pytest.mark.skipif(sys.version_info < (3, 11), reason="builtin ExceptionGroup requires 3.11+")
    def test_unwraps_builtin_exception_group(self):
        target = httpx.ConnectError("refused")
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
        connect_error = httpx.ConnectError("All connection attempts failed")
        transport_ctx = self._make_transport(_FakeExceptionGroup("transport", [connect_error]))

        async def _op(session):
            return "done"

        with pytest.raises(httpx.ConnectError):
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
    @patch("litellm.experimental_mcp_client.client.ClientSession")
    async def test_cleanup_error_after_success_is_swallowed(self, mock_session_cls):
        client = MCPClient(server_url="http://example.com/mcp", transport_type="http")
        init_result = MagicMock()
        init_result.instructions = None
        self._make_session(mock_session_cls, AsyncMock(return_value=init_result))
        transport_ctx = self._make_transport(_FakeExceptionGroup("late", [httpx.ConnectError("late cleanup error")]))

        async def _op(session):
            return "done"

        result = await client._execute_session_operation(transport_ctx, _op)
        assert result == "done"


class TestMCPClientResolvedAuth:
    """A pre-resolved httpx.Auth is attached to the upstream client's auth= slot."""

    @pytest.mark.asyncio
    async def test_resolved_auth_feeds_the_auth_slot(self):
        resolved = httpx.Auth()
        client = MCPClient(server_url="https://upstream.example.com", resolved_auth=resolved)
        http_client = client._create_httpx_client_factory()()
        try:
            assert http_client.auth is resolved
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_resolved_auth_takes_precedence_over_aws_auth(self):
        resolved = httpx.Auth()
        client = MCPClient(
            server_url="https://upstream.example.com",
            resolved_auth=resolved,
            aws_auth=httpx.Auth(),
        )
        http_client = client._create_httpx_client_factory()()
        try:
            assert http_client.auth is resolved
        finally:
            await http_client.aclose()

    @pytest.mark.asyncio
    async def test_without_resolved_auth_falls_back_to_aws_auth(self):
        aws = httpx.Auth()
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
            assert result.isError is True
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

    async def _fake_exec(_transport_ctx, _operation):
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
        await self._to_client_tx.send(SessionMessage(JSONRPCMessage(message)))

    async def _serve(self):
        async for session_message in self._from_client_rx:
            request = session_message.message.root
            method = getattr(request, "method", None)
            if method == "initialize":
                result = InitializeResult(
                    protocolVersion=LATEST_PROTOCOL_VERSION,
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
    travelling as ``McpError`` so it is never blamed on the gateway as a 504.

    This is the other half of the pair: the same real transport and the same real session, so one
    mechanism pins both directions.
    """
    client = _ScriptedClient(
        timeout=30,
        tools_list_error=ErrorData(code=int(httpx.codes.REQUEST_TIMEOUT), message="re-authenticate and retry"),
    )

    with pytest.raises(McpError) as exc_info:
        await asyncio.wait_for(client.list_tools(raise_on_error=True), timeout=10)

    assert not isinstance(exc_info.value, TimeoutError), "an upstream application error is not a gateway timeout"
    assert exc_info.value.error.code == int(httpx.codes.REQUEST_TIMEOUT)

    fault = classify_list_exception(exc_info.value)
    assert fault.tag != "timeout", "an upstream's own application error must never be reported as a gateway timeout"
    assert list_fault_http_status(fault) != 504


def _raise_mcp_error_while_handling_a_timeout(code: int, message: str) -> McpError:
    """An ``McpError`` carrying the context chain it would have if it were raised while a
    ``TimeoutError`` was in flight, which is how the SDK raises its own read timeout."""
    try:
        try:
            raise TimeoutError()
        except TimeoutError:
            raise McpError(ErrorData(code=code, message=message))
    except McpError as raised:
        return raised


def test_as_read_timeout_separates_the_sdk_timeout_from_a_relayed_upstream_error():
    """Neither signal alone is enough. The code alone cannot separate the SDK's own timeout from an
    upstream JSON-RPC error that happens to use 408, and the context chain alone cannot separate it
    from any other relayed error that surfaces while a timeout is being handled, so both must hold.
    """
    timeout_code = int(httpx.codes.REQUEST_TIMEOUT)

    translated = _as_read_timeout(_raise_mcp_error_while_handling_a_timeout(timeout_code, "Timed out while waiting"))
    assert isinstance(translated, TimeoutError)
    assert str(translated) == "Timed out while waiting"

    relayed_408 = McpError(ErrorData(code=timeout_code, message="upstream said 408"))
    assert _as_read_timeout(relayed_408) is None, "an upstream 408 with no elapsed timeout is not our timeout"

    relayed_other = _raise_mcp_error_while_handling_a_timeout(-32603, "upstream internal error")
    assert _as_read_timeout(relayed_other) is None, "a non-timeout code is not our timeout, whatever the chain"

    assert _as_read_timeout(McpError(ErrorData(code=-32603, message="boom"))) is None
    assert _as_read_timeout(RuntimeError("not an McpError")) is None


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


def test_missing_streamable_http_client_error_names_requirement_and_remedy():
    message = str(missing_streamable_http_client_error())

    assert MCP_STREAMABLE_HTTP_REQUIREMENT in message
    assert "pip install 'litellm[mcp]'" in message
    assert metadata.version("mcp") in message


@pytest.mark.asyncio
async def test_http_transport_without_streamable_http_client_raises_actionable_import_error():
    client = MCPClient(
        server_url="https://mcp-server.example.com",
        transport_type=MCPTransport.http,
    )

    with patch.object(  # test-quality-ok: simulates mcp<1.24.0 whose module lacks this import-time symbol
        mcp_client_module, "streamable_http_client", None
    ):
        with pytest.raises(ImportError, match=r"pip install 'litellm\[mcp\]'"):
            await client.list_tools(raise_on_error=True)


def test_mcp_extra_matches_proxy_extra_and_supports_streamable_http():
    try:
        import tomllib
    except ImportError:
        tomllib = pytest.importorskip("tomli")
    from packaging.requirements import Requirement

    pyproject_path = Path(__file__).parents[3] / "pyproject.toml"
    with pyproject_path.open("rb") as f:
        extras = tomllib.load(f)["project"]["optional-dependencies"]

    mcp_extra = extras["mcp"]
    assert len(mcp_extra) == 1

    proxy_mcp_requirements = [req for req in extras["proxy"] if Requirement(req).name == "mcp"]
    assert mcp_extra == proxy_mcp_requirements

    specifier = Requirement(mcp_extra[0]).specifier
    assert not specifier.contains("1.23.0")
    assert specifier.contains("1.28.1")
