import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

# Add the parent directory to the path so we can import litellm
sys.path.insert(0, "../../../")

import litellm.experimental_mcp_client.client as mcp_client_module
from litellm.experimental_mcp_client.client import (
    MCPClient,
    _first_non_cancelled_cause,
)
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

        with pytest.raises(ValueError, match="stdio_config is required for stdio transport"):

            async def _noop(session):
                return None

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
# MCP list result tolerance
# ---------------------------------------------------------------------------


class TestMCPListOperations:
    """MCP list operations should tolerate omitted empty list fields."""

    async def _run_list_operation(self, method_name, result_type):
        client = MCPClient(server_url="http://example.com/mcp", transport_type="http")
        session = AsyncMock()

        async def send_request(_request, actual_result_type):
            assert actual_result_type is result_type
            return actual_result_type.model_validate({})

        session.send_request = AsyncMock(side_effect=send_request)

        async def run_with_session(operation):
            return await operation(session)

        client.run_with_session = AsyncMock(side_effect=run_with_session)

        result = await getattr(client, method_name)()

        assert result == []
        session.send_request.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_list_tools_tolerates_missing_tools_field(self):
        await self._run_list_operation(
            "list_tools", mcp_client_module._LenientListToolsResult
        )

    @pytest.mark.asyncio
    async def test_list_resources_tolerates_missing_resources_field(self):
        await self._run_list_operation(
            "list_resources", mcp_client_module._LenientListResourcesResult
        )

    @pytest.mark.asyncio
    async def test_list_prompts_tolerates_missing_prompts_field(self):
        await self._run_list_operation(
            "list_prompts", mcp_client_module._LenientListPromptsResult
        )


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


def _all_logged_messages(mock_logger):
    return " ".join(
        str(call.args[0])
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
