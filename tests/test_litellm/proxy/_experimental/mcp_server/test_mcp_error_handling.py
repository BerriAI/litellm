"""
Tests for MCP endpoint error handling.

Verifies that auth failures and other errors in MCP endpoints return proper
HTTP status codes (401, 403) instead of 500, and that ProxyException is
caught and converted to a JSON error response.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestHandleStreamableHttpMcpErrorHandling:
    """Tests that handle_streamable_http_mcp properly handles ProxyException."""

    @pytest.mark.asyncio
    async def test_should_return_401_on_auth_failure(self):
        """Auth failures (ProxyException with code=401) should return 401, not 500."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                handle_streamable_http_mcp,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        from litellm.proxy._types import ProxyErrorTypes, ProxyException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/undefined",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "server": ("localhost", 8000),
            "scheme": "http",
        }
        receive = AsyncMock()
        send = AsyncMock()

        auth_error = ProxyException(
            message="Malformed API Key passed in. Ensure Key has `Bearer ` prefix.",
            type=ProxyErrorTypes.auth_error,
            param="None",
            code=401,
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            side_effect=auth_error,
        ):
            await handle_streamable_http_mcp(scope, receive, send)

        assert send.called
        # First call is http.response.start with status code
        start_call = send.call_args_list[0]
        message = start_call[0][0]
        assert message["type"] == "http.response.start"
        assert message["status"] == 401

    @pytest.mark.asyncio
    async def test_should_return_403_on_forbidden(self):
        """ProxyException with code=403 should return 403, not 500."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                handle_streamable_http_mcp,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        from litellm.proxy._types import ProxyErrorTypes, ProxyException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/some_server",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "server": ("localhost", 8000),
            "scheme": "http",
        }
        receive = AsyncMock()
        send = AsyncMock()

        forbidden_error = ProxyException(
            message="Access denied",
            type=ProxyErrorTypes.auth_error,
            param="None",
            code=403,
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            side_effect=forbidden_error,
        ):
            await handle_streamable_http_mcp(scope, receive, send)

        assert send.called
        start_call = send.call_args_list[0]
        message = start_call[0][0]
        assert message["type"] == "http.response.start"
        assert message["status"] == 403

    @pytest.mark.asyncio
    async def test_should_return_error_body_as_json(self):
        """ProxyException responses should include error message and type in JSON body."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                handle_streamable_http_mcp,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        import json

        from litellm.proxy._types import ProxyErrorTypes, ProxyException

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/undefined",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "server": ("localhost", 8000),
            "scheme": "http",
        }
        receive = AsyncMock()
        send = AsyncMock()

        auth_error = ProxyException(
            message="Malformed API Key",
            type=ProxyErrorTypes.auth_error,
            param="None",
            code=401,
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            side_effect=auth_error,
        ):
            await handle_streamable_http_mcp(scope, receive, send)

        # Second call is http.response.body
        body_call = send.call_args_list[1]
        body_message = body_call[0][0]
        assert body_message["type"] == "http.response.body"
        body = json.loads(body_message["body"])
        assert body["error"]["message"] == "Malformed API Key"
        assert body["error"]["type"] == ProxyErrorTypes.auth_error

    @pytest.mark.asyncio
    async def test_should_return_404_for_nonexistent_mcp_server(self):
        """Non-existent MCP server names should return 404, not 200."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                handle_streamable_http_mcp,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/undefined",
            "headers": [
                (b"content-type", b"application/json"),
                (b"accept", b"application/json, text/event-stream"),
                (b"authorization", b"Bearer sk-1620"),
            ],
            "query_string": b"",
            "server": ("localhost", 8000),
            "scheme": "http",
        }
        receive = AsyncMock()
        send = AsyncMock()

        mock_auth = MagicMock()
        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(mock_auth, None, ["undefined"], {}, None, {}),
        ), patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager"
        ) as mock_mgr:
            mock_mgr.get_mcp_server_by_name.return_value = None

            await handle_streamable_http_mcp(scope, receive, send)

        assert send.called
        start_call = send.call_args_list[0]
        message = start_call[0][0]
        assert message["type"] == "http.response.start"
        assert message["status"] == 404

    @pytest.mark.asyncio
    async def test_should_still_500_on_unexpected_exceptions(self):
        """Non-ProxyException errors should still result in a 500 response."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                handle_streamable_http_mcp,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "type": "http",
            "method": "POST",
            "path": "/mcp/test",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
            "server": ("localhost", 8000),
            "scheme": "http",
        }
        receive = AsyncMock()
        send = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected crash"),
        ):
            await handle_streamable_http_mcp(scope, receive, send)

        assert send.called
        start_call = send.call_args_list[0]
        message = start_call[0][0]
        assert message["type"] == "http.response.start"
        assert message["status"] == 500


class TestHandleSseMcpErrorHandling:
    """Tests that handle_sse_mcp properly handles ProxyException."""

    @pytest.mark.asyncio
    async def test_should_return_401_on_auth_failure(self):
        """Auth failures in SSE handler should return 401, not 500."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                handle_sse_mcp,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        from litellm.proxy._types import ProxyErrorTypes, ProxyException

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/sse",
            "headers": [(b"accept", b"text/event-stream")],
            "query_string": b"",
            "server": ("localhost", 8000),
            "scheme": "http",
        }
        receive = AsyncMock()
        send = AsyncMock()

        auth_error = ProxyException(
            message="Malformed API Key passed in.",
            type=ProxyErrorTypes.auth_error,
            param="None",
            code=401,
        )

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            side_effect=auth_error,
        ):
            await handle_sse_mcp(scope, receive, send)

        assert send.called
        start_call = send.call_args_list[0]
        message = start_call[0][0]
        assert message["type"] == "http.response.start"
        assert message["status"] == 401

    @pytest.mark.asyncio
    async def test_should_return_404_for_nonexistent_mcp_server(self):
        """Non-existent MCP server names in SSE handler should return 404, not 200."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                handle_sse_mcp,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/sse",
            "headers": [(b"accept", b"text/event-stream")],
            "query_string": b"",
            "server": ("localhost", 8000),
            "scheme": "http",
        }
        receive = AsyncMock()
        send = AsyncMock()

        mock_auth = MagicMock()
        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            return_value=(mock_auth, None, ["undefined"], {}, None, {}),
        ), patch(
            "litellm.proxy._experimental.mcp_server.server.global_mcp_server_manager"
        ) as mock_mgr:
            mock_mgr.get_mcp_server_by_name.return_value = None

            await handle_sse_mcp(scope, receive, send)

        assert send.called
        start_call = send.call_args_list[0]
        message = start_call[0][0]
        assert message["type"] == "http.response.start"
        assert message["status"] == 404

    @pytest.mark.asyncio
    async def test_should_still_500_on_unexpected_exceptions(self):
        """Non-ProxyException errors in SSE handler should still result in a 500 response."""
        try:
            from litellm.proxy._experimental.mcp_server.server import (
                handle_sse_mcp,
            )
        except ImportError:
            pytest.skip("MCP server not available")

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/sse",
            "headers": [(b"accept", b"text/event-stream")],
            "query_string": b"",
            "server": ("localhost", 8000),
            "scheme": "http",
        }
        receive = AsyncMock()
        send = AsyncMock()

        with patch(
            "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
            new_callable=AsyncMock,
            side_effect=RuntimeError("unexpected crash"),
        ):
            await handle_sse_mcp(scope, receive, send)

        assert send.called
        start_call = send.call_args_list[0]
        message = start_call[0][0]
        assert message["type"] == "http.response.start"
        assert message["status"] == 500
