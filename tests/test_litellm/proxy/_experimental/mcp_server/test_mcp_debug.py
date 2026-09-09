"""
Tests for MCPDebug — MCP OAuth2 debug response headers.
"""

import asyncio
from typing import Final

import pytest
from starlette.types import Message

from litellm.proxy._experimental.mcp_server.outbound_credentials.types import AuthResolution

from litellm.proxy._experimental.mcp_server.mcp_debug import (
    MCP_DEBUG_REQUEST_HEADER,
    MCPDebug,
    MCPAuthDiagnostics,
)


class TestIsDebugEnabled:
    def test_enabled_true(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "true"}) is True

    def test_enabled_yes(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "yes"}) is True

    def test_enabled_one(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "1"}) is True

    def test_disabled_false(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "false"}) is False

    def test_disabled_missing(self):
        assert MCPDebug.is_debug_enabled({"other-header": "value"}) is False

    def test_case_insensitive_header_name(self):
        assert MCPDebug.is_debug_enabled({"X-LiteLLM-MCP-Debug": "true"}) is True

    def test_case_insensitive_value(self):
        assert MCPDebug.is_debug_enabled({MCP_DEBUG_REQUEST_HEADER: "TRUE"}) is True


class TestMask:
    def test_none_returns_none_label(self):
        assert MCPDebug._mask(None) == "(none)"

    def test_empty_returns_none_label(self):
        assert MCPDebug._mask("") == "(none)"

    def test_short_value_masked(self):
        # Short auth values must not be echoed verbatim in debug headers, even though
        # visible_prefix + visible_suffix would otherwise reveal the whole value.
        masked = MCPDebug._mask("sk-1234")
        assert "sk-1234" not in masked
        assert set(masked) == {"*"}
        assert len(masked) == len("sk-1234")

    def test_long_value_masked(self):
        result = MCPDebug._mask("Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9")
        assert result.startswith("Bearer")
        assert result.endswith("VCJ9")
        assert "****" in result or "**" in result

    def test_litellm_key_masked(self):
        result = MCPDebug._mask("Bearer sk-1234567890abcdef")
        assert result.startswith("Bearer")
        assert "sk-1234567890abcdef" not in result


class TestBuildDebugHeaders:
    def test_basic_no_auth(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={"host": "localhost"},
            oauth2_headers=None,
            litellm_api_key=None,
            auth_resolution="no-auth",
            server_url="https://mcp.example.com",
            server_auth_type="oauth2",
        )
        assert headers["x-mcp-debug-inbound-auth"] == "(none)"
        assert headers["x-mcp-debug-oauth2-token"] == "(none)"
        assert headers["x-mcp-debug-auth-resolution"] == "no-auth"
        assert headers["x-mcp-debug-outbound-url"] == "https://mcp.example.com"
        assert headers["x-mcp-debug-server-auth-type"] == "oauth2"

    def test_litellm_key_in_dedicated_header(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={
                "x-litellm-api-key": "Bearer sk-1234567890abcdef",
                "host": "localhost",
            },
            oauth2_headers=None,
            litellm_api_key="Bearer sk-1234567890abcdef",
            auth_resolution="no-auth",
            server_url="https://mcp.example.com",
            server_auth_type="oauth2",
        )
        assert "x-litellm-api-key=" in headers["x-mcp-debug-inbound-auth"]
        assert headers["x-mcp-debug-oauth2-token"] == "(none)"

    def test_same_key_flagged(self):
        """When Authorization and x-litellm-api-key carry the same token."""
        headers = MCPDebug.build_debug_headers(
            inbound_headers={
                "authorization": "Bearer sk-1234567890abcdef",
            },
            oauth2_headers={"Authorization": "Bearer sk-1234567890abcdef"},
            litellm_api_key="Bearer sk-1234567890abcdef",
            auth_resolution="oauth2-passthrough",
            server_url="https://mcp.example.com",
            server_auth_type="oauth2",
        )
        assert "SAME_AS_LITELLM_KEY" in headers["x-mcp-debug-oauth2-token"]

    def test_different_tokens_not_flagged(self):
        """When OAuth2 token is different from LiteLLM key."""
        headers = MCPDebug.build_debug_headers(
            inbound_headers={
                "x-litellm-api-key": "Bearer sk-litellm-key-here",
                "authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.atlassian",
            },
            oauth2_headers={
                "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.atlassian"
            },
            litellm_api_key="Bearer sk-litellm-key-here",
            auth_resolution="oauth2-passthrough",
            server_url="https://mcp.atlassian.com/v1/mcp",
            server_auth_type="oauth2",
        )
        assert "SAME_AS_LITELLM_KEY" not in headers["x-mcp-debug-oauth2-token"]
        assert headers["x-mcp-debug-auth-resolution"] == "oauth2-passthrough"

    def test_m2m_resolution(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={"x-litellm-api-key": "Bearer sk-key"},
            oauth2_headers=None,
            litellm_api_key="Bearer sk-key",
            auth_resolution="m2m-client-credentials",
            server_url="https://mcp.example.com",
            server_auth_type="oauth2",
        )
        assert headers["x-mcp-debug-auth-resolution"] == "m2m-client-credentials"

    def test_missing_server_url(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={},
            oauth2_headers=None,
            litellm_api_key=None,
            auth_resolution="no-auth",
            server_url=None,
            server_auth_type=None,
        )
        assert headers["x-mcp-debug-outbound-url"] == "(unknown)"
        assert headers["x-mcp-debug-server-auth-type"] == "(none)"

    def test_all_five_headers_present(self):
        headers = MCPDebug.build_debug_headers(
            inbound_headers={},
            oauth2_headers=None,
            litellm_api_key=None,
            auth_resolution="no-auth",
            server_url=None,
            server_auth_type=None,
        )
        expected_keys = {
            "x-mcp-debug-inbound-auth",
            "x-mcp-debug-oauth2-token",
            "x-mcp-debug-auth-resolution",
            "x-mcp-debug-outbound-url",
            "x-mcp-debug-server-auth-type",
        }
        assert set(headers.keys()) == expected_keys


class TestWrapSendWithDebugHeaders:
    def test_injects_headers(self):
        captured = []

        async def mock_send(message):
            captured.append(message)

        wrapped = MCPDebug.wrap_send_with_debug_headers(
            mock_send, {"x-mcp-debug-test": "value123"}
        )

        message = {"type": "http.response.start", "status": 200, "headers": []}
        asyncio.run(wrapped(message))

        assert len(captured) == 1
        headers = dict(captured[0]["headers"])
        assert headers[b"x-mcp-debug-test"] == b"value123"

    def test_body_messages_unchanged(self):
        captured = []

        async def mock_send(message):
            captured.append(message)

        wrapped = MCPDebug.wrap_send_with_debug_headers(
            mock_send, {"x-mcp-debug-test": "value"}
        )

        body_msg = {"type": "http.response.body", "body": b"hello"}
        asyncio.run(wrapped(body_msg))

        assert captured[0] == body_msg


@pytest.mark.asyncio
@pytest.mark.parametrize("source", tuple(AuthResolution))
@pytest.mark.parametrize("method", ("GET", "DELETE", "POST"))
async def test_debug_defers_resolution_until_first_frame_only_for_post(source: AuthResolution, method: str) -> None:
    captured: Final[list[Message]] = []
    diagnostics: Final = MCPAuthDiagnostics()

    async def send(message: Message) -> None:
        captured.append(message)

    wrapped: Final = MCPDebug.wrap_send_with_debug_headers(
        send, diagnostics.headers(), diagnostics.headers, request_method=method
    )
    await wrapped({"type": "http.response.start", "status": 200, "headers": []})
    assert len(captured) == (0 if method == "POST" else 1)
    diagnostics.record("s1", source)
    body: Final[Message] = {"type": "http.response.body", "body": b"data: pong\n\n", "more_body": True}
    await wrapped(body)
    assert dict(captured[0]["headers"])[b"x-mcp-debug-auth-resolution"] == (
        source.value.encode() if method == "POST" else b"unresolved"
    )
    assert captured[1] == body


@pytest.mark.asyncio
async def test_early_stream_frame_reports_unresolved_without_waiting() -> None:
    captured: Final[list[Message]] = []
    diagnostics: Final = MCPAuthDiagnostics()

    async def send(message: Message) -> None:
        captured.append(message)

    wrapped: Final = MCPDebug.wrap_send_with_debug_headers(send, {}, diagnostics.headers, request_method="POST")
    await wrapped({"type": "http.response.start", "status": 200, "headers": []})
    await wrapped({"type": "http.response.body", "body": b": ping\n\n", "more_body": True})
    diagnostics.record("s1", AuthResolution.stored_user_token)
    await wrapped({"type": "http.response.body", "body": b"data: pong\n\n", "more_body": False})
    assert len(captured) == 3
    assert dict(captured[0]["headers"])[b"x-mcp-debug-auth-resolution"] == b"unresolved"


def test_diagnostics_keep_requests_separate_and_do_not_collapse_multiple_servers() -> None:
    alice: Final = MCPAuthDiagnostics()
    bob: Final = MCPAuthDiagnostics()
    alice.record("s1", AuthResolution.stored_user_token)
    assert bob.resolution() == "unresolved"
    alice.record("s1", AuthResolution.token_exchange)
    assert alice.resolution() == "token-exchange"
    alice.record("s2", AuthResolution.static_token)
    assert alice.resolution() == "multiple"
    assert alice.headers()["x-mcp-debug-auth-resolutions"] == '{"s1":"token-exchange","s2":"static-token"}'


@pytest.mark.asyncio
async def test_concurrent_mcp_messages_record_on_their_own_http_scope() -> None:
    from unittest.mock import MagicMock

    from mcp.server.lowlevel.server import request_ctx
    from mcp.shared.context import RequestContext
    from starlette.requests import Request

    from litellm.proxy._experimental.mcp_server.mcp_debug import (
        MCP_AUTH_DIAGNOSTICS_SCOPE_KEY,
        record_auth_resolution,
    )

    session: Final = MagicMock()
    first: Final = MCPAuthDiagnostics()
    second: Final = MCPAuthDiagnostics()

    async def record(diagnostics: MCPAuthDiagnostics, source: AuthResolution) -> None:
        context: Final = RequestContext(
            request_id=1, meta=None, session=session, lifespan_context=None,
            request=Request({"type": "http", MCP_AUTH_DIAGNOSTICS_SCOPE_KEY: diagnostics}),
        )
        token: Final = request_ctx.set(context)
        try:
            await asyncio.sleep(0)
            record_auth_resolution("same-server", source)
        finally:
            request_ctx.reset(token)

    await asyncio.gather(record(first, AuthResolution.stored_user_token), record(second, AuthResolution.per_request_header))
    assert first.resolution() == "stored-user-token"
    assert second.resolution() == "per-request-header"
