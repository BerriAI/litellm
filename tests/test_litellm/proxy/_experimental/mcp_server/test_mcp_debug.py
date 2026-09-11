"""
Tests for MCPDebug — MCP OAuth2 debug response headers.
"""

import asyncio
from typing import Final

import pytest
from starlette.types import Message

from litellm.proxy._experimental.mcp_server.outbound_credentials.types import AuthResolution

import httpx

from litellm.proxy._experimental.mcp_server.mcp_debug import (
    MCP_DEBUG_REQUEST_HEADER,
    MCPDebug,
    describe_upstream_http_failure,

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


class TestDescribeUpstreamHttpFailure:
    @staticmethod
    def _status_error(*, body: bytes, response_body: bytes | None = None) -> httpx.HTTPStatusError:
        request = httpx.Request(
            "POST",
            "https://upstream.example/apis/mcp",
            headers={"Authorization": "Bearer secret-token-abcdef0123456789", "Content-Type": "application/json" if body.startswith(b"{") else "application/x-www-form-urlencoded"},
            content=body,
        )
        response = (
            httpx.Response(500, request=request, content=response_body)
            if response_body is not None
            else httpx.Response(500, request=request, stream=httpx.ByteStream(b'{"error":"boom"}'))
        )
        return httpx.HTTPStatusError("500", request=request, response=response)

    def test_includes_method_url_status_and_request_body(self):
        exc = self._status_error(
            body=b'{"method":"initialize","jsonrpc":"2.0","id":0}',
            response_body=b'{"error":"boom"}',
        )
        described = describe_upstream_http_failure(exc)
        assert described is not None
        assert "POST https://upstream.example/ -> HTTP 500" in described
        assert '{"method":"initialize"' in described
        assert 'response body: {"error":"boom"}' in described

    def test_masks_authorization_header_and_secret_body_fields(self):
        exc = self._status_error(
            body=b"grant_type=client_credentials&client_id=abc&client_secret=super-secret-value-1234",
            response_body=b"{}",
        )
        described = describe_upstream_http_failure(exc)
        assert described is not None
        assert "secret-token-abcdef0123456789" not in described
        assert "super-secret-value-1234" not in described
        assert "client_id=abc" in described
        assert "client_secret=" in described

    def test_reports_unread_streamed_response_body(self):
        described = describe_upstream_http_failure(self._status_error(body=b"{}"))
        assert described is not None
        assert "response body: (not read)" in described

    def test_finds_response_behind_cause_chain(self):
        wrapper = RuntimeError("token minting failed")
        wrapper.__cause__ = self._status_error(body=b"{}", response_body=b'{"error":"invalid_client"}')
        described = describe_upstream_http_failure(wrapper)
        assert described is not None
        assert "invalid_client" in described

    def test_returns_none_without_http_response(self):
        assert describe_upstream_http_failure(ConnectionError("refused")) is None


@pytest.mark.parametrize("body", [
    b'{"password":"first second","token":"demo-secret"}',
    b'{"nested":[{"access_token":"first,second"}]}',
    b'client%5Fsecret=first+second&token=demo-secret',
])
def test_failure_log_fully_redacts_structured_secrets(body):
    request = httpx.Request("POST", "https://upstream/mcp?credential=query-secret",
        headers={"X-Custom-Credential": "custom-secret"}, content=body)
    response = httpx.Response(500, request=request, content=body)
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failure", request=request, response=response))
    assert detail is not None
    for secret in ("first", "second", "demo-secret", "custom-secret", "query-secret"):
        assert secret not in detail


def test_failure_log_omits_unstructured_body():
    request = httpx.Request("POST", "https://upstream/mcp", content=b"arbitrary-secret")
    response = httpx.Response(500, request=request, content=b"<html>arbitrary-secret</html>")
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failure", request=request, response=response))
    assert detail is not None
    assert "arbitrary-secret" not in detail
    assert "omitted" in detail


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["error", "empty", "large", "timeout", "read_failure", "closed", "success", "cancel"])
async def test_error_capture_is_bounded_and_preserves_success_and_cancellation(mode):
    from litellm.proxy._experimental.mcp_server.mcp_debug import capture_upstream_error_response

    class Stream(httpx.AsyncByteStream):
        def __init__(self):
            self.reads = 0

        async def __aiter__(self):
            self.reads += 1
            if mode == "timeout":
                await asyncio.sleep(10)
            if mode == "closed":
                raise httpx.StreamClosed()
            if mode == "read_failure":
                raise httpx.ReadError("private-read-error")
            if mode == "cancel":
                raise asyncio.CancelledError
            yield b"" if mode == "empty" else b'{"error":"missing_scope","password":"first second"}' if mode != "large" else b"x" * 20000

    stream = Stream()
    request = httpx.Request("POST", "https://upstream/mcp")
    response = httpx.Response(200 if mode == "success" else 500, request=request, stream=stream)
    if mode == "cancel":
        with pytest.raises(asyncio.CancelledError):
            await capture_upstream_error_response(response)
        return
    await capture_upstream_error_response(response)
    if mode == "success":
        assert stream.reads == 0
        assert await response.aread() == b'{"error":"missing_scope","password":"first second"}'
        return
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failure", request=request, response=response))
    assert detail is not None
    assert "first" not in detail and "second" not in detail and "private-read-error" not in detail
    expected = {"empty": "(empty)", "error": "missing_scope", "large": "capture limit", "timeout": "read failed", "read_failure": "read failed", "closed":"read failed"}
    assert expected[mode] in detail
    if mode == "error":
        assert await response.aread() == b'{"error":"missing_scope","password":"first second"}'


@pytest.mark.parametrize("body", [b"", b'"scalar"', b'{"hint":"line1\\nline2"}', b'{"hint":"' + b'x' * 600 + b'"}'])
def test_failure_preview_handles_empty_scalar_control_and_long_bodies(body):
    request = httpx.Request("POST", "https://user:secret@upstream/mcp?key=private#private", content=body)
    response = httpx.Response(500, request=request, content=body)
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failure", request=request, response=response))
    assert detail is not None
    assert "private" not in detail and "user:secret" not in detail and "\n" not in detail
    if not body:
        assert "(empty)" in detail
    elif body.startswith(b'"'):
        assert "omitted" in detail
    elif len(body) > 512:
        assert "truncated" in detail and len(detail) < 1300
    else:
        assert "line1\\nline2" in detail


@pytest.mark.asyncio
@pytest.mark.parametrize("slow_error", [False, True])
async def test_error_capture_preserves_httpx_auth_retry(slow_error):
    from litellm.proxy._experimental.mcp_server.mcp_debug import capture_upstream_error_response

    class RetryAuth(httpx.Auth):
        def auth_flow(self, request):
            response = yield request
            if response.status_code == 401:
                request.headers["Authorization"] = "Bearer refreshed"
                yield request

    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(10)
            yield b'{"error":"expired_token"}'

    def upstream(request):
        if request.headers.get("Authorization"):
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(401, stream=SlowStream()) if slow_error else httpx.Response(401, json={"error":"expired_token"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream), auth=RetryAuth(),
            event_hooks={"response":[capture_upstream_error_response]}) as client:
        response = await client.get("https://upstream/mcp")
    assert response.status_code == 200 and response.json() == {"ok":True}
    if slow_error:
        assert response.history[0].content == b""
    else:
        assert response.history[0].json() == {"error":"expired_token"}


def test_failure_diagnostics_without_request_and_with_streamed_request():
    response = httpx.Response(503)
    exc = httpx.HTTPStatusError("failed", request=httpx.Request("GET", "https://upstream"), response=response)
    assert describe_upstream_http_failure(exc) == "HTTP 503 | request unavailable"
    request = httpx.Request("POST", "https://upstream", content=iter((b"private-body",)))
    response = httpx.Response(503, request=request)
    described = describe_upstream_http_failure(httpx.HTTPStatusError("failed", request=request, response=response))
    assert described is not None and "streamed, not captured" in described and "private-body" not in described



def test_deep_error_body_is_bounded_without_exposing_nested_values():
    body = b'{"nested":' * 18 + b'{"password":"hidden-value"}' + b'}' * 18
    request = httpx.Request("POST", "https://upstream/mcp", content=body)
    response = httpx.Response(500, request=request, content=body)
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failure", request=request, response=response))
    assert detail is not None and "hidden-value" not in detail
    assert "nested" in detail and "REDACTED" in detail


@pytest.mark.parametrize("body", [b'client%5Fsecret=first+second&client_id=visible', b'client_secret=first%26second&client_id=visible'])
def test_encoded_form_credentials_are_decoded_before_redaction(body):
    request = httpx.Request("POST", "https://upstream/token", content=body,
        headers={"Content-Type":"application/x-www-form-urlencoded"})
    response = httpx.Response(400, request=request, content=body,
        headers={"Content-Type":"application/x-www-form-urlencoded"})
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failure", request=request, response=response))
    assert detail is not None and "client_id=visible" in detail
    assert "first" not in detail and "second" not in detail

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


@pytest.mark.parametrize("source", ["header", "bearer", "basic", "cookie", "query", "form", "json"])
def test_reflected_credentials_are_removed_from_normal_response_fields(source):
    import base64

    secret = "generic-credential-123"
    headers = {"X-Custom":secret} if source == "header" else {"Authorization":"Bearer " + secret} if source == "bearer" else {"Authorization":"Basic " + base64.b64encode(("client:" + secret).encode()).decode()} if source == "basic" else {"Cookie":"session=" + secret} if source == "cookie" else {}
    request = httpx.Request("POST", "https://upstream/token" + ("?credential=" + secret if source == "query" else ""),
        headers=headers, data={"client_secret":secret} if source == "form" else None,
        json={"nested":{"client_secret":secret}} if source == "json" else None)
    response = httpx.Response(401, request=request, json={"error":"invalid_client", "error_description":"Rejected " + secret})
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failed", request=request, response=response))
    assert detail is not None and "invalid_client" in detail
    assert secret not in detail and "REDACTED" in detail



@pytest.mark.parametrize("secret", ['value"with\ncharacters€', "R"])
def test_reflected_values_are_redacted_before_truncation_without_expanding_replacements(secret):
    request = httpx.Request("POST", "https://upstream/token", json={"client_secret":secret})
    response = httpx.Response(401, request=request, json={"error":"invalid_client", "detail":"x" * 460 + secret})
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failed", request=request, response=response))
    assert detail is not None and "invalid_client" in detail
    assert "value" not in detail and "characters" not in detail and len(detail) < 1400


@pytest.mark.parametrize("headers", [{"Authorization":"Basic !!!"}, {"Cookie":"bad@key=opaque"}])
def test_malformed_auth_headers_do_not_break_failure_diagnostics(headers):
    request = httpx.Request("POST", "https://upstream/token", headers=headers)
    response = httpx.Response(401, request=request, json={"error":"invalid_client"})
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failed", request=request, response=response))
    assert detail is not None and "invalid_client" in detail
    assert "!!!" not in detail and "opaque" not in detail


def test_oversized_request_omits_potentially_reflected_response_credentials():
    request = httpx.Request("POST", "https://upstream/token", content=b"x" * 17000)
    response = httpx.Response(401, request=request, json={"error_description":"unknown-secret"})
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failed", request=request, response=response))
    assert detail is not None and "capture limit" in detail and "credentials unavailable" in detail
    assert "unknown-secret" not in detail



@pytest.mark.asyncio
async def test_streamed_error_redacts_reflected_credentials_before_capture():
    import json
    from litellm.proxy._experimental.mcp_server.mcp_debug import capture_upstream_error_response

    secret = "generic-credential-123"
    request = httpx.Request("POST", "https://upstream/token", data={"client_secret":secret})
    raw = json.dumps({"error":"invalid_client", "error_description":"Rejected " + secret}).encode()
    response = httpx.Response(401, request=request, stream=httpx.ByteStream(raw))
    await capture_upstream_error_response(response)
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failed", request=request, response=response))
    assert detail is not None and "invalid_client" in detail and "Rejected" in detail
    assert secret not in detail and "REDACTED" in detail
    assert await response.aread() == raw


@pytest.mark.parametrize("path", ["/credential-path-value/mcp", "/oauth/credential-path-value/token"])
def test_failure_diagnostics_omit_credential_bearing_url_paths(path):
    request = httpx.Request("POST", "https://upstream.example" + path)
    response = httpx.Response(401, request=request, json={"error": "access_denied"})
    error = httpx.HTTPStatusError("denied", request=request, response=response)
    diagnostic = describe_upstream_http_failure(error)
    assert diagnostic is not None
    assert "credential-path-value" not in diagnostic
    assert "POST https://upstream.example/ -> HTTP 401" in diagnostic
    assert "access_denied" in diagnostic


def test_deep_request_omits_response_when_credentials_cannot_be_inspected():
    from litellm.proxy._experimental.mcp_server.utils import MAX_STRUCTURED_CONTENT_SCAN_DEPTH

    raw = "[" * (MAX_STRUCTURED_CONTENT_SCAN_DEPTH + 1) + '{"client_secret":"nested-credential"}' + "]" * (MAX_STRUCTURED_CONTENT_SCAN_DEPTH + 1)
    request = httpx.Request("POST", "https://upstream/token", content=raw, headers={"Content-Type": "application/json"})
    response = httpx.Response(401, request=request, json={"error_description": "Rejected nested-credential"})
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failed", request=request, response=response))
    assert detail is not None and "HTTP 401" in detail
    assert "response body: (omitted: request credentials unavailable)" in detail
    assert "nested-credential" not in detail


@pytest.mark.parametrize("field", ["accessToken", "refreshToken", "clientSecret", "apikey", "CLIENTASSERTION", "cost_token"])
@pytest.mark.parametrize("encoding", ["json", "form"])
def test_compact_credential_fields_and_reflected_values_are_redacted(field, encoding):
    secret = "generic-private-value"
    fields = {field: secret}
    request = httpx.Request("POST", "https://upstream/token", json=fields if encoding == "json" else None,
        data=fields if encoding == "form" else None)
    response = httpx.Response(401, request=request, json={field: secret, "error": "invalid_client", "detail": "Rejected " + secret})
    detail = describe_upstream_http_failure(httpx.HTTPStatusError("failed", request=request, response=response))
    assert detail is not None and "invalid_client" in detail
    assert "REDACTED" in detail and secret not in detail
