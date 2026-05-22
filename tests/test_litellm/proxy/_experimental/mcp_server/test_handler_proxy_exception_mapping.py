"""
Regression coverage for the ASGI MCP handlers' ``ProxyException`` mapping.

Auth and router failures inside ``extract_mcp_auth_context`` raise
``ProxyException``, which is the rest-of-proxy convention but is not an
``HTTPException``. Without explicit handling, the handler's catch-all
``except Exception`` collapsed those into a generic
``500 {"error": "MCP request failed", "details": ""}`` response — losing the
real status code (401/403) and any auth-bootstrap headers (``WWW-Authenticate``)
that MCP clients depend on for OAuth discovery. These tests pin the corrected
behavior: status code preserved, ``headers`` preserved, message preserved.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
)

from litellm.proxy._types import ProxyException


def test_proxy_exception_to_http_exception_preserves_status_message_headers():
    from litellm.proxy._experimental.mcp_server.server import (
        _proxy_exception_to_http_exception,
    )

    proxy_exc = ProxyException(
        message="Unauthorized",
        type="auth_error",
        param=None,
        code=401,
        headers={
            "WWW-Authenticate": (
                "Bearer authorization_uri="
                "http://localhost:4000/.well-known/oauth-authorization-server/atlassian1"
            )
        },
    )

    http_exc = _proxy_exception_to_http_exception(proxy_exc)

    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 401
    assert http_exc.detail == "Unauthorized"
    assert http_exc.headers is not None
    assert http_exc.headers["WWW-Authenticate"].startswith("Bearer authorization_uri=")


def test_proxy_exception_to_http_exception_handles_non_numeric_code():
    """``ProxyException.__init__`` stringifies ``code``; ``int("None")`` would
    rewrite the auth error as a 500 if we coerced naively. Default to 500
    instead so unknown statuses don't masquerade as auth errors."""
    from litellm.proxy._experimental.mcp_server.server import (
        _proxy_exception_to_http_exception,
    )

    proxy_exc = ProxyException(
        message="boom", type="internal_error", param=None, code=None
    )
    http_exc = _proxy_exception_to_http_exception(proxy_exc)
    assert http_exc.status_code == 500
    assert http_exc.detail == "boom"


def test_proxy_exception_to_http_exception_treats_empty_headers_as_none():
    """``HTTPException(headers={})`` is technically valid but signals "I have
    auth hints" downstream — collapse the empty dict to ``None`` so Starlette
    doesn't emit a header block for nothing."""
    from litellm.proxy._experimental.mcp_server.server import (
        _proxy_exception_to_http_exception,
    )

    proxy_exc = ProxyException(
        message="Bad request", type="bad_request", param=None, code=400
    )
    http_exc = _proxy_exception_to_http_exception(proxy_exc)
    assert http_exc.status_code == 400
    assert http_exc.headers is None


@pytest.mark.asyncio
async def test_streamable_handler_re_raises_proxy_exception_as_http_exception():
    """End-to-end: an auth-stack ``ProxyException(401)`` with WWW-Authenticate
    must surface to the ASGI layer as a real ``HTTPException`` (so Starlette
    emits 401 + the header) rather than the catch-all 500 wrapper."""
    from litellm.proxy._experimental.mcp_server.server import (
        handle_streamable_http_mcp,
    )

    scope = {
        "type": "http",
        "path": "/mcp/atlassian1/mcp",
        "method": "POST",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer not-a-real-litellm-key")],
    }

    auth_failure = ProxyException(
        message="Authentication Error, Invalid proxy server token passed.",
        type="auth_error",
        param="Authorization",
        code=401,
        headers={
            "WWW-Authenticate": (
                "Bearer authorization_uri="
                "http://localhost:4000/.well-known/oauth-authorization-server/atlassian1"
            )
        },
    )

    async def fake_receive():
        return {"type": "http.disconnect"}

    sent = []

    async def fake_send(msg):
        sent.append(msg)

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new=AsyncMock(side_effect=auth_failure),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await handle_streamable_http_mcp(scope, fake_receive, fake_send)

    assert excinfo.value.status_code == 401
    assert excinfo.value.detail == auth_failure.message
    assert excinfo.value.headers is not None
    assert "Bearer authorization_uri=" in excinfo.value.headers["WWW-Authenticate"]
    # Sanity check: no 500 body was sent. The handler should let Starlette
    # produce the response, not call ``send`` itself with the legacy 500 shape.
    assert sent == []


@pytest.mark.asyncio
async def test_sse_handler_re_raises_proxy_exception_as_http_exception():
    """Mirror of the streamable test, for the SSE entry point."""
    from litellm.proxy._experimental.mcp_server.server import handle_sse_mcp

    scope = {
        "type": "http",
        "path": "/mcp/atlassian1/sse",
        "method": "GET",
        "query_string": b"",
        "headers": [],
    }

    auth_failure = ProxyException(
        message="Forbidden", type="auth_error", param=None, code=403
    )

    async def fake_receive():
        return {"type": "http.disconnect"}

    sent = []

    async def fake_send(msg):
        sent.append(msg)

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new=AsyncMock(side_effect=auth_failure),
    ):
        with pytest.raises(HTTPException) as excinfo:
            await handle_sse_mcp(scope, fake_receive, fake_send)

    assert excinfo.value.status_code == 403
    assert excinfo.value.detail == "Forbidden"
    assert sent == []


@pytest.mark.asyncio
async def test_streamable_handler_still_collapses_unexpected_exceptions_to_500():
    """The new ``ProxyException`` catch must not regress the existing graceful
    500 fallback for arbitrary exceptions."""
    from litellm.proxy._experimental.mcp_server.server import (
        handle_streamable_http_mcp,
    )

    scope = {
        "type": "http",
        "path": "/mcp/atlassian1/mcp",
        "method": "POST",
        "query_string": b"",
        "headers": [],
    }

    async def fake_receive():
        return {"type": "http.disconnect"}

    sent = []

    async def fake_send(msg):
        sent.append(msg)

    with patch(
        "litellm.proxy._experimental.mcp_server.server.extract_mcp_auth_context",
        new=AsyncMock(side_effect=RuntimeError("kaboom")),
    ):
        # Should NOT raise — handler sends a JSONResponse via ``send``.
        await handle_streamable_http_mcp(scope, fake_receive, fake_send)

    # JSONResponse sends start + body messages.
    statuses = [m.get("status") for m in sent if m.get("type") == "http.response.start"]
    assert statuses == [500]
