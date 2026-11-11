"""
Tests for SecurityHeadersMiddleware.

Verifies anti-framing / content-type headers are present on every response and
that HSTS is opt-in via LITELLM_ENABLE_HSTS.
"""

from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from litellm.proxy.middleware.security_headers_middleware import (
    SecurityHeadersMiddleware,
)


def _make_client(handler):
    app = Starlette(routes=[Route("/", handler)])
    app.add_middleware(SecurityHeadersMiddleware)
    return TestClient(app)


async def _ok(request):
    return JSONResponse({"ok": True})


def test_is_pure_asgi_not_base_http_middleware():
    """BaseHTTPMiddleware degrades streaming; this must be pure ASGI."""
    assert not issubclass(SecurityHeadersMiddleware, BaseHTTPMiddleware)
    assert "__call__" in SecurityHeadersMiddleware.__dict__


def test_static_security_headers_present():
    resp = _make_client(_ok).get("/")
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["content-security-policy"] == "frame-ancestors 'none'"
    assert resp.headers["x-content-type-options"] == "nosniff"


def test_hsts_absent_by_default(monkeypatch):
    monkeypatch.delenv("LITELLM_ENABLE_HSTS", raising=False)
    resp = _make_client(_ok).get("/")
    assert "strict-transport-security" not in resp.headers


def test_hsts_present_when_enabled(monkeypatch):
    monkeypatch.setenv("LITELLM_ENABLE_HSTS", "true")
    resp = _make_client(_ok).get("/")
    assert resp.headers["strict-transport-security"] == (
        "max-age=31536000; includeSubDomains"
    )


def test_hsts_not_enabled_by_arbitrary_value(monkeypatch):
    monkeypatch.setenv("LITELLM_ENABLE_HSTS", "1")
    resp = _make_client(_ok).get("/")
    assert "strict-transport-security" not in resp.headers


def test_does_not_override_existing_header(monkeypatch):
    """A route that sets its own X-Frame-Options must win."""

    async def custom(request):
        return Response("hi", headers={"X-Frame-Options": "SAMEORIGIN"})

    resp = _make_client(custom).get("/")
    assert resp.headers["x-frame-options"] == "SAMEORIGIN"
    # other headers still applied
    assert resp.headers["x-content-type-options"] == "nosniff"
