"""
Unit tests for the optional fastapi-guard middleware wiring.

Mirrors test_security_headers_middleware.py: minimal app, TestClient,
monkeypatched env, default-off assertion, and the pure-ASGI house-style
assertion. fastapi-guard is an optional dependency (not added to
pyproject by this change), so behavior tests skip when it is absent; the
fail-closed path is exercised unconditionally via a poisoned sys.modules
entry.
"""

import itertools
import sys
from typing import Final

import pytest
from starlette.testclient import TestClient

from litellm._logging import verbose_proxy_logger
from litellm.proxy.middleware.guard_middleware import (
    GuardMiddleware,
    guard_enabled,
)

_IP: Final = itertools.count(1)

try:
    import guard  # noqa: F401  # availability probe for skip logic

    _GUARD_AVAILABLE: Final = True
except ImportError:
    _GUARD_AVAILABLE: Final = False


def _unique_ip() -> str:
    n: Final = next(_IP)
    return f"198.51.{n // 250}.{(n % 250) + 1}"


def test_is_pure_asgi_not_base_http_middleware():
    """BaseHTTPMiddleware degrades streaming; this must be pure ASGI."""
    from starlette.middleware.base import BaseHTTPMiddleware

    assert not issubclass(GuardMiddleware, BaseHTTPMiddleware)
    assert "__call__" in GuardMiddleware.__dict__


def test_guard_enabled_flag(monkeypatch):
    monkeypatch.delenv("LITELLM_GUARD_ENABLED", raising=False)
    assert guard_enabled() is False
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    assert guard_enabled() is True


def _build_app():
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def ping(request):
        return JSONResponse({"ok": True})

    app: Final = Starlette(routes=[Route("/ping", ping, methods=["GET"])])
    app.add_middleware(GuardMiddleware)
    return app


def _build_stacked_app():
    """The proxy's nesting: guard innermost, then CORS, then security
    headers outermost (Starlette makes the last-added middleware
    outermost), so guard rejections leave with their headers."""
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from litellm.proxy.middleware.security_headers_middleware import (
        SecurityHeadersMiddleware,
    )

    async def ping(request):
        return JSONResponse({"ok": True})

    app: Final = Starlette(routes=[Route("/ping", ping, methods=["GET"])])
    app.add_middleware(GuardMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://ok.example"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )
    app.add_middleware(SecurityHeadersMiddleware)
    return app


class _ClientIPInjector:
    """TestClient hardcodes scope["client"]; the middleware stack must see
    the test IP, so mutate the scope before it reaches the app."""

    def __init__(self, asgi_app, client_ip):
        self._asgi_app: Final = asgi_app
        self._client_ip: Final = client_ip

    async def __call__(self, scope, receive, send):
        # TestClient hardcodes scope["client"]; the middleware stack must see
        # the TEST-NET address, so build a new scope rather than mutate.
        screened: Final = {**scope, "client": (self._client_ip, 50000)} if scope["type"] == "http" else scope
        await self._asgi_app(screened, receive, send)


def _client(client_ip, asgi_app=None):
    return TestClient(_ClientIPInjector(asgi_app or _build_app(), client_ip))


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LITELLM_GUARD_ENABLED", raising=False)
    client: Final = _client(_unique_ip())
    assert client.get("/ping").status_code == 200


def test_enabled_without_package_fails_closed(monkeypatch):
    """Flag on + package missing: every request is rejected with 503 and
    the misconfiguration is logged exactly once. An operator who enabled
    the guard must not get unscreened passthrough."""
    monkeypatch.setitem(sys.modules, "guard", None)
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    error_messages: Final = []
    monkeypatch.setattr(verbose_proxy_logger, "error", lambda msg, *args: error_messages.append(msg))
    client: Final = _client(_unique_ip())
    first: Final = client.get("/ping")
    second: Final = client.get("/ping")
    assert first.status_code == 503
    assert second.status_code == 503
    assert first.json()["error"]["type"] == "guard_unavailable_error"
    assert len(error_messages) == 1


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_blocked_ip_is_rejected(monkeypatch):
    blocked: Final = _unique_ip()
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", blocked)
    with _client(blocked) as client:
        assert client.get("/ping").status_code == 403


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_untrusted_forwarded_header_is_ignored(monkeypatch):
    """No trusted proxies configured: X-Forwarded-For must not override the
    connecting address, or a private-network caller could spoof its way
    past allowlists, limits, and bans."""
    blocked: Final = "10.4.5.6"
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", blocked)
    with _client(blocked) as client:
        rejected: Final = client.get("/ping", headers={"x-forwarded-for": "198.51.100.9"})
    assert rejected.status_code == 403


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_rejected_response_keeps_cors_and_security_headers(monkeypatch):
    """Guard is nested inside the CORS and security-headers wrappers, so a
    guard rejection carries their headers (readable cross-origin errors)."""
    blocked: Final = _unique_ip()
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", blocked)
    with _client(blocked, _build_stacked_app()) as client:
        rejected: Final = client.get("/ping", headers={"Origin": "https://ok.example"})
    assert rejected.status_code == 403
    # Both wrappers ran on the guard rejection. guard sets its own
    # anti-framing header value and litellm's setdefault defers to it, so
    # presence (not a specific value) is the property pinned here.
    assert rejected.headers["x-frame-options"] in ("DENY", "SAMEORIGIN")
    assert rejected.headers["content-security-policy"] == "frame-ancestors 'none'"
    assert rejected.headers["access-control-allow-origin"] == "https://ok.example"


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_rate_limit_returns_429(monkeypatch):
    client_ip: Final = _unique_ip()
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_RATE_LIMIT", "2")
    monkeypatch.setenv("LITELLM_GUARD_RATE_LIMIT_WINDOW", "60")
    with _client(client_ip) as client:
        codes: Final = [client.get("/ping").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_passive_mode_never_blocks(monkeypatch):
    blocked: Final = _unique_ip()
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_PASSIVE_MODE", "true")
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", blocked)
    with _client(blocked) as client:
        assert client.get("/ping").status_code == 200


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_redis_stays_off_without_config(monkeypatch):
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    from litellm.proxy.middleware.guard_middleware import _build_guard_config

    config: Final = _build_guard_config()
    assert config.enable_redis is False
    assert config.redis_url is None
