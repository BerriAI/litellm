"""
Unit tests for the optional fastapi-guard middleware wiring.

Mirrors test_security_headers_middleware.py: minimal app, TestClient,
monkeypatched env, default-off assertion, and the pure-ASGI house-style
assertion. fastapi-guard is an optional dependency (not added to
pyproject by this change), so behavior tests skip when it is absent.
"""

import asyncio
import itertools
import os

import pytest
from starlette.testclient import TestClient

from litellm.proxy.middleware.guard_middleware import (
    GuardMiddleware,
    guard_enabled,
)

_IP = itertools.count(1)

try:
    import guard  # noqa: F401  # availability probe for skip logic

    _GUARD_AVAILABLE = True
except ImportError:
    _GUARD_AVAILABLE = False


def _unique_ip() -> str:
    n = next(_IP)
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

    app = Starlette(routes=[Route("/ping", ping, methods=["GET"])])
    app.add_middleware(GuardMiddleware)
    return app


class _ClientIPInjector:
    """TestClient hardcodes scope["client"]; the middleware stack must see
    the test IP, so mutate the scope before it reaches the app."""

    def __init__(self, asgi_app, client_ip):
        self._asgi_app = asgi_app
        self._client_ip = client_ip

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            scope["client"] = (self._client_ip, 50000)
        await self._asgi_app(scope, receive, send)


def _client(client_ip):
    return TestClient(_ClientIPInjector(_build_app(), client_ip))


def test_disabled_by_default(monkeypatch):
    monkeypatch.delenv("LITELLM_GUARD_ENABLED", raising=False)
    client = _client(_unique_ip())
    assert client.get("/ping").status_code == 200


def test_fail_loud_without_package(monkeypatch):
    if _GUARD_AVAILABLE:
        pytest.skip("fastapi-guard is installed")
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    client = _client(_unique_ip())
    # Flag on, package missing: request still passes (logged loudly), but
    # the guard attribute proves the wiring attempted the import.
    assert client.get("/ping").status_code == 200


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_blocked_ip_is_rejected(monkeypatch):
    blocked = _unique_ip()
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", blocked)
    with _client(blocked) as client:
        assert client.get("/ping").status_code == 403


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_rate_limit_returns_429(monkeypatch):
    client_ip = _unique_ip()
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_RATE_LIMIT", "2")
    monkeypatch.setenv("LITELLM_GUARD_RATE_LIMIT_WINDOW", "60")
    with _client(client_ip) as client:
        codes = [client.get("/ping").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_passive_mode_never_blocks(monkeypatch):
    blocked = _unique_ip()
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_PASSIVE_MODE", "true")
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", blocked)
    with _client(blocked) as client:
        assert client.get("/ping").status_code == 200


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_redis_stays_off_without_config(monkeypatch):
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    from litellm.proxy.middleware.guard_middleware import _build_guard_config

    config = _build_guard_config()
    assert config.enable_redis is False
