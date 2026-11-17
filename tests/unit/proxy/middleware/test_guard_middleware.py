"""
Unit tests for the optional fastapi-guard middleware wiring.

Mirrors test_security_headers_middleware.py: minimal app, TestClient,
monkeypatched env, default-off assertion, and the pure-ASGI house-style
assertion. fastapi-guard is an optional dependency (not added to
pyproject by this change), so behavior tests skip when it is absent; the
fail-closed path is exercised unconditionally via a poisoned sys.modules
entry.
"""

import gzip
import itertools
import sys
from typing import Final

import pytest
from starlette.testclient import TestClient

from litellm._logging import verbose_proxy_logger
from litellm.proxy.middleware.guard_middleware import (
    GuardMiddleware,
    _networks,
    _resolve_client,
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
    """The proxy's nesting: guard innermost, then request-body
    decompression, then CORS, then security headers outermost (Starlette
    makes the last-added middleware outermost), so guard inspects decoded
    bodies and its rejections leave with the wrappers' headers."""
    from starlette.applications import Starlette
    from starlette.middleware.cors import CORSMiddleware
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    from litellm.proxy.middleware.request_body_decompression_middleware import (
        RequestBodyDecompressionMiddleware,
    )
    from litellm.proxy.middleware.security_headers_middleware import (
        SecurityHeadersMiddleware,
    )

    async def ping(request):
        return JSONResponse({"ok": True})

    app: Final = Starlette(routes=[Route("/ping", ping, methods=["GET"])])
    app.add_middleware(GuardMiddleware)
    app.add_middleware(RequestBodyDecompressionMiddleware)
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


class _StubSecurityConfig:
    """Captures the kwargs the middleware assembles from the environment."""

    def __init__(self, **kwargs):  # kwargs-ok: mirrors the SecurityConfig keyword surface the wrapper calls
        self.kwargs: Final = kwargs


class _StubSecurityMiddleware:
    """Pass-through stand-in recording how the wrapper wired it."""

    instances: Final = []
    seen_paths: Final = []

    def __init__(self, app, config=None):
        self.app: Final = app
        self.config: Final = config
        _StubSecurityMiddleware.instances.append(self)

    async def __call__(self, scope, receive, send):
        _StubSecurityMiddleware.seen_paths.append(scope["path"])
        await self.app(scope, receive, send)


def _install_stub_guard(monkeypatch):
    """Stub guard module so the wiring is exercised where the optional
    package is absent."""
    import types

    stub_module: Final = types.ModuleType("guard")
    stub_module.SecurityConfig = _StubSecurityConfig
    stub_module.SecurityMiddleware = _StubSecurityMiddleware
    monkeypatch.setitem(sys.modules, "guard", stub_module)
    _StubSecurityMiddleware.instances.clear()
    _StubSecurityMiddleware.seen_paths.clear()
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")


def test_enabled_wires_guard_with_env_config(monkeypatch):
    """Flag on + importable guard module: the wrapper builds the config
    from the environment and delegates through the guard middleware. Uses
    a stub module, so the wiring is exercised in CI where the real
    optional package is absent."""
    import types

    stub_module: Final = types.ModuleType("guard")
    stub_module.SecurityConfig = _StubSecurityConfig
    stub_module.SecurityMiddleware = _StubSecurityMiddleware
    monkeypatch.setitem(sys.modules, "guard", stub_module)
    _StubSecurityMiddleware.instances.clear()
    monkeypatch.delenv("LITELLM_GUARD_REDIS_URL", raising=False)
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_RATE_LIMIT", "33")
    client: Final = _client(_unique_ip())
    assert client.get("/ping").status_code == 200
    wired: Final = _StubSecurityMiddleware.instances[-1]
    assert isinstance(wired.config, _StubSecurityConfig)
    assert wired.config.kwargs["rate_limit"] == 33
    assert wired.config.kwargs["enable_redis"] is False
    assert wired.config.kwargs["trusted_proxies"] == ()
    assert wired.config.kwargs["exclude_paths"] == []


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
def test_compressed_attack_body_is_inspected_after_decoding(monkeypatch):
    """Guard is nested inside RequestBodyDecompressionMiddleware, so it
    inspects the decoded body: a gzip-encoded attack payload is rejected,
    while a benign gzip body passes through to routing (405 on GET-only)."""
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    headers: Final = {"content-encoding": "gzip", "content-type": "application/json"}
    attack: Final = gzip.compress(b'{"prompt": "<script>alert(document.cookie)</script>"}')
    benign: Final = gzip.compress(b'{"prompt": "summarize this paragraph"}')
    with _client(_unique_ip(), _build_stacked_app()) as client:
        assert client.post("/ping", content=benign, headers=headers).status_code == 405
        assert client.post("/ping", content=attack, headers=headers).status_code == 400


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


def test_credential_headers_are_redacted_from_guard_logs(monkeypatch):
    """Guard's blocked-request warnings print request headers; its default
    redaction only knows authorization, proxy-authorization, cookie and
    x-api-key, so every other header the proxy accepts as a key (and the
    configured custom key header) must be added."""
    from litellm.proxy import proxy_server
    from litellm.types.proxy.auth.special_headers import SpecialHeaders

    _install_stub_guard(monkeypatch)
    monkeypatch.setattr(proxy_server, "general_settings", {"litellm_key_header_name": "X-Custom-Key"})
    assert _client(_unique_ip()).get("/ping").status_code == 200
    redacted: Final = _StubSecurityMiddleware.instances[-1].config.kwargs["log_sensitive_headers"]
    assert SpecialHeaders.litellm_credential_header_names() <= redacted
    assert "x-custom-key" in redacted


def _server_config(server, monkeypatch, guard_on):
    """The uvicorn.Config the proxy builds for the uvicorn and gunicorn
    worker start paths."""
    from unittest.mock import MagicMock

    import uvicorn
    from starlette.responses import PlainTextResponse

    from litellm.proxy.proxy_cli import ProxyInitializationHelpers

    if guard_on:
        monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    else:
        monkeypatch.delenv("LITELLM_GUARD_ENABLED", raising=False)
    app: Final = PlainTextResponse("ok")
    if server == "uvicorn":
        args: Final = ProxyInitializationHelpers.get_default_unvicorn_init_args("127.0.0.1", 4000)
        return uvicorn.Config(**{**args, "app": app, "log_config": None})

    import os

    import gunicorn.config
    import gunicorn.glogging

    from litellm.proxy.shutdown.graceful_uvicorn_worker import build_graceful_shutdown_uvicorn_worker

    gunicorn_config: Final = gunicorn.config.Config()
    worker: Final = build_graceful_shutdown_uvicorn_worker(7.5, None)(
        0, os.getpid(), [], MagicMock(), 30, gunicorn_config, gunicorn.glogging.Logger(gunicorn_config)
    )
    worker.config.app = app
    return worker.config


@pytest.mark.parametrize("server", ["uvicorn", "gunicorn"])
def test_server_client_ip_rewriting_is_off_when_guard_enabled(server, monkeypatch):
    """Uvicorn rewrites scope["client"] from X-Forwarded-For (by
    FORWARDED_ALLOW_IPS) before guard runs, which would defeat guard's own
    trusted_proxies policy. With the guard on, the server must not do it."""
    assert _server_config(server, monkeypatch, guard_on=True).proxy_headers is False


@pytest.mark.parametrize("server", ["uvicorn", "gunicorn"])
def test_server_client_ip_rewriting_is_untouched_when_guard_disabled(server, monkeypatch):
    assert _server_config(server, monkeypatch, guard_on=False).proxy_headers is True


def _ws_scope(peer, forwarded_for=None):
    headers: Final = [(b"x-forwarded-for", forwarded_for.encode())] if forwarded_for else []
    return {"type": "websocket", "path": "/v1/realtime", "headers": headers, "client": (peer, 50000)}


async def _run_websocket(scope):
    """Returns (reached_app, messages_sent_to_client)."""
    reached: Final = []
    sent: Final = []

    async def app(scope, receive, send):
        reached.append(True)

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    await GuardMiddleware(app)(scope, receive, send)
    return bool(reached), sent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("blocked", "allowed", "trusted", "depth", "peer", "forwarded_for", "admitted"),
    [
        ("203.0.113.0/24", "", "", "1", "203.0.113.7", None, False),
        ("203.0.113.7", "", "", "1", "198.51.100.1", None, True),
        ("", "198.51.100.0/24", "", "1", "203.0.113.7", None, False),
        ("", "198.51.100.0/24", "", "1", "198.51.100.9", None, True),
        # Untrusted peer forging an allowed address: X-Forwarded-For ignored.
        ("", "198.51.100.0/24", "", "1", "203.0.113.7", "198.51.100.9", False),
        # Trusted proxy: the forwarded client decides.
        ("203.0.113.7", "", "10.0.0.0/8", "1", "10.1.1.1", "203.0.113.7", False),
        ("", "198.51.100.0/24", "10.0.0.0/8", "1", "10.1.1.1", "198.51.100.9", True),
        # The last entry is the one the nearest proxy saw; entries to its left are claims.
        ("", "198.51.100.0/24", "10.0.0.0/8", "1", "10.1.1.1", "198.51.100.9, 203.0.113.7", False),
        ("", "198.51.100.0/24", "10.0.0.0/8", "2", "10.1.1.1", "203.0.113.7, 198.51.100.9, 10.2.2.2", True),
        ("", "198.51.100.0/24", "10.0.0.0/8", "3", "10.1.1.1", "203.0.113.7, 198.51.100.9, 10.2.2.2", True),
        # Depth 2 selects the second entry from the right.
        ("198.51.100.9", "", "10.0.0.0/8", "2", "10.1.1.1", "203.0.113.7, 198.51.100.9, 10.2.2.2", False),
        ("203.0.113.7", "", "10.0.0.0/8", "2", "10.1.1.1", "203.0.113.7, 198.51.100.9, 10.2.2.2", True),
        # No lists configured: nothing to enforce.
        ("", "", "", "1", "203.0.113.7", None, True),
    ],
)
async def test_websocket_scopes_enforce_ip_lists(
    monkeypatch, blocked, allowed, trusted, depth, peer, forwarded_for, admitted
):
    _install_stub_guard(monkeypatch)
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", blocked)
    monkeypatch.setenv("LITELLM_GUARD_ALLOWED_IPS", allowed)
    monkeypatch.setenv("LITELLM_GUARD_TRUSTED_PROXIES", trusted)
    monkeypatch.setenv("LITELLM_GUARD_TRUSTED_PROXY_DEPTH", depth)
    reached, sent = await _run_websocket(_ws_scope(peer, forwarded_for))
    assert reached is admitted
    if not admitted:
        assert sent == [{"type": "websocket.close", "code": 1008}]


@pytest.mark.asyncio
async def test_websocket_ip_lists_do_not_block_in_passive_mode(monkeypatch):
    _install_stub_guard(monkeypatch)
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", "203.0.113.7")
    monkeypatch.setenv("LITELLM_GUARD_PASSIVE_MODE", "true")
    reached, _ = await _run_websocket(_ws_scope("203.0.113.7"))
    assert reached is True


@pytest.mark.asyncio
async def test_websocket_fails_closed_without_package(monkeypatch):
    monkeypatch.setitem(sys.modules, "guard", None)
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    reached, sent = await _run_websocket(_ws_scope("198.51.100.1"))
    assert reached is False
    assert sent == [{"type": "websocket.close", "code": 1008}]


@pytest.mark.asyncio
async def test_websocket_passes_through_when_guard_disabled(monkeypatch):
    monkeypatch.delenv("LITELLM_GUARD_ENABLED", raising=False)
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", "203.0.113.7")
    reached, _ = await _run_websocket(_ws_scope("203.0.113.7"))
    assert reached is True


def _build_health_app():
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route

    async def ok(request):
        return JSONResponse({"ok": True})

    paths: Final = [
        "/health",
        "/health/liveliness",
        "/health/liveness",
        "/health/readiness",
        "/health/readiness/details",
        "/health/test_connection",
        "/health/services",
        "/health/drain",
        "/metrics",
    ]
    app: Final = Starlette(routes=[Route(path, ok, methods=["GET"]) for path in paths])
    app.add_middleware(GuardMiddleware)
    return app


def test_only_unauthenticated_probes_bypass_guard_by_default(monkeypatch):
    """guard-core matches exclude_paths by subtree (path == entry or under
    entry + "/"), so excluding /health would also exempt the authenticated
    /health/test_connection and /health/readiness/details. Only the exact
    probe paths bypass guard."""
    _install_stub_guard(monkeypatch)
    monkeypatch.delenv("LITELLM_GUARD_EXCLUDED_PATHS", raising=False)
    client: Final = _client(_unique_ip(), _build_health_app())
    probes: Final = ["/health/liveliness", "/health/liveness", "/health/readiness"]
    screened: Final = [
        "/health",
        "/health/readiness/details",
        "/health/test_connection",
        "/health/services",
        "/health/drain",
        "/metrics",
    ]
    for path in probes + screened:
        assert client.get(path).status_code == 200
    assert _StubSecurityMiddleware.seen_paths == screened


def test_operator_excluded_paths_replace_the_default_probes(monkeypatch):
    _install_stub_guard(monkeypatch)
    monkeypatch.setenv("LITELLM_GUARD_EXCLUDED_PATHS", "/metrics")
    client: Final = _client(_unique_ip(), _build_health_app())
    assert client.get("/health/liveliness").status_code == 200
    assert _StubSecurityMiddleware.seen_paths == ["/health/liveliness"]
    assert _StubSecurityMiddleware.instances[-1].config.kwargs["exclude_paths"] == ["/metrics"]


def test_probes_still_enforce_the_ip_lists(monkeypatch):
    """Guard keeps the IP lists on its own excluded paths, so the wrapper's
    probe bypass must too: a blocked address gets 403 on a probe."""
    blocked: Final = _unique_ip()
    _install_stub_guard(monkeypatch)
    monkeypatch.setenv("LITELLM_GUARD_BLOCKED_IPS", blocked)
    client: Final = _client(blocked, _build_health_app())
    assert client.get("/health/readiness").status_code == 403
    assert _client(_unique_ip(), _build_health_app()).get("/health/readiness").status_code == 200
    assert _StubSecurityMiddleware.seen_paths == []


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
def test_probes_are_not_rate_limited_but_other_health_routes_are(monkeypatch):
    """A 429 on a liveness probe gets the pod killed, so probes skip the rate
    limit; the authenticated /health/readiness/details and /metrics do not."""
    monkeypatch.setenv("LITELLM_GUARD_ENABLED", "true")
    monkeypatch.setenv("LITELLM_GUARD_RATE_LIMIT", "2")
    monkeypatch.setenv("LITELLM_GUARD_RATE_LIMIT_WINDOW", "60")
    with _client(_unique_ip(), _build_health_app()) as client:
        probe_codes: Final = [client.get("/health/readiness").status_code for _ in range(4)]
        details_codes: Final = [client.get("/health/readiness/details").status_code for _ in range(3)]
        metrics_codes: Final = [client.get("/metrics").status_code for _ in range(3)]
    assert probe_codes == [200, 200, 200, 200]
    assert details_codes == [200, 200, 429]
    assert metrics_codes == [429, 429, 429]


def _ws_scope_lines(peer, *forwarded_for_lines):
    headers: Final = [(b"x-forwarded-for", line.encode()) for line in forwarded_for_lines]
    return {"type": "websocket", "path": "/v1/realtime", "headers": headers, "client": (peer, 50000)}


def _allow_198_51_100_through_10_8(monkeypatch):
    _install_stub_guard(monkeypatch)
    monkeypatch.setenv("LITELLM_GUARD_ALLOWED_IPS", "198.51.100.0/24")
    monkeypatch.setenv("LITELLM_GUARD_TRUSTED_PROXIES", "10.0.0.0/8")
    monkeypatch.setenv("LITELLM_GUARD_TRUSTED_PROXY_DEPTH", "1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("lines", "admitted"),
    [
        pytest.param(("198.51.100.9", "203.0.113.7"), False, id="proxy-line-names-a-blocked-client"),
        pytest.param(("203.0.113.7", "198.51.100.9"), True, id="proxy-line-names-an-allowed-client"),
    ],
)
async def test_websocket_reads_every_forwarded_for_line_in_order(monkeypatch, lines, admitted):
    """A proxy that adds its own X-Forwarded-For line instead of appending
    leaves two lines; fastapi-guard joins them in order and reads the chain
    from the right, so the last line decides."""
    _allow_198_51_100_through_10_8(monkeypatch)
    reached, sent = await _run_websocket(_ws_scope_lines("10.1.1.1", *lines))
    assert reached is admitted
    if not admitted:
        assert sent == [{"type": "websocket.close", "code": 1008}]


@pytest.mark.parametrize(
    ("lines", "status"),
    [
        pytest.param(("198.51.100.9", "203.0.113.7"), 403, id="proxy-line-names-a-blocked-client"),
        pytest.param(("203.0.113.7", "198.51.100.9"), 200, id="proxy-line-names-an-allowed-client"),
    ],
)
def test_probe_reads_every_forwarded_for_line_in_order(monkeypatch, lines, status):
    _allow_198_51_100_through_10_8(monkeypatch)
    client: Final = _client("10.1.1.1", _build_health_app())
    response: Final = client.get("/health/readiness", headers=[("x-forwarded-for", line) for line in lines])
    assert response.status_code == status


def test_metacharacter_entry_is_not_a_client_address():
    """guard-core reads the selected entry as an address first
    (ip_extraction.py _extract_from_forwarded_header, :76-80, and
    _resolve_forwarded_chain_right_to_left, :125-132) and refuses one that holds
    a metacharacter, so a wildcard never becomes a client: it resolves to the
    connecting peer."""
    scope: Final = _ws_scope_lines("10.1.1.1", "198.51.100.9", "198.51.100.*")
    resolved: Final = _resolve_client(scope, _networks(("10.0.0.0/8",)), 1)
    assert str(resolved) == "10.1.1.1"


_PARITY_CASES: Final = [
    pytest.param("10.1.1.1", ("198.51.100.9",), 1, id="one-line"),
    pytest.param("10.1.1.1", ("203.0.113.7", "198.51.100.9"), 1, id="two-lines-depth-1"),
    pytest.param("10.1.1.1", ("203.0.113.7", "198.51.100.9"), 2, id="two-lines-depth-2"),
    pytest.param("10.1.1.1", ("203.0.113.7", "198.51.100.9", "10.2.2.2"), 1, id="three-lines-trusted-last"),
    pytest.param("10.1.1.1", ("203.0.113.7", "198.51.100.9", "10.2.2.2"), 2, id="three-lines-depth-2"),
    pytest.param("10.1.1.1", ("203.0.113.7, 198.51.100.9", "10.2.2.2"), 3, id="comma-inside-a-line"),
    pytest.param("10.1.1.1", ("198.51.100.9",), 3, id="chain-shorter-than-depth"),
    pytest.param("10.1.1.1", ("::ffff:198.51.100.9",), 1, id="ipv4-mapped-entry"),
    pytest.param("10.1.1.1", ("2001:db8::1",), 1, id="ipv6-entry"),
    pytest.param("10.1.1.1", ("[2001:db8::1]:443",), 1, id="bracketed-ipv6-with-port"),
    pytest.param("10.1.1.1", ("[2001:db8::1]",), 1, id="bracketed-ipv6"),
    pytest.param("10.1.1.1", ("198.51.100.9:8080",), 1, id="ipv4-with-port"),
    pytest.param("10.1.1.1", ("198.51.100.9:abc",), 1, id="ipv4-with-bad-port"),
    pytest.param("10.1.1.1", ("[2001:db8::1]junk",), 1, id="bracketed-ipv6-with-junk"),
    pytest.param("10.1.1.1", ("not-an-ip",), 1, id="malformed-entry"),
    pytest.param("10.1.1.1", ("198.51.100.9", "not-an-ip"), 1, id="malformed-last-entry"),
    pytest.param("10.1.1.1", ("not-an-ip", "198.51.100.9"), 2, id="malformed-entry-left-of-depth"),
    pytest.param("10.1.1.1", ("198.51.100.9", "not-an-ip"), 2, id="malformed-entry-selected"),
    pytest.param("10.1.1.1", ("198.51.100.*",), 1, id="wildcard-entry"),
    pytest.param("10.1.1.1", ("198.51.100.9", "198.51.100.*", "10.2.2.2"), 2, id="wildcard-selected-behind-proxy"),
    pytest.param("10.1.1.1", ("198.51.100.9\\",), 1, id="backslash-entry"),
    pytest.param("10.1.1.1", ("198.51.100.9?",), 1, id="question-mark-entry"),
    pytest.param("10.1.1.1", ("203.0.113.7", "198.51.100.9"), 3, id="depth-over-chain"),
    pytest.param("10.1.1.1", ("::ffff:10.2.2.2", "198.51.100.9"), 2, id="mapped-trusted-entry"),
    pytest.param("::ffff:10.1.1.1", ("198.51.100.9",), 1, id="ipv4-mapped-peer"),
    pytest.param("203.0.113.7", ("198.51.100.9",), 1, id="untrusted-peer"),
    pytest.param("10.1.1.1", (), 1, id="no-header"),
]


@pytest.mark.skipif(not _GUARD_AVAILABLE, reason="fastapi-guard not installed")
@pytest.mark.asyncio
@pytest.mark.parametrize(("peer", "lines", "depth"), _PARITY_CASES)
async def test_resolve_client_matches_guard_core(peer, lines, depth):
    """The wrapper resolves the client exactly as guard-core does for the same
    peer, header lines (joined as fastapi-guard joins them) and config."""
    from types import SimpleNamespace

    from guard.adapters import _join_repeated_header_lines
    from guard_core._utils.ip_extraction import extract_client_ip
    from starlette.datastructures import Headers

    trusted: Final = ["10.0.0.0/8"]
    raw: Final = [(b"x-forwarded-for", line.encode("latin-1")) for line in lines]
    request: Final = SimpleNamespace(
        client_host=peer, headers=_join_repeated_header_lines(Headers(raw=raw)), state=SimpleNamespace()
    )
    config: Final = SimpleNamespace(trusted_proxies=trusted, trusted_proxy_depth=depth)
    expected: Final = await extract_client_ip(request, config)
    scope: Final = {"type": "http", "path": "/x", "headers": raw, "client": (peer, 50000)}
    resolved: Final = _resolve_client(scope, _networks(trusted), depth)
    assert str(resolved) == expected
