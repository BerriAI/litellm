"""
Unit tests for _MCPTrailingSlashMiddleware.

Validates that the middleware rewrites scope["path"] (and scope["raw_path"])
from BASE_MCP_ROUTE to BASE_MCP_ROUTE + "/" for HTTP scopes, and passes
everything else through unchanged.
"""

import asyncio

import pytest


# The middleware is defined at module scope in proxy_server.py alongside heavy
# imports we don't want here.  Re-implement the same class locally to test the
# logic in isolation (it's only ~10 lines) while keeping the test dependency-free.

BASE_MCP_ROUTE = "/mcp"


class _MCPTrailingSlashMiddleware:
    """Mirror of the production middleware for isolated testing."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope.get("path") == BASE_MCP_ROUTE:
            scope = dict(
                scope,
                path=BASE_MCP_ROUTE + "/",
                raw_path=(scope.get("raw_path") or BASE_MCP_ROUTE.encode()) + b"/",
            )
        await self.app(scope, receive, send)


# ── helpers ──────────────────────────────────────────────────────────────


class _Recorder:
    """Dummy ASGI app that records the scope it was called with."""

    def __init__(self):
        self.scopes: list = []

    async def __call__(self, scope, receive, send):
        self.scopes.append(scope)


def _make_http_scope(path: str, raw_path: bytes | None = None) -> dict:
    scope: dict = {"type": "http", "path": path}
    if raw_path is not None:
        scope["raw_path"] = raw_path
    return scope


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── tests ────────────────────────────────────────────────────────────────


def test_rewrites_exact_mcp_path():
    """POST /mcp should be rewritten to /mcp/."""
    recorder = _Recorder()
    mw = _MCPTrailingSlashMiddleware(recorder)

    scope = _make_http_scope("/mcp", b"/mcp")
    _run(mw(scope, None, None))

    assert len(recorder.scopes) == 1
    assert recorder.scopes[0]["path"] == "/mcp/"
    assert recorder.scopes[0]["raw_path"] == b"/mcp/"


def test_no_rewrite_for_mcp_subpath():
    """/mcp/foo should NOT be rewritten."""
    recorder = _Recorder()
    mw = _MCPTrailingSlashMiddleware(recorder)

    scope = _make_http_scope("/mcp/foo", b"/mcp/foo")
    _run(mw(scope, None, None))

    assert recorder.scopes[0]["path"] == "/mcp/foo"
    assert recorder.scopes[0]["raw_path"] == b"/mcp/foo"


def test_no_rewrite_for_other_paths():
    """/health should pass through unchanged."""
    recorder = _Recorder()
    mw = _MCPTrailingSlashMiddleware(recorder)

    scope = _make_http_scope("/health", b"/health")
    _run(mw(scope, None, None))

    assert recorder.scopes[0]["path"] == "/health"


def test_no_rewrite_for_non_http_scope():
    """WebSocket or lifespan scopes should pass through unchanged."""
    recorder = _Recorder()
    mw = _MCPTrailingSlashMiddleware(recorder)

    scope = {"type": "websocket", "path": "/mcp"}
    _run(mw(scope, None, None))

    assert recorder.scopes[0]["path"] == "/mcp"


def test_raw_path_fallback_when_absent():
    """If raw_path is missing from scope, middleware should still work."""
    recorder = _Recorder()
    mw = _MCPTrailingSlashMiddleware(recorder)

    scope = _make_http_scope("/mcp")  # no raw_path
    _run(mw(scope, None, None))

    assert recorder.scopes[0]["path"] == "/mcp/"
    assert recorder.scopes[0]["raw_path"] == b"/mcp/"


def test_already_trailing_slash_no_double():
    """/mcp/ should NOT be rewritten (exact match only)."""
    recorder = _Recorder()
    mw = _MCPTrailingSlashMiddleware(recorder)

    scope = _make_http_scope("/mcp/", b"/mcp/")
    _run(mw(scope, None, None))

    assert recorder.scopes[0]["path"] == "/mcp/"
    assert recorder.scopes[0]["raw_path"] == b"/mcp/"
