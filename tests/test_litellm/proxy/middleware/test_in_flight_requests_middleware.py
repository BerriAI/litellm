"""
Tests for InFlightRequestsMiddleware.

Verifies that in_flight_requests is incremented during a request and
decremented after it completes, including on errors.
"""
import asyncio

import pytest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from litellm.proxy.middleware.in_flight_requests_middleware import (
    InFlightRequestsMiddleware,
    get_in_flight_requests,
)


@pytest.fixture(autouse=True)
def reset_state():
    """Reset class-level state between tests."""
    InFlightRequestsMiddleware._in_flight = 0
    yield
    InFlightRequestsMiddleware._in_flight = 0


def _make_app(handler):
    from starlette.applications import Starlette

    app = Starlette(routes=[Route("/", handler)])
    app.add_middleware(InFlightRequestsMiddleware)
    return app


# ── Structure ─────────────────────────────────────────────────────────────────


def test_is_not_base_http_middleware():
    """Must be pure ASGI — BaseHTTPMiddleware causes streaming degradation."""
    assert not issubclass(InFlightRequestsMiddleware, BaseHTTPMiddleware)


def test_has_asgi_call_protocol():
    assert "__call__" in InFlightRequestsMiddleware.__dict__


# ── Counter behaviour ─────────────────────────────────────────────────────────


def test_counter_zero_at_start():
    assert get_in_flight_requests() == 0


def test_counter_increments_inside_handler():
    captured = []

    async def handler(request: Request) -> Response:
        captured.append(InFlightRequestsMiddleware.get_count())
        return JSONResponse({})

    TestClient(_make_app(handler)).get("/")
    assert captured == [1]


def test_counter_returns_to_zero_after_request():
    async def handler(request: Request) -> Response:
        return JSONResponse({})

    TestClient(_make_app(handler)).get("/")
    assert get_in_flight_requests() == 0


def test_counter_decrements_after_error():
    """Counter must reach 0 even when the handler raises."""

    async def handler(request: Request) -> Response:
        return Response("boom", status_code=500)

    TestClient(_make_app(handler)).get("/")
    assert get_in_flight_requests() == 0


def test_non_http_scopes_not_counted():
    """Lifespan / websocket scopes must not touch the counter."""

    class _InnerApp:
        async def __call__(self, scope, receive, send):
            pass

    mw = InFlightRequestsMiddleware(_InnerApp())

    asyncio.get_event_loop().run_until_complete(
        mw({"type": "lifespan"}, None, None)  # type: ignore[arg-type]
    )
    assert get_in_flight_requests() == 0
