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

    asyncio.run(mw({"type": "lifespan"}, None, None))  # type: ignore[arg-type]
    assert get_in_flight_requests() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status, expected_calls", [(429, 1), (503, 1), (200, 0), (400, 0), (500, 0)])
async def test_only_shed_responses_are_counted(status, expected_calls):
    """A 500 is the proxy failing, not declining. Counting it here would blur the
    signal an operator uses to decide between throttling and scaling out."""
    from unittest.mock import MagicMock, patch

    from litellm.proxy.middleware.in_flight_requests_middleware import (
        InFlightRequestsMiddleware,
    )

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status, "headers": []})
        await send({"type": "http.response.body", "body": b""})

    logger = MagicMock()
    with patch("litellm.integrations.prometheus.PrometheusLogger.get_instance", return_value=logger):
        await InFlightRequestsMiddleware(app)({"type": "http"}, receive, send)

    assert logger.record_request_shed.call_count == expected_calls
    if expected_calls:
        logger.record_request_shed.assert_called_once_with(status)
    assert [m["type"] for m in sent] == ["http.response.start", "http.response.body"], (
        "the wrapped send must still forward every message downstream"
    )


@pytest.mark.asyncio
async def test_a_broken_metric_never_breaks_the_response():
    from unittest.mock import patch

    from litellm.proxy.middleware.in_flight_requests_middleware import (
        InFlightRequestsMiddleware,
    )

    sent = []

    async def send(message):
        sent.append(message)

    async def receive():
        return {"type": "http.request"}

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 429, "headers": []})

    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.get_instance",
        side_effect=RuntimeError("metrics down"),
    ):
        await InFlightRequestsMiddleware(app)({"type": "http"}, receive, send)

    assert len(sent) == 1
