"""
Tests for GatewayUsageMiddleware.

Verifies that LLM-route requests are counted (total + successful) while
non-LLM routes are ignored.
"""

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.testclient import TestClient

from litellm.proxy.middleware.gateway_usage_middleware import GatewayUsageMiddleware
from litellm.proxy.usage_reporting.gateway_usage_reporter import _counters


@pytest.fixture(autouse=True)
def reset_counters():
    _counters.total_requests = 0
    _counters.successful_requests = 0
    yield
    _counters.total_requests = 0
    _counters.successful_requests = 0


def _make_app(routes: list[Route]) -> Starlette:
    app = Starlette(routes=routes)
    app.add_middleware(GatewayUsageMiddleware)
    return app


async def _ok_handler(request: Request) -> Response:
    return JSONResponse({"status": "ok"})


async def _error_handler(request: Request) -> Response:
    return Response("internal error", status_code=500)


async def _bad_request_handler(request: Request) -> Response:
    return Response("bad request", status_code=400)


def test_counts_chat_completions_request():
    app = _make_app([Route("/v1/chat/completions", _ok_handler, methods=["POST"])])
    TestClient(app).post("/v1/chat/completions")
    assert _counters.total_requests == 1
    assert _counters.successful_requests == 1


def test_counts_failed_request_as_total_but_not_successful():
    app = _make_app([Route("/v1/chat/completions", _error_handler, methods=["POST"])])
    TestClient(app).post("/v1/chat/completions")
    assert _counters.total_requests == 1
    assert _counters.successful_requests == 0


def test_4xx_counts_as_successful():
    app = _make_app([Route("/v1/chat/completions", _bad_request_handler, methods=["POST"])])
    TestClient(app).post("/v1/chat/completions")
    assert _counters.total_requests == 1
    assert _counters.successful_requests == 1


def test_ignores_non_llm_routes():
    app = _make_app([Route("/health", _ok_handler)])
    TestClient(app).get("/health")
    assert _counters.total_requests == 0
    assert _counters.successful_requests == 0


def test_ignores_management_routes():
    app = _make_app([Route("/key/generate", _ok_handler, methods=["POST"])])
    TestClient(app).post("/key/generate")
    assert _counters.total_requests == 0


def test_counts_embeddings_route():
    app = _make_app([Route("/v1/embeddings", _ok_handler, methods=["POST"])])
    TestClient(app).post("/v1/embeddings")
    assert _counters.total_requests == 1
    assert _counters.successful_requests == 1


def test_counts_responses_route():
    app = _make_app([Route("/v1/responses", _ok_handler, methods=["POST"])])
    TestClient(app).post("/v1/responses")
    assert _counters.total_requests == 1
    assert _counters.successful_requests == 1


def test_counts_completions_route():
    app = _make_app([Route("/completions", _ok_handler, methods=["POST"])])
    TestClient(app).post("/completions")
    assert _counters.total_requests == 1
    assert _counters.successful_requests == 1


def test_multiple_requests_accumulate():
    app = _make_app(
        [
            Route("/v1/chat/completions", _ok_handler, methods=["POST"]),
            Route("/v1/embeddings", _error_handler, methods=["POST"]),
        ]
    )
    client = TestClient(app)
    client.post("/v1/chat/completions")
    client.post("/v1/chat/completions")
    client.post("/v1/embeddings")
    assert _counters.total_requests == 3
    assert _counters.successful_requests == 2
