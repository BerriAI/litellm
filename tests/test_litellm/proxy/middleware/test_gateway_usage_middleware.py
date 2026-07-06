"""
Tests for GatewayUsageMiddleware.

Verifies route classification (using RouteChecks.is_llm_api_route),
status code bucketing (including 401/403/429/502/0), streaming responses,
and conditional activation.
"""

from unittest.mock import AsyncMock

import pytest

from litellm.proxy.middleware.gateway_usage_middleware import GatewayUsageMiddleware
from litellm.proxy.usage_reporting.gateway_usage_reporter import _counters


@pytest.fixture(autouse=True)
def _reset_counters():
    _counters.total_requests = 0
    _counters.successful_requests = 0
    yield
    _counters.total_requests = 0
    _counters.successful_requests = 0


def _make_scope(path: str, scope_type: str = "http") -> dict:
    return {"type": scope_type, "path": path}


async def _make_app(status: int):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status})
        await send({"type": "http.response.body", "body": b"ok"})

    return app


async def _make_streaming_app(status: int, chunks: int = 3):
    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": status})
        for i in range(chunks):
            await send({"type": "http.response.body", "body": f"chunk{i}".encode(), "more_body": i < chunks - 1})

    return app


@pytest.mark.asyncio
async def test_counts_chat_completions():
    app = await _make_app(200)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    scope = _make_scope("/v1/chat/completions")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 1
    assert _counters.successful_requests == 1


@pytest.mark.asyncio
async def test_counts_5xx_as_failed():
    app = await _make_app(500)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    scope = _make_scope("/v1/chat/completions")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 1
    assert _counters.successful_requests == 0


@pytest.mark.asyncio
async def test_counts_4xx_as_successful():
    for status in (400, 401, 403, 404, 422, 429):
        _counters.total_requests = 0
        _counters.successful_requests = 0

        app = await _make_app(status)
        middleware = GatewayUsageMiddleware(app)
        middleware._enabled = True

        scope = _make_scope("/v1/chat/completions")
        await middleware(scope, AsyncMock(), AsyncMock())

        assert _counters.total_requests == 1, f"status={status}"
        assert _counters.successful_requests == 1, f"status={status}"


@pytest.mark.asyncio
async def test_counts_502_as_failed():
    app = await _make_app(502)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    scope = _make_scope("/v1/chat/completions")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 1
    assert _counters.successful_requests == 0


@pytest.mark.asyncio
async def test_status_code_zero_counted_as_failed():
    async def app_no_response(scope, receive, send):
        pass

    middleware = GatewayUsageMiddleware(app_no_response)
    middleware._enabled = True

    scope = _make_scope("/v1/chat/completions")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 1
    assert _counters.successful_requests == 0


@pytest.mark.asyncio
async def test_ignores_non_llm_routes():
    app = await _make_app(200)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    scope = _make_scope("/health")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 0
    assert _counters.successful_requests == 0


@pytest.mark.asyncio
async def test_ignores_non_http_scope():
    app = await _make_app(200)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    scope = _make_scope("/v1/chat/completions", scope_type="websocket")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 0


@pytest.mark.asyncio
async def test_multiple_requests_accumulate():
    app = await _make_app(200)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    for _ in range(5):
        scope = _make_scope("/v1/chat/completions")
        await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 5
    assert _counters.successful_requests == 5


@pytest.mark.asyncio
async def test_streaming_response_counted_once():
    app = await _make_streaming_app(200, chunks=5)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    scope = _make_scope("/v1/chat/completions")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 1
    assert _counters.successful_requests == 1


@pytest.mark.asyncio
async def test_middleware_inactive_without_env_var():
    app = await _make_app(200)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = False

    scope = _make_scope("/v1/chat/completions")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 0


@pytest.mark.asyncio
async def test_anthropic_messages_route_counted():
    app = await _make_app(200)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    scope = _make_scope("/v1/messages")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 1
    assert _counters.successful_requests == 1


@pytest.mark.asyncio
async def test_embeddings_route_counted():
    app = await _make_app(200)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    scope = _make_scope("/v1/embeddings")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 1


@pytest.mark.asyncio
async def test_responses_route_counted():
    app = await _make_app(200)
    middleware = GatewayUsageMiddleware(app)
    middleware._enabled = True

    scope = _make_scope("/v1/responses")
    await middleware(scope, AsyncMock(), AsyncMock())

    assert _counters.total_requests == 1
