"""
Tests for BudgetReservationReleaseMiddleware.

Auth reserves budget before the handler runs and hands the reservation to the
request or socket state. A litellm call made through the async client wrapper
claims it for the cost callback that runs after the call; anything still unclaimed
when the response is done or the socket has closed would keep the spend counter
pinned until its TTL, so the middleware releases it.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from datetime import datetime
from typing import Final

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send
from starlette.websockets import WebSocket

import litellm
from litellm.caching import DualCache
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.middleware.budget_reservation_release_middleware import (
    BudgetReservationReleaseMiddleware,
)
from litellm.proxy.spend_tracking.budget_reservation import (
    reconcile_budget_reservation,
    release_unbound_budget_reservation,
    reserve_budget_for_request,
)
from litellm.proxy.utils import ProxyLogging
from litellm.utils import Rules, function_setup

KEY_TOKEN: Final = "hashed-release-middleware-key"
COUNTER_KEY: Final = f"spend:key:{KEY_TOKEN}"
CHAT_BODY: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture
def spend_counter_cache(monkeypatch: pytest.MonkeyPatch) -> DualCache:
    cache: Final = DualCache()
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    return cache


@pytest.fixture
def no_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    for callback_list_name in (
        "callbacks",
        "success_callback",
        "failure_callback",
        "_async_success_callback",
        "_async_failure_callback",
    ):
        monkeypatch.setattr(litellm, callback_list_name, [])


async def _reserve() -> dict:
    reservation: Final = await reserve_budget_for_request(
        request_body=CHAT_BODY,
        route="/v1/chat/completions",
        llm_router=None,
        valid_token=UserAPIKeyAuth(token=KEY_TOKEN, max_budget=1.0, spend=0.0),
        team_object=None,
        user_object=None,
        prisma_client=None,
        user_api_key_cache=UserApiKeyCache(),
        proxy_logging_obj=ProxyLogging(user_api_key_cache=UserApiKeyCache()),
    )
    assert reservation is not None
    assert reservation["reserved_cost"] > 0
    return reservation


async def _chat(reservation: dict, **kwargs: object) -> object:
    return await litellm.acompletion(
        **CHAT_BODY,
        metadata={"user_api_key_budget_reservation": reservation},
        **kwargs,
    )


def _proxy_pre_call_setup(route_type: str, reservation: dict) -> None:
    function_setup(
        original_function=route_type,
        rules_obj=Rules(),
        start_time=datetime.now(),
        **CHAT_BODY,
        litellm_call_id="proxy-pre-call-setup",
        metadata={"user_api_key_budget_reservation": reservation},
    )


def _app(
    handler: Callable[[Request], Awaitable[Response]],
    release: Callable[[Mapping[str, object]], Awaitable[None]] = release_unbound_budget_reservation,
) -> Starlette:
    app: Final = Starlette(routes=[Route("/", handler, methods=["POST"])])
    app.add_middleware(BudgetReservationReleaseMiddleware, release=release)
    return app


async def _post(app: ASGIApp) -> None:
    scope: Final = {
        "type": "http",
        "method": "POST",
        "path": "/",
        "raw_path": b"/",
        "headers": [],
        "query_string": b"",
        "scheme": "http",
        "server": ("testserver", 80),
        "client": ("testclient", 1),
    }

    body_delivered: Final = asyncio.Event()
    client_never_disconnects: Final = asyncio.Event()

    async def receive() -> Message:
        if body_delivered.is_set():
            await client_never_disconnects.wait()
        body_delivered.set()
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        return None

    await app(scope, receive, send)


def _counter(spend_counter_cache: DualCache) -> float | None:
    return spend_counter_cache.in_memory_cache.get_cache(key=COUNTER_KEY)


@pytest.mark.asyncio
async def test_unbound_reservation_is_released_after_the_response(spend_counter_cache: DualCache):
    reservation: Final = await _reserve()
    assert _counter(spend_counter_cache) == pytest.approx(reservation["reserved_cost"])

    async def handler(request: Request) -> Response:
        request.state.budget_reservation = reservation
        return JSONResponse({"id": "batch_123", "status": "cancelling"})

    await _post(_app(handler))

    assert _counter(spend_counter_cache) == pytest.approx(0.0)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_unbound_reservation_is_released_when_the_handler_raises(spend_counter_cache: DualCache):
    reservation: Final = await _reserve()

    async def handler(request: Request) -> Response:
        request.state.budget_reservation = reservation
        raise RuntimeError("upstream refused the cancel")

    with pytest.raises(RuntimeError):
        await _post(_app(handler))

    assert _counter(spend_counter_cache) == pytest.approx(0.0)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_reservation_seen_only_by_the_proxy_pre_call_logging_object_is_released(
    spend_counter_cache: DualCache, no_callbacks: None
):
    reservation: Final = await _reserve()

    async def cancel_batch_without_a_client_wrapper() -> dict:
        return {"id": "batch_123", "status": "cancelling"}

    async def handler(request: Request) -> Response:
        request.state.budget_reservation = reservation
        _proxy_pre_call_setup("acancel_batch", reservation)
        return JSONResponse(await cancel_batch_without_a_client_wrapper())

    await _post(_app(handler))

    assert _counter(spend_counter_cache) == pytest.approx(0.0)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_reservation_of_a_failed_call_is_released_after_the_error_response(
    spend_counter_cache: DualCache, no_callbacks: None
):
    reservation: Final = await _reserve()
    refused: Final = litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-4o")

    async def handler(request: Request) -> Response:
        request.state.budget_reservation = reservation
        _proxy_pre_call_setup("acompletion", reservation)
        try:
            await _chat(reservation, mock_response=refused)
        except litellm.AuthenticationError:
            return JSONResponse({"error": {"message": "bad key"}}, status_code=401)
        raise AssertionError("the mocked call must fail")

    await _post(_app(handler))

    assert _counter(spend_counter_cache) == pytest.approx(0.0)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_reservation_claimed_by_a_completed_call_is_left_for_the_callback(
    spend_counter_cache: DualCache, no_callbacks: None
):
    reservation: Final = await _reserve()
    reserved_cost: Final = reservation["reserved_cost"]

    async def handler(request: Request) -> Response:
        request.state.budget_reservation = reservation
        _proxy_pre_call_setup("acompletion", reservation)
        response: Final = await _chat(reservation, mock_response="ok")
        return JSONResponse(response.model_dump())

    await _post(_app(handler))

    assert _counter(spend_counter_cache) == pytest.approx(reserved_cost)
    assert reservation["finalized"] is False


@pytest.mark.asyncio
async def test_reservation_claimed_by_a_streaming_call_is_left_for_the_callback_that_finishes_after_the_response(
    spend_counter_cache: DualCache, no_callbacks: None
):
    reservation: Final = await _reserve()
    reserved_cost: Final = reservation["reserved_cost"]

    async def handler(request: Request) -> Response:
        request.state.budget_reservation = reservation
        _proxy_pre_call_setup("acompletion", reservation)
        stream: Final = await _chat(reservation, mock_response="ok", stream=True)

        async def sse() -> AsyncIterator[bytes]:
            async for chunk in stream:
                yield f"data: {chunk.model_dump_json()}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")

    await _post(_app(handler))

    assert _counter(spend_counter_cache) == pytest.approx(reserved_cost)
    assert reservation["finalized"] is False

    actual_cost: Final = reserved_cost / 4
    await reconcile_budget_reservation(budget_reservation=reservation, actual_cost=actual_cost)

    assert _counter(spend_counter_cache) == pytest.approx(actual_cost)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_unbound_reservation_of_a_websocket_session_is_released_when_the_socket_closes(
    spend_counter_cache: DualCache,
):
    reservation: Final = await _reserve()

    async def listen_without_a_provider_key(scope: Scope, receive: Receive, send: Send) -> None:
        websocket: Final = WebSocket(scope, receive, send)
        websocket.state.budget_reservation = reservation
        await websocket.close(code=1011, reason="Required 'DEEPGRAM_API_KEY' in environment")

    async def receive() -> Message:
        return {"type": "websocket.connect"}

    async def send(message: Message) -> None:
        return None

    middleware: Final = BudgetReservationReleaseMiddleware(
        listen_without_a_provider_key, release=release_unbound_budget_reservation
    )
    await middleware({"type": "websocket", "path": "/deepgram/v1/listen", "headers": []}, receive, send)

    assert _counter(spend_counter_cache) == pytest.approx(0.0)
    assert reservation["finalized"] is True


@pytest.mark.asyncio
async def test_release_runs_once_per_request_with_the_stamped_reservation():
    released: Final = []
    reservation: Final = {"reserved_cost": 0.5, "entries": [], "finalized": False, "callback_bound": False}

    async def release(budget_reservation: Mapping[str, object]) -> None:
        released.append(budget_reservation)

    async def handler(request: Request) -> Response:
        request.state.budget_reservation = reservation
        return JSONResponse({})

    await _post(_app(handler, release=release))

    assert released == [reservation]
    assert released[0] is reservation


@pytest.mark.asyncio
async def test_request_without_a_reservation_releases_nothing():
    released: Final = []

    async def release(budget_reservation: Mapping[str, object]) -> None:
        released.append(budget_reservation)

    async def unauthenticated(request: Request) -> Response:
        return JSONResponse({})

    async def budget_checks_skipped(request: Request) -> Response:
        request.state.budget_reservation = None
        return JSONResponse({})

    await _post(_app(unauthenticated, release=release))
    await _post(_app(budget_checks_skipped, release=release))

    assert released == []


@pytest.mark.asyncio
async def test_lifespan_scopes_pass_through():
    released: Final = []
    seen: Final = []

    async def release(budget_reservation: Mapping[str, object]) -> None:
        released.append(budget_reservation)

    async def inner(scope: Scope, receive: Receive, send: Send) -> None:
        seen.append(scope["type"])

    async def receive() -> Message:
        return {"type": "lifespan.startup"}

    async def send(message: Message) -> None:
        return None

    middleware: Final = BudgetReservationReleaseMiddleware(inner, release=release)
    await middleware({"type": "lifespan", "state": {"budget_reservation": {"reserved_cost": 1.0}}}, receive, send)

    assert seen == ["lifespan"]
    assert released == []
