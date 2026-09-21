"""
Tests for BudgetReservationReleaseMiddleware.

Auth reserves budget before the handler runs and hands the reservation to the
request state. A handler whose litellm call builds a logging object binds it to
the cost callbacks; anything still unbound when the response is done would keep
the spend counter pinned until its TTL, so the middleware releases it.
"""

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from typing import Final

import pytest
from starlette.applications import Starlette
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from litellm.caching import DualCache
from litellm.litellm_core_utils.litellm_logging import Logging
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

KEY_TOKEN: Final = "hashed-release-middleware-key"
COUNTER_KEY: Final = f"spend:key:{KEY_TOKEN}"
CHAT_BODY: Final = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}


@pytest.fixture
def spend_counter_cache(monkeypatch: pytest.MonkeyPatch) -> DualCache:
    cache: Final = DualCache()
    monkeypatch.setattr(proxy_server, "spend_counter_cache", cache)
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    return cache


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


def _bind_to_a_logging_object(reservation: dict) -> Logging:
    logging_obj: Final = Logging(
        model="gpt-4o",
        messages=CHAT_BODY["messages"],
        stream=True,
        call_type="completion",
        start_time=datetime.now(),
        litellm_call_id="release-middleware-call",
        function_id="release-middleware-fn",
    )
    logging_obj.update_environment_variables(
        litellm_params={"metadata": {"user_api_key_budget_reservation": reservation}},
        optional_params={},
    )
    return logging_obj


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

    async def receive() -> Message:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: Message) -> None:
        return None

    await app(scope, receive, send)


def _counter(spend_counter_cache: DualCache) -> float | None:
    return spend_counter_cache.in_memory_cache.get_cache(key=COUNTER_KEY)


def test_is_not_base_http_middleware():
    assert not issubclass(BudgetReservationReleaseMiddleware, BaseHTTPMiddleware)


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
async def test_bound_reservation_is_left_for_the_callback_that_finishes_after_the_response(
    spend_counter_cache: DualCache,
):
    reservation: Final = await _reserve()
    reserved_cost: Final = reservation["reserved_cost"]

    async def handler(request: Request) -> Response:
        request.state.budget_reservation = reservation
        _bind_to_a_logging_object(reservation)
        return JSONResponse({"choices": []})

    await _post(_app(handler))

    assert _counter(spend_counter_cache) == pytest.approx(reserved_cost)
    assert reservation["finalized"] is False

    actual_cost: Final = reserved_cost / 4
    await reconcile_budget_reservation(budget_reservation=reservation, actual_cost=actual_cost)

    assert _counter(spend_counter_cache) == pytest.approx(actual_cost)
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
async def test_non_http_scopes_pass_through():
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
