import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Final

import httpx
from fastapi import FastAPI

from litellm.litellm_core_utils.ecb_reference_rates import EcbFetchFailed, EcbReferenceRates
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.ui_crud_endpoints.reference_rates_endpoints import (
    EcbFeedFetch,
    _default_fetch,
    _default_rates,
    _default_refresh_lock,
    router,
)

ROUTE: Final = "/get/reference_exchange_rates"
FEED: Final = (
    "<gesmes:Envelope xmlns:gesmes='http://www.gesmes.org/xml/2002-08-01'"
    " xmlns='http://www.ecb.int/vocabulary/2002-08-01/eurofxref'><Cube><Cube time='2026-11-13'>"
    "<Cube currency='USD' rate='1.25'/><Cube currency='GBP' rate='0.5'/><Cube currency='JPY' rate='125'/>"
    "</Cube></Cube></gesmes:Envelope>"
).encode()
RATES_BODY: Final = {
    "source": "ecb",
    "rate_date": "2026-11-13",
    "usd_per_unit": {"USD": 1.0, "EUR": 1.25, "GBP": 2.5, "JPY": 0.01},
}


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class _Feed:
    def __init__(self, *bodies: bytes | EcbFetchFailed) -> None:
        self.bodies = bodies
        self.fetches = 0

    async def __call__(self) -> bytes | EcbFetchFailed:
        self.fetches += 1
        return self.bodies[self.fetches - 1]


async def _caller() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id="viewer")


def _app(rates: EcbReferenceRates, feed: EcbFeedFetch) -> FastAPI:
    app: Final = FastAPI()
    app.include_router(router)
    lock: Final = asyncio.Lock()
    app.dependency_overrides[user_api_key_auth] = _caller
    app.dependency_overrides[_default_rates] = lambda: rates
    app.dependency_overrides[_default_refresh_lock] = lambda: lock
    app.dependency_overrides[_default_fetch] = lambda: feed
    return app


@asynccontextmanager
async def _client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://proxy") as client:
        yield client


def _clock() -> _Clock:
    return _Clock(datetime(2026, 11, 13, 16, tzinfo=timezone.utc))


async def test_first_request_fetches_and_returns_usd_per_unit_for_every_feed_currency() -> None:
    clock: Final = _clock()
    feed: Final = _Feed(FEED)
    async with _client(_app(EcbReferenceRates(clock=clock), feed)) as client:
        first: Final = await client.get(ROUTE)
        again: Final = await client.get(ROUTE)

    assert first.status_code == 200, first.text
    assert first.json() == RATES_BODY
    assert again.json() == RATES_BODY
    assert feed.fetches == 1


async def test_failed_refresh_answers_with_the_fresh_held_rates() -> None:
    clock: Final = _clock()
    feed: Final = _Feed(FEED, EcbFetchFailed("HTTP 503"))
    async with _client(_app(EcbReferenceRates(clock=clock), feed)) as client:
        assert (await client.get(ROUTE)).status_code == 200
        clock.now = datetime(2026, 11, 14, 16, tzinfo=timezone.utc)
        response: Final = await client.get(ROUTE)

    assert response.json() == RATES_BODY
    assert feed.fetches == 2


async def test_never_fetched_rates_answer_503() -> None:
    clock: Final = _clock()
    async with _client(_app(EcbReferenceRates(clock=clock), _Feed(EcbFetchFailed("HTTP 503")))) as client:
        response: Final = await client.get(ROUTE)

    assert response.status_code == 503, response.text


async def test_stale_rates_with_a_failed_refresh_answer_503_instead_of_stale_rates() -> None:
    clock: Final = _clock()
    async with _client(_app(EcbReferenceRates(clock=clock), _Feed(FEED, EcbFetchFailed("HTTP 503")))) as client:
        assert (await client.get(ROUTE)).status_code == 200
        clock.now = datetime(2026, 11, 21, 16, tzinfo=timezone.utc)
        response: Final = await client.get(ROUTE)

    assert response.status_code == 503, response.text
