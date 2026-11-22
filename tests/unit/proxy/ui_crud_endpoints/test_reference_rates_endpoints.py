import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Final, Literal

import httpx
from fastapi import FastAPI

from litellm.litellm_core_utils.ecb_reference_rates import EcbFetchFailed, EcbReferenceRates
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.ui_crud_endpoints.reference_rates_endpoints import (
    EcbFeedFetch,
    _default_fetch,
    _default_rates,
    _default_refresh_cooldown,
    _default_refresh_lock,
    _FailedRefreshCooldown,
    router,
)

ROUTE: Final = "/get/reference_exchange_rates"
DEADLOCK_GUARD_SECONDS: Final = 30.0
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


class _GatedFeed:
    def __init__(self, body: bytes | EcbFetchFailed = FEED) -> None:
        self.body = body
        self.fetches = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self) -> bytes | EcbFetchFailed:
        self.fetches += 1
        self.entered.set()
        await self.release.wait()
        return self.body


class _WaitSignallingLock(asyncio.Lock):
    def __init__(self) -> None:
        super().__init__()
        self.contended = asyncio.Event()

    async def acquire(self) -> Literal[True]:
        if self.locked():
            self.contended.set()
        return await super().acquire()


class _Seconds:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def _caller() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(api_key="sk-test", user_id="viewer")


def _app(
    rates: EcbReferenceRates,
    feed: EcbFeedFetch,
    refresh_lock: asyncio.Lock | None = None,
    cooldown: _FailedRefreshCooldown | None = None,
) -> FastAPI:
    app: Final = FastAPI()
    app.include_router(router)
    lock: Final = refresh_lock or asyncio.Lock()
    refresh_cooldown: Final = cooldown or _FailedRefreshCooldown()
    app.dependency_overrides[_default_refresh_cooldown] = lambda: refresh_cooldown
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


async def test_concurrent_first_requests_share_one_fetch() -> None:
    feed: Final = _GatedFeed()
    lock: Final = _WaitSignallingLock()
    async with _client(_app(EcbReferenceRates(clock=_clock()), feed, lock)) as client:
        first: Final = asyncio.create_task(client.get(ROUTE))
        second: Final = asyncio.create_task(client.get(ROUTE))
        try:
            await asyncio.wait_for(feed.entered.wait(), DEADLOCK_GUARD_SECONDS)
            await asyncio.wait_for(lock.contended.wait(), DEADLOCK_GUARD_SECONDS)
            feed.release.set()
            responses: Final = await asyncio.wait_for(asyncio.gather(first, second), DEADLOCK_GUARD_SECONDS)
        finally:
            feed.release.set()
            first.cancel()
            second.cancel()
            await asyncio.gather(first, second, return_exceptions=True)

    assert [response.json() for response in responses] == [RATES_BODY, RATES_BODY]
    assert feed.fetches == 1


async def test_a_failed_cold_refresh_is_not_retried_until_its_cooldown_ends() -> None:
    seconds: Final = _Seconds()
    feed: Final = _Feed(EcbFetchFailed("HTTP 503"), FEED)
    async with _client(
        _app(EcbReferenceRates(clock=_clock()), feed, cooldown=_FailedRefreshCooldown(seconds))
    ) as client:
        during_outage: Final = [(await client.get(ROUTE)).status_code for _ in range(3)]
        seconds.now = 59.9
        before_cooldown_ends: Final = (await client.get(ROUTE)).status_code
        fetches_before: Final = feed.fetches
        seconds.now = 60.0
        recovered: Final = await client.get(ROUTE)

    assert during_outage == [503, 503, 503]
    assert before_cooldown_ends == 503
    assert fetches_before == 1
    assert recovered.json() == RATES_BODY
    assert feed.fetches == 2


async def test_a_failed_refresh_keeps_serving_held_rates_without_refetching_until_its_cooldown_ends() -> None:
    clock: Final = _clock()
    seconds: Final = _Seconds()
    feed: Final = _Feed(FEED, EcbFetchFailed("HTTP 503"), FEED)
    async with _client(_app(EcbReferenceRates(clock=clock), feed, cooldown=_FailedRefreshCooldown(seconds))) as client:
        assert (await client.get(ROUTE)).status_code == 200
        clock.now = datetime(2026, 11, 14, 16, tzinfo=timezone.utc)
        during_outage: Final = [(await client.get(ROUTE)).json() for _ in range(3)]
        fetches_during_outage: Final = feed.fetches
        seconds.now = 60.0
        recovered: Final = await client.get(ROUTE)

    assert during_outage == [RATES_BODY] * 3
    assert fetches_during_outage == 2
    assert recovered.json() == RATES_BODY
    assert feed.fetches == 3


async def test_concurrent_cold_requests_share_one_failed_fetch() -> None:
    feed: Final = _GatedFeed(EcbFetchFailed("HTTP 503"))
    lock: Final = _WaitSignallingLock()
    cooldown: Final = _FailedRefreshCooldown(_Seconds())
    async with _client(_app(EcbReferenceRates(clock=_clock()), feed, lock, cooldown)) as client:
        first: Final = asyncio.create_task(client.get(ROUTE))
        second: Final = asyncio.create_task(client.get(ROUTE))
        try:
            await asyncio.wait_for(feed.entered.wait(), DEADLOCK_GUARD_SECONDS)
            await asyncio.wait_for(lock.contended.wait(), DEADLOCK_GUARD_SECONDS)
            feed.release.set()
            responses: Final = await asyncio.wait_for(asyncio.gather(first, second), DEADLOCK_GUARD_SECONDS)
        finally:
            feed.release.set()
            first.cancel()
            second.cancel()
            await asyncio.gather(first, second, return_exceptions=True)

    assert [response.status_code for response in responses] == [503, 503]
    assert feed.fetches == 1
