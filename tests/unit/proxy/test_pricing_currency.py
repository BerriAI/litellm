import asyncio
from collections.abc import Mapping
from datetime import date, datetime, timedelta, timezone
from typing import Final

import pytest

from litellm.litellm_core_utils.ecb_reference_rates import EcbFetchFailed, EcbReferenceRates, UsdRate
from litellm.proxy.pricing_currency import PricingCurrencyRejected, ReferenceRateLookup, usd_request_prices

DAY: Final = date(2026, 11, 13)
CHAT: Final = "/v1/chat/completions"
DEADLOCK_GUARD_SECONDS: Final = 30.0


def _feed(day: date = DAY, usd: str = "1.25") -> bytes:
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<gesmes:Envelope xmlns:gesmes='http://www.gesmes.org/xml/2002-08-01'"
        " xmlns='http://www.ecb.int/vocabulary/2002-08-01/eurofxref'>"
        f"<Cube><Cube time='{day.isoformat()}'>"
        f"<Cube currency='USD' rate='{usd}'/><Cube currency='GBP' rate='0.5'/><Cube currency='JPY' rate='125'/>"
        "</Cube></Cube></gesmes:Envelope>"
    ).encode()


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


class _Feed:
    """Serves one queued body per fetch. ``served`` is set as a fetch returns, and the refresh that awaited it
    finishes without yielding again, so a waiter on ``served`` sees the refreshed rates."""

    def __init__(self, *bodies: bytes | EcbFetchFailed) -> None:
        self.bodies = bodies
        self.fetches = 0
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.served = asyncio.Event()

    async def __call__(self) -> bytes | EcbFetchFailed:
        self.fetches += 1
        self.entered.set()
        await self.release.wait()
        self.served.set()
        return self.bodies[self.fetches - 1]


def _lookup(feed: _Feed, clock: _Clock | None = None) -> ReferenceRateLookup:
    return ReferenceRateLookup(
        EcbReferenceRates(clock=clock or _Clock(datetime(2026, 11, 13, 16, tzinfo=timezone.utc))), feed
    )


@pytest.mark.parametrize(
    ("prices", "expected"),
    [
        (
            {"input_cost_per_token": 0.5, "output_cost_per_token": 2},
            {"input_cost_per_token": 0.625, "output_cost_per_token": 2.5},
        ),
        ({"input_cost_per_token": 0.5}, {"input_cost_per_token": 0.625}),
        ({"output_cost_per_token": 0.25, "input_cost_per_token": None}, {"output_cost_per_token": 0.3125}),
        (
            {"input_cost_per_token": 0, "output_cost_per_token": 4},
            {"input_cost_per_token": 0.0, "output_cost_per_token": 5.0},
        ),
    ],
)
async def test_supplied_prices_convert_to_usd_and_omitted_directions_stay_unpriced(
    prices: Mapping[str, object], expected: Mapping[str, float]
) -> None:
    result: Final = await usd_request_prices({"pricing_currency": "EUR", **prices}, CHAT, _lookup(_Feed(_feed())))

    assert result == expected


async def test_cross_rate_converts_through_eur() -> None:
    result: Final = await usd_request_prices(
        {"pricing_currency": "JPY", "input_cost_per_token": 200}, CHAT, _lookup(_Feed(_feed()))
    )

    assert result == {"input_cost_per_token": 2.0}


async def test_usd_prices_pass_through_without_any_fetch() -> None:
    feed: Final = _Feed()

    result: Final = await usd_request_prices(
        {"pricing_currency": "USD", "input_cost_per_token": 3e-6}, CHAT, _lookup(feed)
    )

    assert result == {"input_cost_per_token": 3e-6}
    assert feed.fetches == 0


@pytest.mark.parametrize(
    ("body", "route"),
    [
        pytest.param({"pricing_currency": "EUR", "input_cost_per_token": 1e-6}, "/v1/embeddings", id="non-chat-route"),
        pytest.param({"pricing_currency": "EUR", "input_cost_per_token": 1e-6}, "/v1/responses", id="responses-route"),
        pytest.param({"pricing_currency": "eur", "input_cost_per_token": 1e-6}, CHAT, id="lowercase-code"),
        pytest.param({"pricing_currency": "EURO", "input_cost_per_token": 1e-6}, CHAT, id="four-letters"),
        pytest.param({"pricing_currency": 978, "input_cost_per_token": 1e-6}, CHAT, id="numeric-code"),
        pytest.param({"pricing_currency": "EUR"}, CHAT, id="no-prices"),
        pytest.param({"pricing_currency": "EUR", "input_cost_per_token": None}, CHAT, id="only-null-price"),
        pytest.param({"pricing_currency": "EUR", "input_cost_per_token": True}, CHAT, id="boolean-price"),
        pytest.param({"pricing_currency": "EUR", "input_cost_per_token": "1e-6"}, CHAT, id="string-price"),
        pytest.param({"pricing_currency": "EUR", "input_cost_per_token": -1e-6}, CHAT, id="negative-price"),
        pytest.param({"pricing_currency": "EUR", "input_cost_per_token": float("nan")}, CHAT, id="nan-price"),
        pytest.param({"pricing_currency": "EUR", "output_cost_per_token": float("inf")}, CHAT, id="inf-price"),
        pytest.param({"pricing_currency": "EUR", "input_cost_per_token": 10**400}, CHAT, id="int-beyond-float"),
        pytest.param(
            {"pricing_currency": "EUR", "input_cost_per_token": 1e-6, "cache_read_input_token_cost": 1e-7},
            CHAT,
            id="unconverted-cache-price",
        ),
        pytest.param(
            {"pricing_currency": "EUR", "input_cost_per_token": 1e-6, "input_cost_per_second": 1e-3},
            CHAT,
            id="unconverted-per-second-price",
        ),
        pytest.param(
            {
                "pricing_currency": "EUR",
                "input_cost_per_token": 1e-6,
                "metadata": {"model_info": {"output_cost_per_token": 2e-6}},
            },
            CHAT,
            id="metadata-model-info-price",
        ),
        pytest.param({"pricing_currency": "XYZ", "input_cost_per_token": 1e-6}, CHAT, id="currency-not-in-feed"),
        pytest.param({"pricing_currency": "GBP", "input_cost_per_token": 1e308}, CHAT, id="usd-overflows"),
        pytest.param({"pricing_currency": "JPY", "input_cost_per_token": 5e-324}, CHAT, id="usd-underflows"),
    ],
)
async def test_explicit_currency_requests_that_cannot_be_priced_safely_are_rejected(
    body: Mapping[str, object], route: str
) -> None:
    result: Final = await usd_request_prices(body, route, _lookup(_Feed(_feed())))

    assert isinstance(result, PricingCurrencyRejected)
    assert result.status_code == 400


async def test_unavailable_rates_reject_with_service_unavailable() -> None:
    result: Final = await usd_request_prices(
        {"pricing_currency": "EUR", "input_cost_per_token": 1e-6}, CHAT, _lookup(_Feed(EcbFetchFailed("HTTP 503")))
    )

    assert result == PricingCurrencyRejected("ECB reference rates are unavailable, retry later", 503)


async def test_cold_lookups_share_one_refresh_that_survives_a_cancelled_caller() -> None:
    feed: Final = _Feed(_feed())
    feed.release.clear()
    lookup: Final = _lookup(feed)
    cancelled_caller: Final = asyncio.create_task(lookup.usd_per_unit("EUR"))
    waiting_caller: Final = asyncio.create_task(lookup.usd_per_unit("GBP"))
    try:
        await asyncio.wait_for(feed.entered.wait(), DEADLOCK_GUARD_SECONDS)
        cancelled_caller.cancel()
        await asyncio.gather(cancelled_caller, return_exceptions=True)
        feed.release.set()
        gbp: Final = await asyncio.wait_for(waiting_caller, DEADLOCK_GUARD_SECONDS)
    finally:
        feed.release.set()
        cancelled_caller.cancel()
        waiting_caller.cancel()
        await asyncio.gather(cancelled_caller, waiting_caller, return_exceptions=True)

    assert cancelled_caller.cancelled()
    assert gbp == UsdRate(currency="GBP", usd_per_unit=2.5, rate_date=DAY, source="ecb")
    assert feed.fetches == 1


async def test_held_rates_answer_without_waiting_for_a_due_refresh() -> None:
    clock: Final = _Clock(datetime(2026, 11, 13, 16, tzinfo=timezone.utc))
    feed: Final = _Feed(_feed(), _feed(day=DAY + timedelta(days=1), usd="2"))
    lookup: Final = _lookup(feed, clock)
    body: Final = {"pricing_currency": "EUR", "input_cost_per_token": 0.5}
    first: Final = await usd_request_prices(body, CHAT, lookup)
    feed.served.clear()
    feed.release.clear()
    clock.now = datetime(2026, 11, 14, 16, tzinfo=timezone.utc)

    try:
        while_refreshing: Final = await asyncio.wait_for(usd_request_prices(body, CHAT, lookup), DEADLOCK_GUARD_SECONDS)
    finally:
        feed.release.set()
        await asyncio.wait_for(feed.served.wait(), DEADLOCK_GUARD_SECONDS)
    after_refresh: Final = await usd_request_prices(body, CHAT, lookup)

    assert first == while_refreshing == {"input_cost_per_token": 0.625}
    assert after_refresh == {"input_cost_per_token": 1.0}


async def test_stale_rates_with_a_failed_refresh_are_rejected_not_used() -> None:
    clock: Final = _Clock(datetime(2026, 11, 13, 16, tzinfo=timezone.utc))
    lookup: Final = _lookup(_Feed(_feed(), EcbFetchFailed("HTTP 503")), clock)
    body: Final = {"pricing_currency": "EUR", "input_cost_per_token": 0.5}
    assert await usd_request_prices(body, CHAT, lookup) == {"input_cost_per_token": 0.625}
    clock.now = datetime(2026, 11, 21, 16, tzinfo=timezone.utc)

    result: Final = await asyncio.wait_for(usd_request_prices(body, CHAT, lookup), DEADLOCK_GUARD_SECONDS)

    assert result == PricingCurrencyRejected(f"ECB reference rates dated {DAY} are too old to convert prices", 503)


async def test_failed_background_refresh_keeps_the_fresh_held_rates() -> None:
    clock: Final = _Clock(datetime(2026, 11, 13, 16, tzinfo=timezone.utc))
    rates: Final = EcbReferenceRates(clock=clock)
    feed: Final = _Feed(_feed(), EcbFetchFailed("HTTP 503"))
    lookup: Final = ReferenceRateLookup(rates, feed)
    body: Final = {"pricing_currency": "EUR", "input_cost_per_token": 0.5}
    assert await usd_request_prices(body, CHAT, lookup) == {"input_cost_per_token": 0.625}
    feed.served.clear()
    clock.now = datetime(2026, 11, 14, 16, tzinfo=timezone.utc)

    try:
        during: Final = await usd_request_prices(body, CHAT, lookup)
    finally:
        await asyncio.wait_for(feed.served.wait(), DEADLOCK_GUARD_SECONDS)

    assert during == {"input_cost_per_token": 0.625}
    assert rates.usd_per_unit("EUR") == UsdRate(currency="EUR", usd_per_unit=1.25, rate_date=DAY, source="ecb")
    assert rates.refresh_due()


async def test_a_refresh_that_raises_rejects_the_request_instead_of_failing_it() -> None:
    async def raising_fetch() -> bytes | EcbFetchFailed:
        raise RuntimeError("transport bug")

    lookup: Final = ReferenceRateLookup(
        EcbReferenceRates(clock=_Clock(datetime(2026, 11, 13, 16, tzinfo=timezone.utc))), raising_fetch
    )

    result: Final = await usd_request_prices({"pricing_currency": "EUR", "input_cost_per_token": 0.5}, CHAT, lookup)

    assert result == PricingCurrencyRejected("ECB reference rates are unavailable, retry later", 503)


class _Seconds:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


async def test_a_failed_refresh_is_not_retried_until_its_cooldown_ends() -> None:
    seconds: Final = _Seconds()
    feed: Final = _Feed(EcbFetchFailed("HTTP 503"), _feed())
    lookup: Final = ReferenceRateLookup(
        EcbReferenceRates(clock=_Clock(datetime(2026, 11, 13, 16, tzinfo=timezone.utc))), feed, seconds
    )
    body: Final = {"pricing_currency": "EUR", "input_cost_per_token": 0.5}
    unavailable: Final = PricingCurrencyRejected("ECB reference rates are unavailable, retry later", 503)

    during_outage: Final = [await usd_request_prices(body, CHAT, lookup) for _ in range(3)]
    seconds.now = 59.9
    before_cooldown_ends: Final = await usd_request_prices(body, CHAT, lookup)
    fetches_before: Final = feed.fetches
    seconds.now = 60.0
    after_cooldown: Final = await usd_request_prices(body, CHAT, lookup)

    assert during_outage == [unavailable] * 3
    assert before_cooldown_ends == unavailable
    assert fetches_before == 1
    assert after_cooldown == {"input_cost_per_token": 0.625}
    assert feed.fetches == 2
    assert await usd_request_prices({**body, "pricing_currency": "USD"}, CHAT, lookup) == {"input_cost_per_token": 0.5}


class _CancelledThenServed:
    def __init__(self) -> None:
        self.fetches = 0

    async def __call__(self) -> bytes | EcbFetchFailed:
        self.fetches += 1
        if self.fetches == 1:
            raise asyncio.CancelledError
        return _feed()


async def test_a_cancelled_refresh_does_not_block_the_next_lookup() -> None:
    fetch: Final = _CancelledThenServed()
    lookup: Final = ReferenceRateLookup(
        EcbReferenceRates(clock=_Clock(datetime(2026, 11, 13, 16, tzinfo=timezone.utc))), fetch, _Seconds()
    )
    body: Final = {"pricing_currency": "EUR", "input_cost_per_token": 0.5}

    with pytest.raises(asyncio.CancelledError):
        await usd_request_prices(body, CHAT, lookup)
    after_cancel: Final = await usd_request_prices(body, CHAT, lookup)

    assert after_cancel == {"input_cost_per_token": 0.625}
    assert fetch.fetches == 2
