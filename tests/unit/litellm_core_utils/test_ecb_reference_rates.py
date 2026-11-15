from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta, timezone

import httpx
import pytest

from litellm.litellm_core_utils.ecb_reference_rates import (
    ECB_DAILY_RATES_URL,
    ECB_MAX_FEED_BYTES,
    EcbFeedRejected,
    EcbFetchFailed,
    EcbRatesStale,
    EcbRatesUnavailable,
    EcbReferenceRates,
    EcbRefreshed,
    EcbRefreshNotDue,
    UnsupportedCurrency,
    UsdRate,
    fetch_ecb_daily_feed,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

DAY = date(2026, 11, 13)
QUOTES = (("USD", "1.25"), ("GBP", "0.5"), ("JPY", "125"))


def _feed(day: str = DAY.isoformat(), quotes: tuple[tuple[str, str], ...] = QUOTES) -> bytes:
    cubes = "".join(f"<Cube currency='{code}' rate='{rate}'/>" for code, rate in quotes)
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<gesmes:Envelope xmlns:gesmes='http://www.gesmes.org/xml/2002-08-01'"
        " xmlns='http://www.ecb.int/vocabulary/2002-08-01/eurofxref'>"
        "<gesmes:subject>Reference rates</gesmes:subject>"
        f"<Cube><Cube time='{day}'>{cubes}</Cube></Cube>"
        "</gesmes:Envelope>"
    ).encode()


def _at(day: date, hour: int = 16) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=timezone.utc)


def _serve(body: bytes | EcbFetchFailed) -> Callable[[], Awaitable[bytes | EcbFetchFailed]]:
    async def fetch() -> bytes | EcbFetchFailed:
        return body

    return fetch


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


async def _rates_holding(feed: bytes, clock: _Clock) -> EcbReferenceRates:
    rates = EcbReferenceRates(clock=clock)
    assert isinstance(await rates.arefresh(_serve(feed)), EcbRefreshed)
    return rates


async def test_fetch_returns_the_feed_body(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, content=_feed())

    http = AsyncHTTPHandler(transport=httpx.MockTransport(handler))
    try:
        assert await fetch_ecb_daily_feed(http) == _feed()
    finally:
        await http.close()
    assert seen == [ECB_DAILY_RATES_URL]


def _status_503(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, content=b"unavailable")


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _oversized(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=_feed() + b" " * ECB_MAX_FEED_BYTES)


@pytest.mark.parametrize("handler", [_status_503, _connect_error, _read_timeout, _oversized])
async def test_fetch_failures_are_values(
    handler: Callable[[httpx.Request], httpx.Response], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    http = AsyncHTTPHandler(transport=httpx.MockTransport(handler))
    try:
        assert isinstance(await fetch_ecb_daily_feed(http), EcbFetchFailed)
    finally:
        await http.close()


async def test_fetch_in_offline_mode_sends_no_request(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", "true")
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=_feed())

    http = AsyncHTTPHandler(transport=httpx.MockTransport(handler))
    try:
        assert isinstance(await fetch_ecb_daily_feed(http), EcbFetchFailed)
    finally:
        await http.close()
    assert seen == []


@pytest.mark.parametrize(
    ("currency", "usd_per_unit"),
    [("EUR", 1.25), ("GBP", 2.5), ("JPY", 0.01)],
)
async def test_cross_rates_convert_through_eur(currency: str, usd_per_unit: float):
    rates = await _rates_holding(_feed(), _Clock(_at(DAY)))

    assert rates.usd_per_unit(currency) == UsdRate(
        currency=currency, usd_per_unit=usd_per_unit, rate_date=DAY, source="ecb"
    )


def test_usd_is_identity_without_any_feed():
    rates = EcbReferenceRates(clock=_Clock(_at(DAY)))

    assert rates.usd_per_unit("USD") == UsdRate(currency="USD", usd_per_unit=1.0, rate_date=None, source="identity")
    assert rates.usd_per_unit("EUR") == EcbRatesUnavailable(currency="EUR")


@pytest.mark.parametrize("currency", ["XYZ", "gbp", ""])
async def test_currency_outside_the_feed_is_unsupported(currency: str):
    rates = await _rates_holding(_feed(), _Clock(_at(DAY)))

    assert rates.usd_per_unit(currency) == UnsupportedCurrency(currency=currency, rate_date=DAY)


@pytest.mark.parametrize(
    ("now", "usable"),
    [
        (datetime(2026, 11, 20, 23, 59, 59, tzinfo=timezone.utc), True),
        (datetime(2026, 11, 21, 4, 0, tzinfo=timezone(timedelta(hours=5))), True),
        (datetime(2026, 11, 21, 0, 0, tzinfo=timezone.utc), False),
        (datetime(2026, 11, 20, 20, 0, tzinfo=timezone(timedelta(hours=-5))), False),
    ],
)
async def test_rates_are_usable_for_seven_calendar_days_after_the_rate_date(now: datetime, usable: bool):
    clock = _Clock(_at(DAY))
    rates = await _rates_holding(_feed(), clock)
    clock.now = now

    result = rates.usd_per_unit("GBP")

    assert isinstance(result, UsdRate) is usable
    if not usable:
        assert result == EcbRatesStale(currency="GBP", rate_date=DAY, as_of=date(2026, 11, 21))


async def test_rates_dated_after_the_clock_are_stale():
    clock = _Clock(_at(DAY))
    rates = await _rates_holding(_feed(), clock)
    clock.now = _at(DAY - timedelta(days=1))

    assert rates.usd_per_unit("GBP") == EcbRatesStale(currency="GBP", rate_date=DAY, as_of=DAY - timedelta(days=1))


async def test_refresh_is_due_once_a_day():
    clock = _Clock(_at(DAY))
    rates = await _rates_holding(_feed(), clock)
    newer = _feed(day=(DAY + timedelta(days=1)).isoformat(), quotes=(("USD", "2"), ("GBP", "1")))

    clock.now = _at(DAY) + timedelta(days=1) - timedelta(microseconds=1)
    not_due = await rates.arefresh(_serve(newer))
    clock.now = _at(DAY) + timedelta(days=1)
    due = await rates.arefresh(_serve(newer))

    assert isinstance(not_due, EcbRefreshNotDue)
    assert not_due.snapshot.rate_date == DAY
    assert isinstance(due, EcbRefreshed)
    assert rates.usd_per_unit("GBP") == UsdRate(
        currency="GBP", usd_per_unit=2.0, rate_date=DAY + timedelta(days=1), source="ecb"
    )


async def test_same_day_feed_counts_as_the_daily_check():
    clock = _Clock(_at(DAY))
    rates = await _rates_holding(_feed(), clock)
    clock.now = _at(DAY + timedelta(days=1))
    assert isinstance(await rates.arefresh(_serve(_feed())), EcbRefreshed)

    clock.now = _at(DAY + timedelta(days=1), hour=20)

    assert isinstance(await rates.arefresh(_serve(_feed())), EcbRefreshNotDue)


async def test_first_refresh_failure_leaves_only_usd_available():
    rates = EcbReferenceRates(clock=_Clock(_at(DAY)))

    assert await rates.arefresh(_serve(EcbFetchFailed("HTTP 503"))) == EcbFetchFailed("HTTP 503")
    assert rates.usd_per_unit("EUR") == EcbRatesUnavailable(currency="EUR")
    assert isinstance(rates.usd_per_unit("USD"), UsdRate)


def _feed_with_rate(rate: str) -> bytes:
    return _feed(quotes=(("USD", "1.25"), ("GBP", rate)))


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(EcbFetchFailed("HTTP 503"), id="fetch-failed"),
        pytest.param(b"not xml", id="not-xml"),
        pytest.param(b"", id="empty"),
        pytest.param(_feed() + b" " * ECB_MAX_FEED_BYTES, id="oversized"),
        pytest.param(_feed().replace(b"?>", b"?><!DOCTYPE x [<!ENTITY a 'b'>]>", 1), id="doctype"),
        pytest.param(_feed(quotes=(("GBP", "0.5"),)), id="usd-missing"),
        pytest.param(_feed(quotes=()), id="no-quotes"),
        pytest.param(_feed(quotes=(("USD", "1.25"), ("GBP", "0.5"), ("GBP", "0.6"))), id="duplicate"),
        pytest.param(_feed(quotes=(("USD", "1.25"), ("EUR", "1"))), id="eur-quoted"),
        pytest.param(_feed(quotes=(("USD", "1.25"), ("gbp", "0.5"))), id="lowercase-code"),
        pytest.param(_feed_with_rate("0"), id="zero"),
        pytest.param(_feed_with_rate("0.0000"), id="zero-decimal"),
        pytest.param(_feed_with_rate("-0.5"), id="negative"),
        pytest.param(_feed_with_rate("NaN"), id="nan"),
        pytest.param(_feed_with_rate("inf"), id="inf"),
        pytest.param(_feed_with_rate("9" * 400), id="overflows-to-inf"),
        pytest.param(_feed_with_rate("5e-1"), id="exponent"),
        pytest.param(_feed_with_rate(""), id="empty-rate"),
        pytest.param(_feed().replace(b"<Cube currency='JPY' rate='125'/>", b"<Cube currency='JPY'/>"), id="no-rate"),
        pytest.param(_feed(day="2026-13-01"), id="invalid-date"),
        pytest.param(_feed(day="20261113"), id="compact-date"),
        pytest.param(_feed(day=(DAY + timedelta(days=2)).isoformat()), id="future-date"),
        pytest.param(_feed(day=(DAY - timedelta(days=1)).isoformat()), id="older-than-held"),
        pytest.param(_feed().replace(b"time=", b"date="), id="undated"),
        pytest.param(
            _feed().replace(
                b"</Cube></Cube>", b"</Cube><Cube time='2026-11-12'><Cube currency='USD' rate='1'/></Cube></Cube>"
            ),
            id="two-rate-sets",
        ),
    ],
)
async def test_unusable_refresh_keeps_the_last_good_rates(body: bytes | EcbFetchFailed):
    clock = _Clock(_at(DAY))
    rates = await _rates_holding(_feed(), clock)
    clock.now = _at(DAY + timedelta(days=1))

    result = await rates.arefresh(_serve(body))

    assert isinstance(result, EcbFetchFailed | EcbFeedRejected)
    assert rates.usd_per_unit("GBP") == UsdRate(currency="GBP", usd_per_unit=2.5, rate_date=DAY, source="ecb")
    assert rates.refresh_due()


async def test_held_rates_survive_a_later_refresh():
    clock = _Clock(_at(DAY))
    rates = EcbReferenceRates(clock=clock)
    first = await rates.arefresh(_serve(_feed()))
    held = rates.usd_per_unit("GBP")
    clock.now = _at(DAY + timedelta(days=1))

    await rates.arefresh(_serve(_feed(day=(DAY + timedelta(days=1)).isoformat(), quotes=(("USD", "2"), ("GBP", "1")))))

    assert held == UsdRate(currency="GBP", usd_per_unit=2.5, rate_date=DAY, source="ecb")
    assert isinstance(first, EcbRefreshed)
    assert dict(first.snapshot.units_per_eur) == {"EUR": 1.0, "USD": 1.25, "GBP": 0.5, "JPY": 125.0}
    with pytest.raises(TypeError):
        first.snapshot.units_per_eur["GBP"] = 9.0
