import asyncio
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timedelta, timezone
from typing import Final

import httpx
import pytest

from litellm.litellm_core_utils.ecb_reference_rates import (
    ECB_DAILY_RATES_URL,
    ECB_FETCH_TIMEOUT_SECONDS,
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
    UsdRateLookup,
    fetch_ecb_daily_feed,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler

DAY: Final = date(2026, 11, 13)
NEXT_DAY: Final = DAY + timedelta(days=1)
QUOTES: Final = (("USD", "1.25"), ("GBP", "0.5"), ("JPY", "125"))
DEADLOCK_GUARD_SECONDS: Final = 30.0
HELD_GBP: Final = UsdRate(currency="GBP", usd_per_unit=2.5, rate_date=DAY, source="ecb")
DOCTYPE: Final = "<!DOCTYPE x [<!ENTITY a 'b'>]>"


def _feed(day: str = DAY.isoformat(), quotes: tuple[tuple[str, str], ...] = QUOTES) -> bytes:
    cubes: Final = "".join(f"<Cube currency='{code}' rate='{rate}'/>" for code, rate in quotes)
    return (
        "<?xml version='1.0' encoding='UTF-8'?>"
        "<gesmes:Envelope xmlns:gesmes='http://www.gesmes.org/xml/2002-08-01'"
        " xmlns='http://www.ecb.int/vocabulary/2002-08-01/eurofxref'>"
        "<gesmes:subject>Reference rates</gesmes:subject>"
        f"<Cube><Cube time='{day}'>{cubes}</Cube></Cube>"
        "</gesmes:Envelope>"
    ).encode()


def _next_day_feed() -> bytes:
    return _feed(day=NEXT_DAY.isoformat(), quotes=(("USD", "2"), ("GBP", "1")))


def _feed_with_rate(rate: str) -> bytes:
    return _feed(quotes=(("USD", "1.25"), ("GBP", rate)))


def _doctype_feed_in(codec: str) -> bytes:
    return (
        _feed()
        .decode()
        .replace("encoding='UTF-8'?>", f"encoding='UTF-16'?>{DOCTYPE}")
        .replace("<gesmes:subject>Reference rates", "<gesmes:subject>&a;")
        .encode(codec)
    )


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
    rates: Final = EcbReferenceRates(clock=clock)
    assert isinstance(await rates.arefresh(_serve(feed)), EcbRefreshed)
    return rates


def _feed_from_ecb_url_only(request: httpx.Request) -> httpx.Response:
    if str(request.url) != ECB_DAILY_RATES_URL:
        return httpx.Response(404)
    return httpx.Response(200, content=_feed())


async def test_fetch_returns_the_feed_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    http: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_feed_from_ecb_url_only))
    try:
        assert await fetch_ecb_daily_feed(http) == _feed()
    finally:
        await http.close()


def _status_503(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, content=b"unavailable")


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("proxy http://user:secret@proxy.internal refused", request=request)


def _read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out", request=request)


def _oversized(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=_feed() + b" " * ECB_MAX_FEED_BYTES)


@pytest.mark.parametrize(
    ("handler", "failure"),
    [
        (_status_503, EcbFetchFailed("HTTP 503")),
        (_connect_error, EcbFetchFailed("ConnectError")),
        (_read_timeout, EcbFetchFailed("ReadTimeout")),
        (_oversized, EcbFetchFailed("HTTPResponseLimitError")),
    ],
)
async def test_fetch_failures_are_values_without_error_details(
    handler: Callable[[httpx.Request], httpx.Response], failure: EcbFetchFailed, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    http: Final = AsyncHTTPHandler(transport=httpx.MockTransport(handler))
    try:
        assert await fetch_ecb_daily_feed(http) == failure
    finally:
        await http.close()


def _no_request_allowed(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"offline mode sent {request.method} {request.url}")


class _SkippableClockLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        self.skipped_seconds = 0.0
        super().__init__()

    def time(self) -> float:
        return super().time() + self.skipped_seconds

    def skip(self, seconds: float) -> None:
        self.skipped_seconds += seconds


class _StalledGet:
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.cancelled = asyncio.Event()
        self.released = asyncio.Event()

    async def get(self, url: str, *, timeout: float, max_response_bytes: int) -> httpx.Response:
        self.entered.set()
        try:
            await self.released.wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return httpx.Response(200, content=_feed())


def test_stalled_fetch_is_cancelled_at_the_overall_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_OFFLINE_MODE", raising=False)
    loop: Final = _SkippableClockLoop()

    async def fetch_past_the_deadline() -> tuple[bytes | EcbFetchFailed, bool]:
        http: Final = _StalledGet()
        fetch: Final = asyncio.create_task(fetch_ecb_daily_feed(http))
        try:
            await asyncio.wait_for(http.entered.wait(), DEADLOCK_GUARD_SECONDS)
            loop.skip(ECB_FETCH_TIMEOUT_SECONDS + 1)
            result: Final = await asyncio.wait_for(asyncio.shield(fetch), DEADLOCK_GUARD_SECONDS)
            return result, http.cancelled.is_set()
        finally:
            http.released.set()
            fetch.cancel()
            await asyncio.gather(fetch, return_exceptions=True)

    try:
        result, get_was_cancelled = loop.run_until_complete(fetch_past_the_deadline())
    finally:
        loop.close()

    assert result == EcbFetchFailed("TimeoutError")
    assert get_was_cancelled


async def test_fetch_in_offline_mode_sends_no_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_OFFLINE_MODE", "true")
    http: Final = AsyncHTTPHandler(transport=httpx.MockTransport(_no_request_allowed))
    try:
        assert isinstance(await fetch_ecb_daily_feed(http), EcbFetchFailed)
    finally:
        await http.close()


@pytest.mark.parametrize(
    ("currency", "usd_per_unit"),
    [("EUR", 1.25), ("GBP", 2.5), ("JPY", 0.01)],
)
async def test_cross_rates_convert_through_eur(currency: str, usd_per_unit: float) -> None:
    rates: Final = await _rates_holding(_feed(), _Clock(_at(DAY)))

    assert rates.usd_per_unit(currency) == UsdRate(
        currency=currency, usd_per_unit=usd_per_unit, rate_date=DAY, source="ecb"
    )


def test_usd_is_identity_without_any_feed() -> None:
    rates: Final = EcbReferenceRates(clock=_Clock(_at(DAY)))

    assert rates.usd_per_unit("USD") == UsdRate(currency="USD", usd_per_unit=1.0, rate_date=None, source="identity")
    assert rates.usd_per_unit("EUR") == EcbRatesUnavailable(currency="EUR")


@pytest.mark.parametrize("currency", ["XYZ", "gbp", ""])
async def test_currency_outside_the_feed_is_unsupported(currency: str) -> None:
    rates: Final = await _rates_holding(_feed(), _Clock(_at(DAY)))

    assert rates.usd_per_unit(currency) == UnsupportedCurrency(currency=currency, rate_date=DAY)


@pytest.mark.parametrize(
    ("now", "expected"),
    [
        (datetime(2026, 11, 20, 23, 59, 59, tzinfo=timezone.utc), HELD_GBP),
        (datetime(2026, 11, 21, 4, 0, tzinfo=timezone(timedelta(hours=5))), HELD_GBP),
        (
            datetime(2026, 11, 21, 0, 0, tzinfo=timezone.utc),
            EcbRatesStale(currency="GBP", rate_date=DAY, as_of=date(2026, 11, 21)),
        ),
        (
            datetime(2026, 11, 20, 20, 0, tzinfo=timezone(timedelta(hours=-5))),
            EcbRatesStale(currency="GBP", rate_date=DAY, as_of=date(2026, 11, 21)),
        ),
        (_at(DAY - timedelta(days=1)), EcbRatesStale(currency="GBP", rate_date=DAY, as_of=DAY - timedelta(days=1))),
    ],
)
async def test_held_rates_are_usable_for_seven_utc_calendar_days_after_the_rate_date(
    now: datetime, expected: UsdRateLookup
) -> None:
    clock: Final = _Clock(_at(DAY))
    rates: Final = await _rates_holding(_feed(), clock)
    clock.now = now

    assert rates.usd_per_unit("GBP") == expected


@pytest.mark.parametrize(("days_old", "accepted"), [(7, True), (8, False)])
async def test_first_feed_older_than_seven_days_is_rejected_and_stays_due(days_old: int, accepted: bool) -> None:
    rates: Final = EcbReferenceRates(clock=_Clock(_at(DAY)))

    result: Final = await rates.arefresh(_serve(_feed(day=(DAY - timedelta(days=days_old)).isoformat())))

    assert isinstance(result, EcbRefreshed if accepted else EcbFeedRejected)
    assert isinstance(rates.usd_per_unit("GBP"), UsdRate if accepted else EcbRatesUnavailable)
    assert rates.refresh_due() is not accepted


async def test_refresh_is_due_once_a_day() -> None:
    clock: Final = _Clock(_at(DAY))
    rates: Final = await _rates_holding(_feed(), clock)

    clock.now = _at(DAY) + timedelta(days=1) - timedelta(microseconds=1)
    not_due: Final = await rates.arefresh(_serve(_next_day_feed()))
    clock.now = _at(DAY) + timedelta(days=1)
    due: Final = await rates.arefresh(_serve(_next_day_feed()))

    assert isinstance(not_due, EcbRefreshNotDue)
    assert not_due.snapshot.rate_date == DAY
    assert isinstance(due, EcbRefreshed)
    assert rates.usd_per_unit("GBP") == UsdRate(currency="GBP", usd_per_unit=2.0, rate_date=NEXT_DAY, source="ecb")


async def test_same_day_feed_counts_as_the_daily_check() -> None:
    clock: Final = _Clock(_at(DAY))
    rates: Final = await _rates_holding(_feed(), clock)
    clock.now = _at(NEXT_DAY)
    assert isinstance(await rates.arefresh(_serve(_feed())), EcbRefreshed)

    clock.now = _at(NEXT_DAY, hour=20)

    assert isinstance(await rates.arefresh(_serve(_feed())), EcbRefreshNotDue)


async def test_same_day_correction_applies_at_the_next_daily_refresh() -> None:
    clock: Final = _Clock(_at(DAY))
    rates: Final = await _rates_holding(_feed(quotes=(("USD", "1.2"), ("GBP", "0.5"))), clock)
    held: Final = rates.usd_per_unit("EUR")
    corrected: Final = _feed(quotes=(("USD", "1.5"), ("GBP", "0.5")))

    clock.now = _at(DAY, hour=20)
    not_due: Final = await rates.arefresh(_serve(corrected))
    still_held: Final = rates.usd_per_unit("EUR")
    clock.now = _at(NEXT_DAY)
    due: Final = await rates.arefresh(_serve(corrected))

    assert isinstance(not_due, EcbRefreshNotDue)
    assert still_held == UsdRate(currency="EUR", usd_per_unit=1.2, rate_date=DAY, source="ecb")
    assert isinstance(due, EcbRefreshed)
    assert rates.usd_per_unit("EUR") == UsdRate(currency="EUR", usd_per_unit=1.5, rate_date=DAY, source="ecb")
    assert held == UsdRate(currency="EUR", usd_per_unit=1.2, rate_date=DAY, source="ecb")


async def test_first_refresh_failure_leaves_only_usd_available() -> None:
    rates: Final = EcbReferenceRates(clock=_Clock(_at(DAY)))

    assert await rates.arefresh(_serve(EcbFetchFailed("HTTP 503"))) == EcbFetchFailed("HTTP 503")
    assert rates.usd_per_unit("EUR") == EcbRatesUnavailable(currency="EUR")
    assert isinstance(rates.usd_per_unit("USD"), UsdRate)


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(EcbFetchFailed("HTTP 503"), id="fetch-failed"),
        pytest.param(b"not xml", id="not-xml"),
        pytest.param(b"", id="empty"),
        pytest.param(_feed() + b" " * ECB_MAX_FEED_BYTES, id="oversized"),
        pytest.param(_feed().replace(b"?>", f"?>{DOCTYPE}".encode(), 1), id="doctype"),
        pytest.param(_doctype_feed_in("utf-16"), id="utf16-bom-doctype"),
        pytest.param(_doctype_feed_in("utf-16-le"), id="utf16le-doctype"),
        pytest.param(_doctype_feed_in("utf-16-be"), id="utf16be-doctype"),
        pytest.param(
            _feed().replace(
                b"encoding='UTF-8'?>", b"encoding='utf-7'?>+ADwAIQ-DOCTYPE x +AFsAPAAh-ENTITY a 'b'+AD4AXQA+-"
            ),
            id="utf7-doctype",
        ),
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
        pytest.param(
            _feed(quotes=(("USD", "1" + "0" * 300), ("GBP", "0." + "0" * 20 + "1"))), id="cross-rate-overflows"
        ),
        pytest.param(
            _feed(quotes=(("USD", "0." + "0" * 299 + "1"), ("GBP", "1" + "0" * 30))), id="cross-rate-underflows"
        ),
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
async def test_unusable_refresh_keeps_the_last_good_rates(body: bytes | EcbFetchFailed) -> None:
    clock: Final = _Clock(_at(DAY))
    rates: Final = await _rates_holding(_feed(), clock)
    clock.now = _at(NEXT_DAY)

    result: Final = await rates.arefresh(_serve(body))

    assert isinstance(result, EcbFetchFailed | EcbFeedRejected)
    assert rates.usd_per_unit("GBP") == HELD_GBP
    assert rates.usd_per_unit("EUR") == UsdRate(currency="EUR", usd_per_unit=1.25, rate_date=DAY, source="ecb")
    assert rates.refresh_due()


async def test_delayed_older_response_cannot_replace_a_newer_refresh() -> None:
    rates: Final = EcbReferenceRates(clock=_Clock(_at(NEXT_DAY)))
    older_requested: Final = asyncio.Event()
    release_older: Final = asyncio.Event()

    async def delayed_older_feed() -> bytes | EcbFetchFailed:
        older_requested.set()
        await release_older.wait()
        return _feed()

    slow_refresh: Final = asyncio.create_task(rates.arefresh(delayed_older_feed))
    try:
        await asyncio.wait_for(older_requested.wait(), timeout=DEADLOCK_GUARD_SECONDS)
        newer: Final = await rates.arefresh(_serve(_next_day_feed()))
        release_older.set()
        older: Final = await asyncio.wait_for(slow_refresh, timeout=DEADLOCK_GUARD_SECONDS)
    finally:
        release_older.set()
        slow_refresh.cancel()
        await asyncio.gather(slow_refresh, return_exceptions=True)

    assert isinstance(newer, EcbRefreshed)
    assert isinstance(older, EcbFeedRejected)
    assert rates.usd_per_unit("GBP") == UsdRate(currency="GBP", usd_per_unit=2.0, rate_date=NEXT_DAY, source="ecb")


async def test_held_rates_survive_a_later_refresh() -> None:
    clock: Final = _Clock(_at(DAY))
    rates: Final = EcbReferenceRates(clock=clock)
    first: Final = await rates.arefresh(_serve(_feed()))
    held: Final = rates.usd_per_unit("GBP")
    clock.now = _at(NEXT_DAY)

    assert isinstance(await rates.arefresh(_serve(_next_day_feed())), EcbRefreshed)

    assert held == HELD_GBP
    assert isinstance(first, EcbRefreshed)
    assert dict(first.snapshot.units_per_eur) == {"EUR": 1.0, "USD": 1.25, "GBP": 0.5, "JPY": 125.0}
    with pytest.raises(TypeError):
        first.snapshot.units_per_eur["GBP"] = 9.0


def test_all_usd_rates_are_empty_without_a_feed() -> None:
    assert EcbReferenceRates(clock=_Clock(_at(DAY))).usd_rates() == {}


async def test_all_usd_rates_list_every_held_currency_except_usd_until_they_go_stale() -> None:
    clock: Final = _Clock(_at(DAY))
    rates: Final = await _rates_holding(_feed(), clock)

    fresh: Final = rates.usd_rates()
    clock.now = _at(DAY + timedelta(days=8))

    assert fresh == {
        "EUR": UsdRate(currency="EUR", usd_per_unit=1.25, rate_date=DAY, source="ecb"),
        "GBP": HELD_GBP,
        "JPY": UsdRate(currency="JPY", usd_per_unit=0.01, rate_date=DAY, source="ecb"),
    }
    assert rates.usd_rates() == {}
