"""ECB euro foreign exchange reference rates, expressed as USD per unit of a currency.

These are informational reference rates, not settlement quotes. ``EcbReferenceRates.arefresh`` is the only
operation that touches the network and ``usd_per_unit`` never does, so callers refresh outside request handling.
"""

import asyncio
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from http import HTTPStatus
from types import MappingProxyType
from typing import Final, Literal, Protocol, TypeAlias

import httpx

from litellm.litellm_core_utils.offline_mode import is_offline_mode

ECB_DAILY_RATES_URL: Final = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
ECB_FETCH_TIMEOUT_SECONDS: Final = 5.0
ECB_MAX_FEED_BYTES: Final = 64 * 1024
ECB_MAX_RATE_AGE_DAYS: Final = 7
ECB_REFRESH_INTERVAL: Final = timedelta(days=1)

_CUBE: Final = "{http://www.ecb.int/vocabulary/2002-08-01/eurofxref}Cube"
_CURRENCY_CODE: Final = re.compile(r"[A-Z]{3}")
_RATE: Final = re.compile(r"[0-9]+(?:\.[0-9]+)?")
_RATE_DATE: Final = re.compile(r"[0-9]{4}-[0-9]{2}-[0-9]{2}")


@dataclass(frozen=True, slots=True)
class EcbFetchFailed:
    reason: str


@dataclass(frozen=True, slots=True)
class EcbFeedRejected:
    reason: str


@dataclass(frozen=True, slots=True)
class EcbRateSnapshot:
    rate_date: date
    units_per_eur: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class EcbRefreshed:
    snapshot: EcbRateSnapshot


@dataclass(frozen=True, slots=True)
class EcbRefreshNotDue:
    snapshot: EcbRateSnapshot


@dataclass(frozen=True, slots=True)
class UsdRate:
    currency: str
    usd_per_unit: float
    rate_date: date | None
    source: Literal["identity", "ecb"]


@dataclass(frozen=True, slots=True)
class EcbRatesUnavailable:
    currency: str


@dataclass(frozen=True, slots=True)
class EcbRatesStale:
    currency: str
    rate_date: date
    as_of: date


@dataclass(frozen=True, slots=True)
class UnsupportedCurrency:
    currency: str
    rate_date: date


EcbRefreshResult: TypeAlias = EcbRefreshed | EcbRefreshNotDue | EcbFetchFailed | EcbFeedRejected
UsdRateLookup: TypeAlias = UsdRate | EcbRatesUnavailable | EcbRatesStale | UnsupportedCurrency

_USD_IDENTITY: Final = UsdRate(currency="USD", usd_per_unit=1.0, rate_date=None, source="identity")


def _utc_date(moment: datetime) -> date:
    return moment.astimezone(timezone.utc).date()


def _is_usable(rate: float) -> bool:
    return math.isfinite(rate) and rate > 0


def _parse_quote(cube: ET.Element) -> tuple[str, float] | EcbFeedRejected:
    code: Final = cube.get("currency")
    text: Final = cube.get("rate")
    if cube.tag != _CUBE or code is None or text is None:
        return EcbFeedRejected("malformed rate entry")
    if _CURRENCY_CODE.fullmatch(code) is None:
        return EcbFeedRejected(f"invalid currency code {code!r}")
    if _RATE.fullmatch(text) is None:
        return EcbFeedRejected(f"invalid rate {text!r} for {code}")
    rate: Final = float(text)
    if not _is_usable(rate):
        return EcbFeedRejected(f"unusable rate {text!r} for {code}")
    return code, rate


def _parse_rate_date(text: str | None) -> date | None:
    if text is None or _RATE_DATE.fullmatch(text) is None:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def parse_ecb_daily_feed(body: bytes, today: date) -> EcbRateSnapshot | EcbFeedRejected:
    if len(body) > ECB_MAX_FEED_BYTES:
        return EcbFeedRejected("feed exceeds the size limit")
    try:
        text: Final = body.decode("utf-8")
    except UnicodeDecodeError:
        return EcbFeedRejected("feed is not UTF-8")
    # expat detects UTF-16 and UTF-32 from NUL bytes even when the parser encoding is forced
    if "\x00" in text:
        return EcbFeedRejected("feed is not UTF-8")
    if "<!DOCTYPE" in text:
        return EcbFeedRejected("feed declares a DOCTYPE")
    try:
        root: Final = ET.fromstring(body, parser=ET.XMLParser(encoding="utf-8"))
    except ET.ParseError as e:
        return EcbFeedRejected(f"feed is not well-formed XML: {e}")
    dated: Final = root.findall(f"{_CUBE}/{_CUBE}[@time]")
    if len(dated) != 1:
        return EcbFeedRejected(f"expected one dated rate set, found {len(dated)}")
    rate_date: Final = _parse_rate_date(dated[0].get("time"))
    if rate_date is None:
        return EcbFeedRejected("invalid rate date")
    if rate_date > today:
        return EcbFeedRejected(f"rate date {rate_date} is after {today}")
    if (today - rate_date).days > ECB_MAX_RATE_AGE_DAYS:
        return EcbFeedRejected(f"rate date {rate_date} is more than {ECB_MAX_RATE_AGE_DAYS} days before {today}")
    parsed: Final = tuple(map(_parse_quote, dated[0]))
    failures: Final = tuple(entry for entry in parsed if isinstance(entry, EcbFeedRejected))
    if failures:
        return failures[0]
    quotes: Final = tuple(entry for entry in parsed if not isinstance(entry, EcbFeedRejected))
    codes: Final = tuple(code for code, _ in quotes)
    if len(frozenset(codes)) != len(codes):
        return EcbFeedRejected("duplicate currency code")
    if "EUR" in codes:
        return EcbFeedRejected("EUR listed as a quote currency")
    if "USD" not in codes:
        return EcbFeedRejected("USD rate missing")
    units_per_eur: Final = MappingProxyType({"EUR": 1.0, **dict(quotes)})
    usd_per_eur: Final = units_per_eur["USD"]
    unusable: Final = tuple(code for code, units in units_per_eur.items() if not _is_usable(usd_per_eur / units))
    if unusable:
        return EcbFeedRejected(f"USD cross rate out of range for {unusable[0]}")
    return EcbRateSnapshot(rate_date=rate_date, units_per_eur=units_per_eur)


class EcbFeedHttpClient(Protocol):
    async def get(self, url: str, *, timeout: float, max_response_bytes: int) -> httpx.Response: ...


async def fetch_ecb_daily_feed(http: EcbFeedHttpClient) -> bytes | EcbFetchFailed:
    if is_offline_mode():
        return EcbFetchFailed("LITELLM_OFFLINE_MODE is set")
    try:
        response: Final = await asyncio.wait_for(
            http.get(ECB_DAILY_RATES_URL, timeout=ECB_FETCH_TIMEOUT_SECONDS, max_response_bytes=ECB_MAX_FEED_BYTES),
            timeout=ECB_FETCH_TIMEOUT_SECONDS,
        )
    except (httpx.HTTPError, ValueError, asyncio.TimeoutError) as e:
        return EcbFetchFailed(type(e).__name__)
    if response.status_code != HTTPStatus.OK:
        return EcbFetchFailed(f"HTTP {response.status_code}")
    return response.content


@dataclass(frozen=True, slots=True)
class _CacheState:
    snapshot: EcbRateSnapshot
    checked_at: datetime


class EcbReferenceRates:
    def __init__(self, clock: Callable[[], datetime]) -> None:
        self._clock: Final = clock
        self._state: _CacheState | None = None

    def refresh_due(self) -> bool:
        state: Final = self._state
        return state is None or self._clock() - state.checked_at >= ECB_REFRESH_INTERVAL

    async def arefresh(self, fetch: Callable[[], Awaitable[bytes | EcbFetchFailed]]) -> EcbRefreshResult:
        held: Final = self._state
        if held is not None and not self.refresh_due():
            return EcbRefreshNotDue(held.snapshot)
        body: Final = await fetch()
        if isinstance(body, EcbFetchFailed):
            return body
        checked_at: Final = self._clock()
        parsed: Final = parse_ecb_daily_feed(body, today=_utc_date(checked_at))
        if isinstance(parsed, EcbFeedRejected):
            return parsed
        current: Final = self._state
        if current is not None and parsed.rate_date < current.snapshot.rate_date:
            return EcbFeedRejected(f"rate date {parsed.rate_date} is older than held {current.snapshot.rate_date}")
        self._state = _CacheState(snapshot=parsed, checked_at=checked_at)
        return EcbRefreshed(parsed)

    def usd_per_unit(self, currency: str) -> UsdRateLookup:
        if currency == "USD":
            return _USD_IDENTITY
        state: Final = self._state
        if state is None:
            return EcbRatesUnavailable(currency)
        snapshot: Final = state.snapshot
        today: Final = _utc_date(self._clock())
        if not 0 <= (today - snapshot.rate_date).days <= ECB_MAX_RATE_AGE_DAYS:
            return EcbRatesStale(currency=currency, rate_date=snapshot.rate_date, as_of=today)
        units_per_eur: Final = snapshot.units_per_eur.get(currency)
        if units_per_eur is None:
            return UnsupportedCurrency(currency=currency, rate_date=snapshot.rate_date)
        return UsdRate(
            currency=currency,
            usd_per_unit=snapshot.units_per_eur["USD"] / units_per_eur,
            rate_date=snapshot.rate_date,
            source="ecb",
        )

    def usd_rates(self) -> Mapping[str, UsdRate]:
        state: Final = self._state
        currencies: Final = () if state is None else tuple(c for c in state.snapshot.units_per_eur if c != "USD")
        lookups: Final = tuple(self.usd_per_unit(currency) for currency in currencies)
        return MappingProxyType({rate.currency: rate for rate in lookups if isinstance(rate, UsdRate)})
