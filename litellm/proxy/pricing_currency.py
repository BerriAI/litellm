"""Converts a chat completion's explicitly supplied token prices from ``pricing_currency`` to USD before routing."""

import asyncio
import math
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from types import MappingProxyType
from typing import Final, Literal

from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.litellm_core_utils.ecb_reference_rates import (
    EcbFeedRejected,
    EcbFetchFailed,
    EcbRatesStale,
    EcbRatesUnavailable,
    EcbReferenceRates,
    EcbRefreshResult,
    UnsupportedCurrency,
    UsdRate,
    UsdRateLookup,
    fetch_ecb_daily_feed,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory has an untyped params argument
)
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.utils import CustomPricingLiteLLMParams

PRICING_CURRENCY_FIELD: Final = "pricing_currency"
CHAT_COMPLETION_ROUTES: Final = frozenset(
    {
        "/v1/chat/completions",
        "/chat/completions",
        "/engines/{model:path}/chat/completions",
        "/openai/deployments/{model:path}/chat/completions",
    }
)
_CONVERTED_PRICE_FIELDS: Final = ("input_cost_per_token", "output_cost_per_token")
_PRICING_FIELDS: Final = frozenset(CustomPricingLiteLLMParams.model_fields)
_UNCONVERTED_PRICING_FIELDS: Final = tuple(sorted(_PRICING_FIELDS - frozenset(_CONVERTED_PRICE_FIELDS)))
_CURRENCY_CODE: Final = re.compile(r"[A-Z]{3}")
_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])
_FAILED_REFRESH_COOLDOWN_SECONDS: Final = 60.0


@dataclass(frozen=True, slots=True)
class PricingCurrencyRejected:
    message: str
    status_code: Literal[400, 503]


class ReferenceRateLookup:
    """Answers from held ECB rates without waiting and refreshes them in one shared task when due. Only a lookup with
    no usable rates waits for that task, whose fetch ``fetch_ecb_daily_feed`` bounds. Callers that are cancelled
    while waiting leave the shared refresh running."""

    def __init__(
        self,
        rates: EcbReferenceRates,
        fetch: Callable[[], Awaitable[bytes | EcbFetchFailed]],
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._rates: Final = rates
        self._fetch: Final = fetch
        self._monotonic: Final = monotonic
        self._refresh: tuple[asyncio.AbstractEventLoop, asyncio.Task[EcbRefreshResult], float] | None = None

    async def usd_per_unit(self, currency: str) -> UsdRateLookup:
        held: Final = self._rates.usd_per_unit(currency)
        if not isinstance(held, EcbRatesUnavailable | EcbRatesStale):
            if currency != "USD" and self._rates.refresh_due():
                self._shared_refresh()
            return held
        await asyncio.shield(self._shared_refresh())
        return self._rates.usd_per_unit(currency)

    def _shared_refresh(self) -> "asyncio.Task[EcbRefreshResult]":
        loop: Final = asyncio.get_running_loop()
        running: Final = self._refresh
        if running is not None and running[0] is loop and self._reusable(running[1], running[2]):
            return running[1]
        task: Final = loop.create_task(self._refresh_once())
        self._refresh = (loop, task, self._monotonic())
        return task

    def _reusable(self, task: "asyncio.Task[EcbRefreshResult]", started_at: float) -> bool:
        if not task.done():
            return True
        if task.cancelled() or not isinstance(task.result(), EcbFetchFailed | EcbFeedRejected):
            return False
        return self._monotonic() - started_at < _FAILED_REFRESH_COOLDOWN_SECONDS

    async def _refresh_once(self) -> EcbRefreshResult:
        try:
            result: Final = await self._rates.arefresh(self._fetch)
        except Exception as e:  # noqa: BLE001  # a failed refresh keeps the held rates and must not fail a request
            verbose_proxy_logger.warning("ECB reference rate refresh raised %s", type(e).__name__)
            return EcbFetchFailed(type(e).__name__)
        if isinstance(result, EcbFetchFailed | EcbFeedRejected):
            verbose_proxy_logger.warning("ECB reference rate refresh failed: %s", result.reason)
        return result


async def _fetch_ecb_feed() -> bytes | EcbFetchFailed:
    return await fetch_ecb_daily_feed(get_async_httpx_client(llm_provider=httpxSpecialProvider.ModelCostMap))


reference_rates: Final = ReferenceRateLookup(
    EcbReferenceRates(clock=lambda: datetime.now(timezone.utc)), _fetch_ecb_feed
)


def _as_mapping(value: object) -> Mapping[str, object]:
    return _STR_MAPPING.validate_python(value) if isinstance(value, Mapping) else MappingProxyType({})


def _price(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        price: Final = float(value)
    except OverflowError:
        return None
    return price if math.isfinite(price) and price >= 0 else None


def _mixed_pricing_field(data: Mapping[str, object]) -> str | None:
    model_infos: Final = (
        _as_mapping(_as_mapping(data.get(key)).get("model_info")) for key in ("metadata", "litellm_metadata")
    )
    unconverted: Final = (
        *(field for field in _UNCONVERTED_PRICING_FIELDS if data.get(field) is not None),
        *(f"model_info.{field}" for field in chain.from_iterable(model_infos) if field in _PRICING_FIELDS),
    )
    return unconverted[0] if unconverted else None


def _validated_request(
    data: Mapping[str, object], route: str
) -> tuple[str, Mapping[str, float]] | PricingCurrencyRejected:
    currency: Final = data.get(PRICING_CURRENCY_FIELD)
    if route not in CHAT_COMPLETION_ROUTES:
        return PricingCurrencyRejected(f"{PRICING_CURRENCY_FIELD} is only supported on /v1/chat/completions", 400)
    if not isinstance(currency, str) or _CURRENCY_CODE.fullmatch(currency) is None:
        return PricingCurrencyRejected(f"{PRICING_CURRENCY_FIELD} must be a three-letter upper-case ISO 4217 code", 400)
    mixed: Final = _mixed_pricing_field(data)
    if mixed is not None:
        return PricingCurrencyRejected(f"{PRICING_CURRENCY_FIELD} cannot be combined with {mixed}", 400)
    supplied: Final = tuple(field for field in _CONVERTED_PRICE_FIELDS if data.get(field) is not None)
    if not supplied:
        return PricingCurrencyRejected(f"{PRICING_CURRENCY_FIELD} requires {' or '.join(_CONVERTED_PRICE_FIELDS)}", 400)
    prices: Final = MappingProxyType({field: _price(data[field]) for field in supplied})
    invalid: Final = next((field for field, price in prices.items() if price is None), None)
    if invalid is not None:
        return PricingCurrencyRejected(f"{invalid} must be a finite non-negative number", 400)
    return currency, MappingProxyType({field: price for field, price in prices.items() if price is not None})


def _usd_rate(lookup: UsdRateLookup) -> UsdRate | PricingCurrencyRejected:
    match lookup:
        case UsdRate():
            return lookup
        case UnsupportedCurrency(currency=currency, rate_date=rate_date):
            return PricingCurrencyRejected(f"no ECB reference rate for {currency} on {rate_date}", 400)
        case EcbRatesStale(rate_date=rate_date):
            return PricingCurrencyRejected(f"ECB reference rates dated {rate_date} are too old to convert prices", 503)
        case EcbRatesUnavailable():
            return PricingCurrencyRejected("ECB reference rates are unavailable, retry later", 503)


async def usd_request_prices(
    data: Mapping[str, object], route: str, rates: ReferenceRateLookup
) -> Mapping[str, float] | PricingCurrencyRejected:
    """The supplied ``input_cost_per_token``/``output_cost_per_token`` in USD. A direction the request did not price
    is left out, so it keeps the deployment's USD price."""
    validated: Final = _validated_request(data, route)
    if isinstance(validated, PricingCurrencyRejected):
        return validated
    currency, prices = validated
    rate: Final = _usd_rate(await rates.usd_per_unit(currency))
    if isinstance(rate, PricingCurrencyRejected):
        return rate
    converted: Final = MappingProxyType({field: price * rate.usd_per_unit for field, price in prices.items()})
    out_of_range: Final = next(
        (field for field, usd in converted.items() if not math.isfinite(usd) or (usd == 0 and prices[field] != 0)),
        None,
    )
    if out_of_range is not None:
        return PricingCurrencyRejected(f"{out_of_range} is out of range after conversion to USD", 400)
    return converted
