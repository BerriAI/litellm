import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime, timezone
from typing import Annotated, Final, Literal, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ConfigDict

from litellm.litellm_core_utils.ecb_reference_rates import (
    EcbFeedRejected,
    EcbFetchFailed,
    EcbReferenceRates,
    EcbRefreshResult,
    fetch_ecb_daily_feed,
)
from litellm.llms.custom_httpx.http_handler import (
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory has an untyped params argument
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.llms.custom_http import httpxSpecialProvider

router: Final = APIRouter()

EcbFeedFetch: TypeAlias = Callable[[], Awaitable[bytes | EcbFetchFailed]]
_FAILED_REFRESH_COOLDOWN_SECONDS: Final = 60.0


class _FailedRefreshCooldown:
    def __init__(self, monotonic: Callable[[], float] = time.monotonic) -> None:
        self._monotonic: Final = monotonic
        self._failed_at: float | None = None

    def active(self) -> bool:
        failed_at: Final = self._failed_at
        return failed_at is not None and self._monotonic() - failed_at < _FAILED_REFRESH_COOLDOWN_SECONDS

    def record(self, result: EcbRefreshResult) -> None:
        self._failed_at = self._monotonic() if isinstance(result, EcbFetchFailed | EcbFeedRejected) else None


class ReferenceExchangeRates(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    source: Literal["ecb"]
    rate_date: date
    usd_per_unit: Mapping[str, float]


_reference_rates: Final = EcbReferenceRates(clock=lambda: datetime.now(timezone.utc))
_refresh_lock: Final = asyncio.Lock()
_refresh_cooldown: Final = _FailedRefreshCooldown()


def _default_rates() -> EcbReferenceRates:
    return _reference_rates


def _default_refresh_lock() -> asyncio.Lock:
    return _refresh_lock


def _default_refresh_cooldown() -> _FailedRefreshCooldown:
    return _refresh_cooldown


async def _fetch_on_serving_loop() -> bytes | EcbFetchFailed:
    return await fetch_ecb_daily_feed(get_async_httpx_client(llm_provider=httpxSpecialProvider.UI))


def _default_fetch() -> EcbFeedFetch:
    return _fetch_on_serving_loop


@router.get(
    "/get/reference_exchange_rates",
    tags=["Budget & Spend Tracking"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=ReferenceExchangeRates,
    responses={503: {"description": "No ECB reference rate published in the last 7 days is available"}},
)
async def reference_exchange_rates(
    rates: Annotated[EcbReferenceRates, Depends(_default_rates)],
    refresh_lock: Annotated[asyncio.Lock, Depends(_default_refresh_lock)],
    fetch: Annotated[EcbFeedFetch, Depends(_default_fetch)],
    cooldown: Annotated[_FailedRefreshCooldown, Depends(_default_refresh_cooldown)],
) -> ReferenceExchangeRates:
    """
    ECB euro foreign exchange reference rates as USD per unit of each currency, with USD at 1. They estimate a USD
    total in another currency for display and are not settlement quotes. Refreshes from the ECB feed when due.
    """
    if rates.refresh_due() and not cooldown.active():
        async with refresh_lock:
            if rates.refresh_due() and not cooldown.active():
                cooldown.record(await rates.arefresh(fetch))
    usd_rates: Final = rates.usd_rates()
    rate_date: Final = next((rate.rate_date for rate in usd_rates.values() if rate.rate_date is not None), None)
    if rate_date is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ECB reference rates unavailable")
    return ReferenceExchangeRates(
        source="ecb",
        rate_date=rate_date,
        usd_per_unit={"USD": 1.0, **{currency: rate.usd_per_unit for currency, rate in usd_rates.items()}},
    )
