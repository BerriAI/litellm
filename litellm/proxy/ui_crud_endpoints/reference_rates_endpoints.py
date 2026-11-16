import asyncio
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime, timezone
from typing import Annotated, Final, Literal, TypeAlias

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import ConfigDict

from litellm.litellm_core_utils.ecb_reference_rates import EcbFetchFailed, EcbReferenceRates, fetch_ecb_daily_feed
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.llms.base import LiteLLMBaseModel

router: Final = APIRouter()

EcbFeedFetch: TypeAlias = Callable[[], Awaitable[bytes | EcbFetchFailed]]


class ReferenceExchangeRates(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)

    source: Literal["ecb"]
    rate_date: date
    usd_per_unit: Mapping[str, float]


_reference_rates: Final = EcbReferenceRates(clock=lambda: datetime.now(timezone.utc))
_refresh_lock: Final = asyncio.Lock()


def _default_rates() -> EcbReferenceRates:
    return _reference_rates


def _default_refresh_lock() -> asyncio.Lock:
    return _refresh_lock


def _default_fetch() -> EcbFeedFetch:
    from litellm.llms.custom_httpx.http_handler import get_async_httpx_client  # noqa: PLC0415  # heavy, first use only
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415  # imported with the client

    client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.UI)

    async def fetch() -> bytes | EcbFetchFailed:
        return await fetch_ecb_daily_feed(client)

    return fetch


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
) -> ReferenceExchangeRates:
    """
    ECB euro foreign exchange reference rates as USD per unit of each currency, with USD at 1. They estimate a USD
    total in another currency for display and are not settlement quotes. Fetches the ECB feed at most once a day.
    """
    if rates.refresh_due():
        async with refresh_lock:
            if rates.refresh_due():
                await rates.arefresh(fetch)
    usd_rates: Final = rates.usd_rates()
    rate_date: Final = next((rate.rate_date for rate in usd_rates.values() if rate.rate_date is not None), None)
    if rate_date is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="ECB reference rates unavailable")
    return ReferenceExchangeRates(
        source="ecb",
        rate_date=rate_date,
        usd_per_unit={"USD": 1.0, **{currency: rate.usd_per_unit for currency, rate in usd_rates.items()}},
    )
