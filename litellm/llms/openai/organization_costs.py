"""OpenAI's organization costs endpoint: the USD the organization was billed per UTC day, read with an admin key."""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.constants import (
    OPENAI_ORGANIZATION_COSTS_PAGE_LIMIT,
    OPENAI_ORGANIZATION_COSTS_URL,
    PROVIDER_BILLING_TIMEOUT_SECONDS,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

OPENAI_ADMIN_KEY_ENV_VAR: Final = "OPENAI_ADMIN_KEY"

BillingHttpGet: TypeAlias = Callable[
    [str, Mapping[str, object], Mapping[str, str]],  # mutable-ok: Callable parameter list is type syntax
    Awaitable[httpx.Response],
]


@dataclass(frozen=True, slots=True)
class OpenAICostsRequestFailed:
    detail: str


class _OpenAICostAmount(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    value: float
    currency: Literal["usd"]


class _OpenAICostResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    amount: _OpenAICostAmount


class _OpenAICostBucket(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    start_time: int
    results: tuple[_OpenAICostResult, ...] = ()


class _OpenAICostsPage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    data: tuple[_OpenAICostBucket, ...]
    has_more: bool = False
    next_page: str | None = None


async def provider_billing_get(url: str, params: Mapping[str, object], headers: Mapping[str, str]) -> httpx.Response:
    client: Final = get_async_httpx_client(llm_provider=httpxSpecialProvider.ProviderBilling)
    return await client.get(
        url,
        params=dict(params),  # mutable-ok: AsyncHTTPHandler.get takes dict params
        headers=dict(headers),  # mutable-ok: AsyncHTTPHandler.get takes dict headers
        timeout=PROVIDER_BILLING_TIMEOUT_SECONDS,
    )


def _utc_midnight(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp())


def _bucket_day(bucket: _OpenAICostBucket) -> str:
    return datetime.fromtimestamp(bucket.start_time, tz=timezone.utc).date().isoformat()


async def fetch_openai_daily_costs(
    start_date: date,
    end_date: date,
    *,
    admin_key: str,
    project_ids: Sequence[str] = (),
    http_get: BillingHttpGet = provider_billing_get,
) -> Mapping[str, float] | OpenAICostsRequestFailed:
    """USD billed by OpenAI per UTC day (ISO date) over the closed range, following pagination to the end."""
    scope: Final = (("project_ids[]", tuple(project_ids)),) if project_ids else ()
    window: Final[Mapping[str, object]] = MappingProxyType(
        {
            key: value
            for key, value in (
                ("start_time", _utc_midnight(start_date)),
                ("end_time", _utc_midnight(end_date + timedelta(days=1))),
                ("bucket_width", "1d"),
                ("limit", OPENAI_ORGANIZATION_COSTS_PAGE_LIMIT),
                *scope,
            )
        }
    )
    headers: Final[Mapping[str, str]] = MappingProxyType({"Authorization": f"Bearer {admin_key}"})

    async def fetch_from(page: str | None) -> tuple[_OpenAICostBucket, ...] | OpenAICostsRequestFailed:
        params: Final[Mapping[str, object]] = MappingProxyType(
            {key: value for key, value in (*window.items(), ("page", page)) if value is not None}
        )
        try:
            response: Final = await http_get(OPENAI_ORGANIZATION_COSTS_URL, params, headers)
        except httpx.HTTPError as exc:
            return OpenAICostsRequestFailed(f"request failed: {exc}")
        if response.status_code != 200:
            return OpenAICostsRequestFailed(f"HTTP {response.status_code}: {response.text[:300]}")
        try:
            parsed: Final = _OpenAICostsPage.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            return OpenAICostsRequestFailed(f"unexpected response shape: {exc}")
        if not parsed.has_more or parsed.next_page is None:
            return parsed.data
        rest: Final = await fetch_from(parsed.next_page)
        return rest if isinstance(rest, OpenAICostsRequestFailed) else parsed.data + rest

    buckets: Final = await fetch_from(None)
    if isinstance(buckets, OpenAICostsRequestFailed):
        return buckets
    days: Final = frozenset(_bucket_day(bucket) for bucket in buckets)
    return MappingProxyType(
        {
            day: sum(
                result.amount.value for bucket in buckets if _bucket_day(bucket) == day for result in bucket.results
            )
            for day in days
        }
    )
