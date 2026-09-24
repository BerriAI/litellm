"""Compare the spend LiteLLM captured for a provider against what that provider billed for the same UTC days.

LiteLLM's side is ``LiteLLM_DailyUserSpend``, summed over the ``custom_llm_provider`` values that land on the
provider's bill. The provider's side is its billing API, read with the customer's own billing credential
(OpenAI: the organization costs endpoint and an admin key in ``OPENAI_ADMIN_KEY``).
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, assert_never

import httpx
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    OPENAI_ORGANIZATION_COSTS_PAGE_LIMIT,
    OPENAI_ORGANIZATION_COSTS_URL,
    PROVIDER_BILLING_TIMEOUT_SECONDS,
    SPEND_CAPTURE_RATE_CHECK_JOB_ID,
    SPEND_CAPTURE_RATE_CHECK_LOCK_TTL_SECONDS,
    SPEND_CAPTURE_RATE_DOCS_URL,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.proxy.spend_capture_rate import (
    CaptureRateDay,
    CaptureRateReport,
    SpendCaptureProvider,
    SpendCaptureRateCheckSettings,
)

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
    from litellm.proxy.utils import PrismaClient

OPENAI_ADMIN_KEY_ENV_VAR: Final = "OPENAI_ADMIN_KEY"
OPENAI_BILLED_LITELLM_PROVIDERS: Final = ("openai", "text-completion-openai")

BillingHttpGet: TypeAlias = Callable[
    [str, Mapping[str, object], Mapping[str, str]],  # mutable-ok: Callable parameter list is type syntax
    Awaitable[httpx.Response],
]
CaptureRatePublisher: TypeAlias = Callable[[SpendCaptureProvider, float], None]  # mutable-ok: Callable params

_CAPTURED_SPEND_BY_DAY_SQL: Final = """
    SELECT date, COALESCE(SUM(spend), 0)::float AS spend
    FROM "LiteLLM_DailyUserSpend"
    WHERE date >= $1 AND date <= $2 AND custom_llm_provider = ANY($3::text[])
    GROUP BY date
"""


@dataclass(frozen=True, slots=True)
class ProviderBillingCredentialMissing:
    provider: SpendCaptureProvider
    env_var: str


@dataclass(frozen=True, slots=True)
class ProviderBillingRequestFailed:
    provider: SpendCaptureProvider
    detail: str


ProviderBillingFailure: TypeAlias = ProviderBillingCredentialMissing | ProviderBillingRequestFailed


class _CapturedSpendRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    date: str
    spend: float


_CAPTURED_SPEND_ROWS: Final = TypeAdapter(tuple[_CapturedSpendRow, ...])


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


async def _provider_billing_get(url: str, params: Mapping[str, object], headers: Mapping[str, str]) -> httpx.Response:
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
    http_get: BillingHttpGet = _provider_billing_get,
) -> Mapping[str, float] | ProviderBillingRequestFailed:
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

    async def fetch_from(page: str | None) -> tuple[_OpenAICostBucket, ...] | ProviderBillingRequestFailed:
        params: Final[Mapping[str, object]] = MappingProxyType(
            {key: value for key, value in (*window.items(), ("page", page)) if value is not None}
        )
        try:
            response: Final = await http_get(OPENAI_ORGANIZATION_COSTS_URL, params, headers)
        except httpx.HTTPError as exc:
            return ProviderBillingRequestFailed("openai", f"request failed: {exc}")
        if response.status_code != 200:
            return ProviderBillingRequestFailed("openai", f"HTTP {response.status_code}: {response.text[:300]}")
        try:
            parsed: Final = _OpenAICostsPage.model_validate(response.json())
        except (ValueError, ValidationError) as exc:
            return ProviderBillingRequestFailed("openai", f"unexpected response shape: {exc}")
        if not parsed.has_more or parsed.next_page is None:
            return parsed.data
        rest: Final = await fetch_from(parsed.next_page)
        return rest if isinstance(rest, ProviderBillingRequestFailed) else parsed.data + rest

    buckets: Final = await fetch_from(None)
    if isinstance(buckets, ProviderBillingRequestFailed):
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


async def captured_spend_by_day(
    prisma_client: "PrismaClient",
    *,
    litellm_providers: Sequence[str],
    start_date: date,
    end_date: date,
) -> Mapping[str, float]:
    """LiteLLM's tracked spend per UTC day (ISO date) for the given ``custom_llm_provider`` values."""
    rows: Final = await prisma_client.db.query_raw(
        _CAPTURED_SPEND_BY_DAY_SQL, start_date.isoformat(), end_date.isoformat(), tuple(litellm_providers)
    )
    return MappingProxyType({row.date: row.spend for row in _CAPTURED_SPEND_ROWS.validate_python(rows)})


def _ratio(captured: float, billed: float) -> float | None:
    return None if billed <= 0 else captured / billed


def _days(start_date: date, end_date: date) -> tuple[date, ...]:
    return tuple(start_date + timedelta(days=offset) for offset in range((end_date - start_date).days + 1))


def compute_capture_rate(
    *,
    provider: SpendCaptureProvider,
    start_date: date,
    end_date: date,
    captured_by_day: Mapping[str, float],
    billed_by_day: Mapping[str, float],
    threshold: float,
) -> CaptureRateReport:
    days: Final = tuple(
        CaptureRateDay(
            date=day.isoformat(),
            captured_spend=captured_by_day.get(day.isoformat(), 0.0),
            provider_spend=billed_by_day.get(day.isoformat(), 0.0),
            capture_rate=_ratio(captured_by_day.get(day.isoformat(), 0.0), billed_by_day.get(day.isoformat(), 0.0)),
        )
        for day in _days(start_date, end_date)
    )
    captured: Final = sum(day.captured_spend for day in days)
    billed: Final = sum(day.provider_spend for day in days)
    rate: Final = _ratio(captured, billed)
    return CaptureRateReport(
        provider=provider,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        captured_spend=captured,
        provider_spend=billed,
        capture_rate=rate,
        threshold=threshold,
        below_threshold=rate is not None and rate < threshold,
        days=days,
    )


async def capture_rate_report(
    prisma_client: "PrismaClient",
    *,
    provider: SpendCaptureProvider,
    start_date: date,
    end_date: date,
    threshold: float,
    openai_project_ids: Sequence[str] = (),
    http_get: BillingHttpGet = _provider_billing_get,
) -> CaptureRateReport | ProviderBillingFailure:
    match provider:
        case "openai":
            admin_key: Final = get_secret_str(OPENAI_ADMIN_KEY_ENV_VAR)
            if admin_key is None:
                return ProviderBillingCredentialMissing(provider, OPENAI_ADMIN_KEY_ENV_VAR)
            billed: Final = await fetch_openai_daily_costs(
                start_date, end_date, admin_key=admin_key, project_ids=openai_project_ids, http_get=http_get
            )
            if isinstance(billed, ProviderBillingRequestFailed):
                return billed
            captured: Final = await captured_spend_by_day(
                prisma_client,
                litellm_providers=OPENAI_BILLED_LITELLM_PROVIDERS,
                start_date=start_date,
                end_date=end_date,
            )
            return compute_capture_rate(
                provider=provider,
                start_date=start_date,
                end_date=end_date,
                captured_by_day=captured,
                billed_by_day=billed,
                threshold=threshold,
            )
        case _:
            assert_never(provider)


def alert_message(result: CaptureRateReport | ProviderBillingFailure) -> str | None:
    """The alert a check outcome warrants, or ``None`` when the capture rate is healthy."""
    match result:
        case ProviderBillingCredentialMissing(provider=provider, env_var=env_var):
            return (
                f"Spend capture-rate check: {env_var} is not set, so the {provider} bill cannot be read. "
                f"Set it or remove general_settings.spend_capture_rate_check. {SPEND_CAPTURE_RATE_DOCS_URL}"
            )
        case ProviderBillingRequestFailed(provider=provider, detail=detail):
            return f"Spend capture-rate check: could not read the {provider} bill ({detail}). {SPEND_CAPTURE_RATE_DOCS_URL}"
        case CaptureRateReport():
            if not result.below_threshold or result.capture_rate is None:
                return None
            return (
                f"Spend capture rate for {result.provider} is {result.capture_rate:.1%}, under the "
                f"{result.threshold:.0%} threshold: LiteLLM captured ${result.captured_spend:,.2f} of the "
                f"${result.provider_spend:,.2f} {result.provider} bill for {result.start_date} to {result.end_date}. "
                f"Requests reach {result.provider} outside LiteLLM or cost tracking is dropping spend. "
                f"{SPEND_CAPTURE_RATE_DOCS_URL}"
            )
        case _:
            assert_never(result)


async def run_spend_capture_rate_check(
    prisma_client: "PrismaClient",
    settings: SpendCaptureRateCheckSettings,
    *,
    alert: Callable[[str], Awaitable[None]],
    publish: CaptureRatePublisher,
    today: date | None = None,
    http_get: BillingHttpGet = _provider_billing_get,
) -> tuple[CaptureRateReport | ProviderBillingFailure, ...]:
    """Check every configured provider over the closed days before ``today``, publish each rate, alert on the rest."""
    end_date: Final = (today or datetime.now(timezone.utc).date()) - timedelta(days=1)
    start_date: Final = end_date - timedelta(days=settings.lookback_days - 1)
    results: Final = tuple(
        [
            await capture_rate_report(
                prisma_client,
                provider=provider,
                start_date=start_date,
                end_date=end_date,
                threshold=settings.threshold,
                openai_project_ids=settings.openai_project_ids,
                http_get=http_get,
            )
            for provider in settings.providers
        ]
    )
    for result in results:
        await _publish_and_alert(result, alert=alert, publish=publish)
    return results


async def _publish_and_alert(
    result: CaptureRateReport | ProviderBillingFailure,
    *,
    alert: Callable[[str], Awaitable[None]],
    publish: CaptureRatePublisher,
) -> None:
    if isinstance(result, CaptureRateReport) and result.capture_rate is not None:
        publish(result.provider, result.capture_rate)
    message: Final = alert_message(result)
    if message is not None:
        await alert(message)
    verbose_proxy_logger.info("Spend capture-rate check: %s", result)


async def run_scheduled_spend_capture_rate_check(
    prisma_client: "PrismaClient",
    settings: SpendCaptureRateCheckSettings,
    *,
    pod_lock_manager: "PodLockManager | None",
    alert: Callable[[str], Awaitable[None]],
    publish: CaptureRatePublisher,
    today: date | None = None,
    http_get: BillingHttpGet = _provider_billing_get,
) -> tuple[CaptureRateReport | ProviderBillingFailure, ...]:
    """Every worker computes and publishes its own gauge; the cross-pod lock lets one replica per window alert."""
    alerts_here: Final = await _claims_alert_window(pod_lock_manager)
    if not alerts_here:
        verbose_proxy_logger.info("Spend capture-rate check: another pod alerts this window, publishing only")
    return await run_spend_capture_rate_check(
        prisma_client,
        settings,
        alert=alert if alerts_here else _drop_alert,
        publish=publish,
        today=today,
        http_get=http_get,
    )


async def _drop_alert(message: str) -> None:
    return None


async def _claims_alert_window(pod_lock_manager: "PodLockManager | None") -> bool:
    """The lock is left to expire, so every replica firing within its TTL of the winner stays quiet."""
    redis_cache: Final = None if pod_lock_manager is None else pod_lock_manager.redis_cache
    if pod_lock_manager is None or redis_cache is None:
        return True
    acquired: Final = await pod_lock_manager.acquire_lock(
        cronjob_id=SPEND_CAPTURE_RATE_CHECK_JOB_ID, ttl=SPEND_CAPTURE_RATE_CHECK_LOCK_TTL_SECONDS
    )
    return acquired or not await _lock_is_held(pod_lock_manager, redis_cache)


async def _lock_is_held(pod_lock_manager: "PodLockManager", redis_cache: "RedisCache") -> bool:
    try:
        return bool(
            await redis_cache.async_get_cache(pod_lock_manager.get_redis_lock_key(SPEND_CAPTURE_RATE_CHECK_JOB_ID))
        )
    except Exception as exc:  # noqa: BLE001  # an unreadable lock must not skip the check
        verbose_proxy_logger.warning("Spend capture-rate check: could not read the lock: %s", exc)
        return False
