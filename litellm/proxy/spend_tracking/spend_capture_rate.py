"""Compare the spend LiteLLM captured for a provider against what that provider billed for the same UTC days.

LiteLLM's side is ``LiteLLM_DailyUserSpend``, summed over the ``custom_llm_provider`` values that land on the
provider's bill. The provider's side is its billing API, read with the customer's own billing credential
(OpenAI: the organization costs endpoint and an admin key in ``OPENAI_ADMIN_KEY``).
"""

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeAlias

from pydantic import BaseModel, ConfigDict, TypeAdapter
from typing_extensions import assert_never

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    SPEND_CAPTURE_RATE_CHECK_JOB_ID,
    SPEND_CAPTURE_RATE_CHECK_LOCK_TTL_SECONDS,
    SPEND_CAPTURE_RATE_DOCS_URL,
)
from litellm.llms.openai.organization_costs import (
    OPENAI_ADMIN_KEY_ENV_VAR,
    BillingHttpGet,
    OpenAICostsRequestFailed,
    fetch_openai_daily_costs,
    provider_billing_get,
)
from litellm.secret_managers.main import get_secret_str
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

OPENAI_BILLED_LITELLM_PROVIDERS: Final = ("openai", "text-completion-openai")

CaptureRatePublisher: TypeAlias = Callable[[SpendCaptureProvider, float | None], None]  # mutable-ok: Callable params

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
CheckResult: TypeAlias = CaptureRateReport | ProviderBillingFailure


class _CapturedSpendRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    date: str
    spend: float


_CAPTURED_SPEND_ROWS: Final = TypeAdapter(tuple[_CapturedSpendRow, ...])


async def captured_spend_by_day(
    prisma_client: "PrismaClient",
    *,
    litellm_providers: Sequence[str],
    start_date: date,
    end_date: date,
) -> Mapping[str, float]:
    """LiteLLM's tracked spend per UTC day (ISO date) for the given ``custom_llm_provider`` values."""
    rows: Final = await prisma_client.replica_db.query_raw(
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
    http_get: BillingHttpGet = provider_billing_get,
) -> CheckResult:
    match provider:
        case "openai":
            admin_key: Final = get_secret_str(OPENAI_ADMIN_KEY_ENV_VAR)
            if admin_key is None:
                return ProviderBillingCredentialMissing(provider, OPENAI_ADMIN_KEY_ENV_VAR)
            billed: Final = await fetch_openai_daily_costs(
                start_date, end_date, admin_key=admin_key, project_ids=openai_project_ids, http_get=http_get
            )
            if isinstance(billed, OpenAICostsRequestFailed):
                return ProviderBillingRequestFailed(provider, billed.detail)
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


def alert_message(result: CheckResult) -> str | None:
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


def _published_rate(result: CheckResult) -> float | None:
    """The gauge value: the rate, or ``None`` (NaN on the gauge) when this window produced no rate."""
    return result.capture_rate if isinstance(result, CaptureRateReport) else None


async def _check_every_provider(
    prisma_client: "PrismaClient",
    settings: SpendCaptureRateCheckSettings,
    *,
    publish: CaptureRatePublisher,
    today: date | None,
    http_get: BillingHttpGet,
) -> tuple[CheckResult, ...]:
    """Check every configured provider over the closed days before ``today`` and publish each outcome."""
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
        publish(result.provider, _published_rate(result))
        verbose_proxy_logger.info("Spend capture-rate check: %s", result)
    return results


def _alert_messages(results: Sequence[CheckResult]) -> tuple[str, ...]:
    return tuple(message for message in map(alert_message, results) if message is not None)


async def run_spend_capture_rate_check(
    prisma_client: "PrismaClient",
    settings: SpendCaptureRateCheckSettings,
    *,
    alert: Callable[[str], Awaitable[None]],
    publish: CaptureRatePublisher,
    today: date | None = None,
    http_get: BillingHttpGet = provider_billing_get,
) -> tuple[CheckResult, ...]:
    """Check every configured provider, publish each rate, and alert on every outcome that warrants one."""
    results: Final = await _check_every_provider(
        prisma_client, settings, publish=publish, today=today, http_get=http_get
    )
    for message in _alert_messages(results):
        await alert(message)
    return results


async def run_scheduled_spend_capture_rate_check(
    prisma_client: "PrismaClient",
    settings: SpendCaptureRateCheckSettings,
    *,
    pod_lock_manager: "PodLockManager | None",
    alert: Callable[[str], Awaitable[None]],
    publish: CaptureRatePublisher,
    today: date | None = None,
    http_get: BillingHttpGet = provider_billing_get,
) -> tuple[CheckResult, ...]:
    """Every worker publishes its own gauge; the first replica whose finished check has an alert claims the window."""
    results: Final = await _check_every_provider(
        prisma_client, settings, publish=publish, today=today, http_get=http_get
    )
    messages: Final = _alert_messages(results)
    if not messages:
        return results
    if not await _claims_alert_window(pod_lock_manager):
        verbose_proxy_logger.info("Spend capture-rate check: another pod alerted this window")
        return results
    for message in messages:
        await alert(message)
    return results


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
    except Exception as exc:  # noqa: BLE001  # an unreadable lock must not silence the alert
        verbose_proxy_logger.warning("Spend capture-rate check: could not read the lock: %s", exc)
        return False
