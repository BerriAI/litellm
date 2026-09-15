"""Roll closed UTC days of ``LiteLLM_DailyUserSpend`` up into ``LiteLLM_DailyGlobalSpend``.

Only days that are over get rolled up, so a pod still flushing per-key spend for the current
day can never leave the global table short; usage reads serve days through the recorded
marker from the global table and later days live from the per-key table. The marker lives in
``LiteLLM_Config``. This runs as a background cron, never in a Prisma migration, since on a
large deployment the first backfill is minutes of work.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    DAILY_GLOBAL_SPEND_RECONCILE_JOB_ID,
    DAILY_GLOBAL_SPEND_RECONCILE_LOCK_TTL_SECONDS,
    DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM,
)
from litellm.repositories.config_repository import ConfigRepository

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
    from litellm.proxy.utils import PrismaClient

_REPLAY_DAYS: Final = 1
GLOBAL_SPEND_TABLE_NAME: Final = "LiteLLM_DailyGlobalSpend"
# The unique constraint, in constraint order. NULL never matches itself in a unique index, so
# every column is normalized to '' or the same group would be inserted again on every run.
_KEY_COLUMNS: Final = ("date", "model", "model_group", "custom_llm_provider", "mcp_namespaced_tool_name", "endpoint")
_METRIC_COLUMNS: Final = (
    "prompt_tokens",
    "completion_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
    "compression_saved_tokens",
    "api_requests",
    "successful_requests",
    "failed_requests",
    "compression_savings_spend",
    "prompt_caching_savings_spend",
    "gateway_injected_caching_savings_spend",
    "autorouter_savings_spend",
    "spend",
)


def _quoted(columns: tuple[str, ...]) -> str:
    return ", ".join(f'"{column}"' for column in columns)


def _reconcile_day_sql() -> str:
    normalized_keys: Final = ", ".join(f"COALESCE(\"{column}\", '')" for column in _KEY_COLUMNS)
    sums: Final = ", ".join(f'SUM("{column}")' for column in _METRIC_COLUMNS)
    overwrite: Final = ", ".join(f'"{column}" = EXCLUDED."{column}"' for column in _METRIC_COLUMNS)
    return (
        f'INSERT INTO "{GLOBAL_SPEND_TABLE_NAME}" ("id", {_quoted(_KEY_COLUMNS)}, {_quoted(_METRIC_COLUMNS)}, '
        '"updated_at")\n'
        f"SELECT gen_random_uuid()::text, {normalized_keys}, {sums}, (NOW() AT TIME ZONE 'UTC')\n"
        'FROM "LiteLLM_DailyUserSpend" WHERE "date" = $1\n'
        f"GROUP BY {normalized_keys}\n"
        f"ON CONFLICT ({_quoted(_KEY_COLUMNS)}) DO UPDATE SET {overwrite}, "
        "\"updated_at\" = (NOW() AT TIME ZONE 'UTC')"
    )


RECONCILE_DAY_SQL: Final = _reconcile_day_sql()
_PENDING_DAYS_SQL: Final = (
    'SELECT DISTINCT "date" FROM "LiteLLM_DailyUserSpend" WHERE "date" >= $1 AND "date" <= $2 ORDER BY "date"'
)


class ReconciledThrough(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    reconciled_through: str


class _MarkerRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    param_value: object = None


class _DateRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    date: str


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    days_reconciled: tuple[str, ...]
    reconciled_through: str | None
    failed_day: str | None = None


def _marker_from_param_value(value: object) -> str | None:
    try:
        parsed: Final = (
            ReconciledThrough.model_validate_json(value)
            if isinstance(value, str)
            else ReconciledThrough.model_validate(value)
        )
    except ValidationError:
        return None
    return parsed.reconciled_through


async def reconciled_through(prisma_client: "PrismaClient") -> str | None:
    """The last UTC day ``LiteLLM_DailyGlobalSpend`` is known to cover, or None before the first run."""
    from litellm.proxy.utils import get_config_param

    row: Final = await get_config_param(prisma_client, DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM)
    return None if row is None else _marker_from_param_value(_MarkerRow.model_validate(row).param_value)


async def _record_reconciled_through(prisma_client: "PrismaClient", day: str) -> None:
    from litellm.proxy.utils import invalidate_config_param

    await ConfigRepository(prisma_client).set_param(
        DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM, ReconciledThrough(reconciled_through=day).model_dump_json()
    )
    await invalidate_config_param(DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM)


def _first_pending_day(marker: str | None) -> str:
    if marker is None:
        return ""
    return (date.fromisoformat(marker) - timedelta(days=_REPLAY_DAYS)).isoformat()


async def pending_days(prisma_client: "PrismaClient", today: date) -> tuple[str, ...]:
    """Every closed UTC day (strictly before today) still to roll up, oldest first. The marker
    day and the one before it are replayed so per-key rows that landed after their day was
    rolled up (a flush straddling midnight, a late retry) are folded in."""
    marker: Final = await reconciled_through(prisma_client)
    last_closed_day: Final = (today - timedelta(days=1)).isoformat()
    rows: Final = await prisma_client.db.query_raw(_PENDING_DAYS_SQL, _first_pending_day(marker), last_closed_day)
    return tuple(_DateRow.model_validate(row).date for row in rows)


async def reconcile_day(prisma_client: "PrismaClient", day: str) -> None:
    """Rewrite one day of the global table from the per-key sums. Idempotent: a rerun
    overwrites every group with the same totals."""
    await prisma_client.db.execute_raw(RECONCILE_DAY_SQL, day)


async def run_daily_global_spend_reconcile(
    prisma_client: "PrismaClient",
    today: date | None = None,
) -> ReconcileResult:
    """Roll up every pending day, advancing the marker after each; a failing day stops the run
    with the marker on the last good day so the next run resumes there."""
    effective_today: Final = today or datetime.now(timezone.utc).date()
    days: Final = await pending_days(prisma_client, effective_today)
    done: Final = await _reconcile_until_failure(prisma_client, days)
    failed: Final = days[len(done)] if len(done) < len(days) else None
    marker: Final = done[-1] if done else await reconciled_through(prisma_client)
    return ReconcileResult(days_reconciled=done, reconciled_through=marker, failed_day=failed)


async def _reconcile_until_failure(prisma_client: "PrismaClient", days: tuple[str, ...]) -> tuple[str, ...]:
    for index, day in enumerate(days):
        if not await _reconcile_and_record(prisma_client, day):
            return days[:index]
    return days


async def _reconcile_and_record(prisma_client: "PrismaClient", day: str) -> bool:
    try:
        await reconcile_day(prisma_client, day)
        await _record_reconciled_through(prisma_client, day)
    except Exception as exc:  # noqa: BLE001  # one bad day must not lose the days already done
        verbose_proxy_logger.exception("Daily global spend reconcile: day %s failed: %s", day, exc)
        return False
    return True


async def run_scheduled_daily_global_spend_reconcile(
    prisma_client: "PrismaClient",
    pod_lock_manager: "PodLockManager | None" = None,
    alert: Callable[[str], Awaitable[None]] | None = None,
    today: date | None = None,
) -> ReconcileResult | None:
    """Run the reconcile under a cross-pod lock so one proxy does the work; the lock only saves
    effort (each day is an idempotent rewrite), so an unreachable Redis runs unguarded rather than skipping."""
    redis_cache: Final = None if pod_lock_manager is None else pod_lock_manager.redis_cache
    if pod_lock_manager is None or redis_cache is None:
        return await _run_and_alert(prisma_client, alert=alert, today=today)

    acquired: Final = await pod_lock_manager.acquire_lock(
        cronjob_id=DAILY_GLOBAL_SPEND_RECONCILE_JOB_ID, ttl=DAILY_GLOBAL_SPEND_RECONCILE_LOCK_TTL_SECONDS
    )
    if not acquired and await _lock_is_held(pod_lock_manager, redis_cache):
        verbose_proxy_logger.info("Daily global spend reconcile: another pod holds the lock, skipping this run")
        return None
    try:
        return await _run_and_alert(prisma_client, alert=alert, today=today)
    finally:
        if acquired:
            await pod_lock_manager.release_lock(cronjob_id=DAILY_GLOBAL_SPEND_RECONCILE_JOB_ID)


async def _lock_is_held(pod_lock_manager: "PodLockManager", redis_cache: "RedisCache") -> bool:
    try:
        lock_key: Final = pod_lock_manager.get_redis_lock_key(DAILY_GLOBAL_SPEND_RECONCILE_JOB_ID)
        return bool(await redis_cache.async_get_cache(lock_key))
    except Exception as exc:  # noqa: BLE001  # an unreadable lock must not skip the run
        verbose_proxy_logger.warning("Daily global spend reconcile: could not read the lock: %s", exc)
        return False


async def _run_and_alert(
    prisma_client: "PrismaClient",
    *,
    alert: Callable[[str], Awaitable[None]] | None,
    today: date | None,
) -> ReconcileResult:
    result: Final = await run_daily_global_spend_reconcile(prisma_client, today=today)
    if result.days_reconciled:
        verbose_proxy_logger.info(
            "Daily global spend reconcile: rolled up %d day(s), reconciled through %s",
            len(result.days_reconciled),
            result.reconciled_through,
        )
    if result.failed_day is not None and alert is not None:
        await alert(
            f"Daily global spend reconcile stopped at {result.failed_day}; usage totals keep reading the per-key "
            f"table for ranges past {result.reconciled_through or 'the beginning'} until the next run succeeds."
        )
    return result
