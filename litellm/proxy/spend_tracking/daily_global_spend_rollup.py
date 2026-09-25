"""Roll closed UTC days of ``LiteLLM_DailyUserSpend`` up into ``LiteLLM_DailyGlobalSpend``.

Only days that are over get rolled up, so a pod still flushing per-key spend for the current
day can never leave the global table short; usage reads serve days through the recorded
marker from the global table and later days live from the per-key table. Per-key rows are
dated by request start, so spend can land on a day that was already rolled up (a flush
straddling midnight, a retry after an outage). Each run therefore also rewrites every closed
day that has rows touched since the previous run's scan, whatever the date. The marker lives
in ``LiteLLM_Config``. This runs as a background cron, never in a Prisma migration, since on
a large deployment the first backfill is minutes of work.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    DAILY_GLOBAL_SPEND_RECONCILE_JOB_ID,
    DAILY_GLOBAL_SPEND_RECONCILE_LOCK_TTL_SECONDS,
    DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM,
)

if TYPE_CHECKING:
    from litellm.caching.redis_cache import RedisCache
    from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
    from litellm.proxy.utils import PrismaClient

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
    "total_response_time_ms",
    "timed_requests",
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
_DB_NOW_SQL: Final = "SELECT (NOW() AT TIME ZONE 'UTC')::text AS now, (NOW() AT TIME ZONE 'UTC')::date::text AS today"
_ALL_CLOSED_DAYS_SQL: Final = 'SELECT DISTINCT "date" FROM "LiteLLM_DailyUserSpend" WHERE "date" <= $1 ORDER BY "date"'
# Pod clocks drift from the database clock and from each other, so rows are picked up from a
# little before the previous scan; rewriting a day twice is idempotent.
_PENDING_DAYS_SQL: Final = (
    'SELECT DISTINCT "date" FROM "LiteLLM_DailyUserSpend" WHERE "date" <= $1 '
    'AND ("date" > $2 OR "updated_at" >= $3::timestamp - INTERVAL \'1 hour\') '
    'ORDER BY "date"'
)
# Runs can overlap (Redis unreachable, lock expired on a long backfill), so the database keeps the
# later of the stored and the incoming day and scan time in one statement; GREATEST skips NULL.
_ADVANCE_MARKER_SQL: Final = (
    'INSERT INTO "LiteLLM_Config" ("param_name", "param_value") '
    "VALUES ($1, jsonb_build_object('reconciled_through', $2::text, 'scanned_at', $3::text)) "
    'ON CONFLICT ("param_name") DO UPDATE SET "param_value" = jsonb_build_object('
    "'reconciled_through', GREATEST(\"LiteLLM_Config\".\"param_value\" ->> 'reconciled_through', "
    "EXCLUDED.\"param_value\" ->> 'reconciled_through'), "
    "'scanned_at', GREATEST(\"LiteLLM_Config\".\"param_value\" ->> 'scanned_at', "
    "EXCLUDED.\"param_value\" ->> 'scanned_at'))"
)


class ReconciledThrough(BaseModel):
    """``reconciled_through`` is the last closed UTC day the global table covers. ``scanned_at`` is
    the database clock when the scan behind the last fully successful run started: every per-key
    row written before it, on any day through the marker, is in the global table."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    reconciled_through: str
    scanned_at: str | None = None


class _MarkerRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    param_value: object = None


class _DateRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    date: str


class _NowRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    now: str
    today: str


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    days_reconciled: tuple[str, ...]
    reconciled_through: str | None
    failed_day: str | None = None


@dataclass(frozen=True, slots=True)
class _PendingScan:
    marker: ReconciledThrough | None
    scanned_at: str
    days: tuple[str, ...]


def _marker_from_param_value(value: object) -> ReconciledThrough | None:
    try:
        return (
            ReconciledThrough.model_validate_json(value)
            if isinstance(value, str)
            else ReconciledThrough.model_validate(value)
        )
    except ValidationError:
        return None


async def read_marker(prisma_client: "PrismaClient") -> ReconciledThrough | None:
    from litellm.proxy.utils import get_config_param

    row: Final = await get_config_param(prisma_client, DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM)
    return None if row is None else _marker_from_param_value(_MarkerRow.model_validate(row).param_value)


async def reconciled_through(prisma_client: "PrismaClient") -> str | None:
    """The last UTC day ``LiteLLM_DailyGlobalSpend`` is known to cover, or None before the first run."""
    marker: Final = await read_marker(prisma_client)
    return None if marker is None else marker.reconciled_through


async def _advance_marker(prisma_client: "PrismaClient", days: tuple[str, ...], *, scanned_at: str | None) -> None:
    """Move the stored marker to the last of ``days`` and to ``scanned_at`` where those are later
    than what is stored, so a slower overlapping run can only add to a faster run's marker."""
    from litellm.proxy.utils import invalidate_config_param

    await prisma_client.db.execute_raw(
        _ADVANCE_MARKER_SQL,
        DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM,
        max(days) if days else None,
        scanned_at,
    )
    await invalidate_config_param(DAILY_GLOBAL_SPEND_RECONCILED_THROUGH_PARAM)


async def _db_now(prisma_client: "PrismaClient") -> _NowRow:
    rows: Final = await prisma_client.replica_db.query_raw(_DB_NOW_SQL)
    return _NowRow.model_validate(rows[0])


async def _scan_pending(prisma_client: "PrismaClient") -> _PendingScan:
    """Every closed UTC day (strictly before the database's today) still to roll up, oldest first:
    days past the marker, plus any day with per-key rows written since the scan behind the marker.
    Before a run has fully succeeded there is no such scan, so every closed day is rolled up."""
    marker: Final = await read_marker(prisma_client)
    db_now: Final = await _db_now(prisma_client)
    last_closed_day: Final = (date.fromisoformat(db_now.today) - timedelta(days=1)).isoformat()
    rows: Final = (
        await prisma_client.replica_db.query_raw(_ALL_CLOSED_DAYS_SQL, last_closed_day)
        if marker is None or marker.scanned_at is None
        else await prisma_client.replica_db.query_raw(
            _PENDING_DAYS_SQL, last_closed_day, marker.reconciled_through, marker.scanned_at
        )
    )
    return _PendingScan(marker, db_now.now, tuple(_DateRow.model_validate(row).date for row in rows))


async def reconcile_day(prisma_client: "PrismaClient", day: str) -> None:
    """Rewrite one day of the global table from the per-key sums. Idempotent: a rerun
    overwrites every group with the same totals."""
    await prisma_client.db.execute_raw(RECONCILE_DAY_SQL, day)


async def run_daily_global_spend_reconcile(prisma_client: "PrismaClient") -> ReconcileResult:
    """Roll up every pending day, advancing the marker after each; a failing day stops the run
    with the marker on the last good day so the next run resumes there. The scan time is only
    recorded once every pending day is done, so late rows a failed run saw are found again."""
    scan: Final = await _scan_pending(prisma_client)
    done: Final = await _reconcile_until_failure(prisma_client, scan)
    if len(done) < len(scan.days):
        marker: Final = await reconciled_through(prisma_client)
        return ReconcileResult(days_reconciled=done, reconciled_through=marker, failed_day=scan.days[len(done)])
    if scan.marker is not None or done:
        await _advance_marker(prisma_client, done, scanned_at=scan.scanned_at)
    return ReconcileResult(days_reconciled=done, reconciled_through=await reconciled_through(prisma_client))


async def _reconcile_until_failure(prisma_client: "PrismaClient", scan: _PendingScan) -> tuple[str, ...]:
    for index, day in enumerate(scan.days):
        if not await _reconcile_and_record(prisma_client, scan.days[: index + 1]):
            return scan.days[:index]
    return scan.days


async def _reconcile_and_record(prisma_client: "PrismaClient", done_with_this: tuple[str, ...]) -> bool:
    day: Final = done_with_this[-1]
    try:
        await reconcile_day(prisma_client, day)
        await _advance_marker(prisma_client, done_with_this, scanned_at=None)
    except Exception as exc:  # noqa: BLE001  # one bad day must not lose the days already done
        verbose_proxy_logger.exception("Daily global spend reconcile: day %s failed: %s", day, exc)
        return False
    return True


async def run_scheduled_daily_global_spend_reconcile(
    prisma_client: "PrismaClient",
    pod_lock_manager: "PodLockManager | None" = None,
    alert: Callable[[str], Awaitable[None]] | None = None,
) -> ReconcileResult | None:
    """Run the reconcile under a cross-pod lock so one proxy does the work; the lock only saves
    effort (each day is an idempotent rewrite), so an unreachable Redis runs unguarded rather than skipping."""
    redis_cache: Final = None if pod_lock_manager is None else pod_lock_manager.redis_cache
    if pod_lock_manager is None or redis_cache is None:
        return await _run_and_alert(prisma_client, alert=alert)

    acquired: Final = await pod_lock_manager.acquire_lock(
        cronjob_id=DAILY_GLOBAL_SPEND_RECONCILE_JOB_ID, ttl=DAILY_GLOBAL_SPEND_RECONCILE_LOCK_TTL_SECONDS
    )
    if not acquired and await _lock_is_held(pod_lock_manager, redis_cache):
        verbose_proxy_logger.info("Daily global spend reconcile: another pod holds the lock, skipping this run")
        return None
    try:
        return await _run_and_alert(prisma_client, alert=alert)
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
) -> ReconcileResult:
    result: Final = await run_daily_global_spend_reconcile(prisma_client)
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
