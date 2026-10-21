import asyncio
import math
import re
import time
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, Literal, TypeAlias

from pydantic import ConfigDict, TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import (
    SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS,
    SPEND_LOG_CLEANUP_BATCH_SIZE,
    SPEND_LOG_CLEANUP_BATCH_TIMEOUT_SECONDS,
    SPEND_LOG_CLEANUP_JOB_NAME,
    SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES,
    SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP,
    SPEND_LOG_CLEANUP_RUN_BUDGET_SECONDS,
    SPEND_LOG_RUN_LOOPS,
    SPEND_LOG_SIZE_CLEANUP_RECHECK_BATCHES,
)
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.db.db_span import db_span
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup_metrics import (
    RunOutcome,
    SpendLogCleanupMetrics,
)
from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import (
    RemainingTimeoutMs,
    SpendLogsPartitionManager,
    bounded_tx,
    trusted_sql,
)
from litellm.proxy.utils import PrismaClient
from litellm.types.llms.base import LiteLLMBaseModel

StopReason: TypeAlias = Literal["exhausted", "target_reached", "budget_exhausted", "batch_cap_reached", "aborted"]

_SIZE_RE: Final = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([KMGT]?)I?B?\s*$", re.IGNORECASE)
_SIZE_UNITS: Final = {"": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def size_in_bytes(value: str | int) -> int:
    """
    Parse a storage size such as '180GB', '512 MiB' or '1.5TB' into bytes; a bare int is bytes already.

    Units are powers of 1024, the same convention pg_size_pretty uses, so the number an operator reads
    off `SELECT pg_size_pretty(pg_total_relation_size(...))` is the number they can put in the config.
    """
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"size must be positive, got {value}")
        return value
    match: Final = _SIZE_RE.match(value)
    if match is None:
        raise ValueError(f"invalid size {value!r}, expected a number with a unit such as '180GB'")
    size: Final = int(float(match.group(1)) * _SIZE_UNITS[match.group(2).upper()])
    if size <= 0:
        raise ValueError(f"size must be positive, got {value!r}")
    return size


@dataclass(frozen=True, slots=True)
class SizeRule:
    """The size cleanup rule as configured: start cleaning above `limit_bytes`, stop under `target_bytes`."""

    limit_bytes: int
    target_bytes: int


@dataclass(frozen=True, slots=True)
class StorageProbe:
    """
    What the catalog says about the spend-log tables, read without scanning them.

    `allocated_bytes` is what the files take on disk (heap, indexes and TOAST of LiteLLM_SpendLogs and
    LiteLLM_SpendLogToolIndex, partitions included). Deleting rows does not shrink it: deleted rows become dead
    tuples, autovacuum later turns them into free space that new rows reuse, and only VACUUM FULL or pg_repack
    hands the bytes back to the filesystem. Neither dead tuples nor free space are live data, and Postgres
    tracks only the former (`n_dead_tup`, zeroed by vacuum), so the size rule keeps its own count of the
    reclaimable bytes, see `SizeCleanupState`. `inserted_total` and `deleted_total` are the cumulative
    `n_tup_ins` and `n_tup_del` counters that count is driven by.
    """

    allocated_bytes: int
    live_rows: int
    dead_rows: int
    inserted_total: int
    deleted_total: int

    @property
    def stats_are_stale(self) -> bool:
        """No row counters at all while the files hold data: the statistics were reset, e.g. by a crash."""
        return self.allocated_bytes > 0 and self.live_rows + self.dead_rows == 0


class SizeCleanupState(LiteLLMBaseModel):
    """
    The size rule's running count of reclaimable bytes, kept in LiteLLM_Config between checks.

    `reclaimable_bytes` is the space inside the files that holds no live row: the bytes of every row deleted
    since the count started (by this job or anyone else, from `n_tup_del`), less the free space new rows have
    since been written into (bytes inserted beyond what the files grew by), less what a VACUUM FULL handed back.
    Live bytes are `allocated_bytes - reclaimable_bytes`, whether or not autovacuum has run yet. `bytes_per_row`
    converts rows to bytes and is refreshed from live bytes over live rows on every check.
    """

    model_config = ConfigDict(frozen=True)
    allocated_bytes: int
    reclaimable_bytes: int
    bytes_per_row: float
    inserted_total: int
    deleted_total: int


def advance_size_state(state: SizeCleanupState | None, probe: StorageProbe) -> SizeCleanupState:
    """
    Roll the reclaimable count forward to what the catalog shows now.

    Without a prior state the count starts from the dead rows Postgres still knows about, at the files' average
    bytes per row. After the cumulative counters went backwards (a statistics reset, e.g. a crash) the prior
    count is carried forward less any shrink of the files, or the dead-row seed if that is larger, because the
    reset changed the counters and not the files. A probe with no row counters at all (`stats_are_stale`)
    cannot seed anything: the prior count is carried forward, and with no prior count the files are taken as
    fully reclaimable so nothing is deleted on its say-so.
    """
    if probe.stats_are_stale:
        return SizeCleanupState(
            allocated_bytes=probe.allocated_bytes,
            reclaimable_bytes=probe.allocated_bytes
            if state is None
            else min(probe.allocated_bytes, state.reclaimable_bytes),
            bytes_per_row=1.0 if state is None else state.bytes_per_row,
            inserted_total=probe.inserted_total,
            deleted_total=probe.deleted_total,
        )
    if state is None or probe.inserted_total < state.inserted_total or probe.deleted_total < state.deleted_total:
        seed_bytes_per_row: Final = probe.allocated_bytes / max(probe.live_rows + probe.dead_rows, 1)
        seeded: Final = min(probe.allocated_bytes, round(probe.dead_rows * seed_bytes_per_row))
        if state is None:
            return SizeCleanupState(
                allocated_bytes=probe.allocated_bytes,
                reclaimable_bytes=seeded,
                bytes_per_row=max(seed_bytes_per_row, 1.0),
                inserted_total=probe.inserted_total,
                deleted_total=probe.deleted_total,
            )
        carried: Final = max(
            seeded,
            min(probe.allocated_bytes, state.reclaimable_bytes - max(0, state.allocated_bytes - probe.allocated_bytes)),
        )
        carried_live_bytes: Final = probe.allocated_bytes - carried
        return SizeCleanupState(
            allocated_bytes=probe.allocated_bytes,
            reclaimable_bytes=carried,
            bytes_per_row=max(carried_live_bytes / probe.live_rows, 1.0)
            if probe.live_rows > 0
            else state.bytes_per_row,
            inserted_total=probe.inserted_total,
            deleted_total=probe.deleted_total,
        )
    growth: Final = probe.allocated_bytes - state.allocated_bytes
    bytes_deleted: Final = (probe.deleted_total - state.deleted_total) * state.bytes_per_row
    free_space_refilled: Final = max(
        0.0, (probe.inserted_total - state.inserted_total) * state.bytes_per_row - max(0, growth)
    )
    returned_to_filesystem: Final = max(0, -growth)
    reclaimable: Final = min(
        probe.allocated_bytes,
        max(0, round(state.reclaimable_bytes + bytes_deleted - free_space_refilled - returned_to_filesystem)),
    )
    live_bytes: Final = probe.allocated_bytes - reclaimable
    return SizeCleanupState(
        allocated_bytes=probe.allocated_bytes,
        reclaimable_bytes=reclaimable,
        bytes_per_row=max(live_bytes / probe.live_rows, 1.0) if probe.live_rows > 0 else state.bytes_per_row,
        inserted_total=probe.inserted_total,
        deleted_total=probe.deleted_total,
    )


@dataclass(frozen=True, slots=True)
class SpendLogStorage:
    """What the spend-log tables weigh at one point: the files on disk and the live bytes inside them."""

    allocated_bytes: int
    reclaimable_bytes: int
    bytes_per_row: float
    live_rows: int

    @property
    def estimated_live_bytes(self) -> int:
        return max(0, self.allocated_bytes - self.reclaimable_bytes)

    def after_deleting(self, rows: int, allocated_bytes: int) -> "SpendLogStorage":
        """
        The same storage once `rows` more rows are gone and the files measure `allocated_bytes`.

        Files that grew while the pass only deleted (a visibility map or free space map fork a DELETE creates,
        an index page split) hold no live row, so the growth counts as reclaimable; rows inserted meanwhile are
        taken back out at the next check from n_tup_ins.
        """
        return SpendLogStorage(
            allocated_bytes=allocated_bytes,
            reclaimable_bytes=min(
                allocated_bytes,
                round(
                    self.reclaimable_bytes + rows * self.bytes_per_row + max(0, allocated_bytes - self.allocated_bytes)
                ),
            ),
            bytes_per_row=self.bytes_per_row,
            live_rows=max(0, self.live_rows - rows),
        )

    def to_state(self, inserted_total: int, deleted_total: int) -> SizeCleanupState:
        return SizeCleanupState(
            allocated_bytes=self.allocated_bytes,
            reclaimable_bytes=self.reclaimable_bytes,
            bytes_per_row=self.bytes_per_row,
            inserted_total=inserted_total,
            deleted_total=deleted_total,
        )


Cutoff: TypeAlias = datetime | str
"""Rows strictly older than this are expired: a timestamp, or an ISO calendar day for tables keyed by day"""


def _cutoff_cast(cutoff: Cutoff) -> str:
    return "timestamptz" if isinstance(cutoff, datetime) else "text"


def _cutoff_text(cutoff: Cutoff) -> str:
    return cutoff.isoformat() if isinstance(cutoff, datetime) else cutoff


@dataclass(frozen=True, slots=True)
class TableCleanupResult:
    """Outcome of pruning one table, so the caller can report why a run ended."""

    table_name: str
    rows_deleted: int
    stop_reason: StopReason


@dataclass(frozen=True, slots=True)
class _SizeStepState:
    """Progress of the size pass across its re-check steps."""

    logs_deleted: int
    index_deleted: int
    batches_left: int
    stop_reason: StopReason | None

    def stopped(self, reason: StopReason) -> "_SizeStepState":
        return _SizeStepState(
            logs_deleted=self.logs_deleted,
            index_deleted=self.index_deleted,
            batches_left=self.batches_left,
            stop_reason=reason,
        )


@dataclass(frozen=True, slots=True)
class SizePassReport:
    """What one size pass measured and achieved, for the run summary and the storage gauges."""

    rule: SizeRule
    before: SpendLogStorage
    after: SpendLogStorage
    partitions_dropped: tuple[str, ...]
    results: tuple[TableCleanupResult, ...]

    @property
    def target_reached(self) -> bool:
        return self.after.estimated_live_bytes <= self.rule.target_bytes

    @property
    def above_target_bytes(self) -> int:
        return max(0, self.after.estimated_live_bytes - self.rule.target_bytes)


class _RunProgress:
    """How far one cleanup run has got, reported if that run is cancelled"""

    def __init__(self) -> None:
        self.rows_deleted: int = 0
        self.batches: int = 0

    def record_batch(self, rows_deleted: int) -> None:
        self.rows_deleted += rows_deleted
        self.batches += 1


_run_progress: ContextVar[_RunProgress] = ContextVar("spend_log_cleanup_run_progress")


def _record_run_batch(rows_deleted: int) -> None:
    """Count a batch towards the run in progress, if a run is what issued it"""
    progress: Final = _run_progress.get(None)
    if progress is not None:
        progress.record_batch(rows_deleted)


class _RemainingRow(LiteLLMBaseModel):
    """One row of the capped outstanding-rows probe, validated out of prisma's untyped result."""

    remaining: int


_REMAINING_ROWS: Final = TypeAdapter(list[_RemainingRow])


class _StorageRow(LiteLLMBaseModel):
    """One row of the storage probe, validated out of prisma's untyped result."""

    model_config = ConfigDict(frozen=True)
    allocated_bytes: int
    live_rows: int
    dead_rows: int
    inserted_total: int
    deleted_total: int


_STORAGE_ROWS: Final = TypeAdapter(list[_StorageRow])


class _SizeStateRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    state_json: str


_SIZE_STATE_ROWS: Final = TypeAdapter(list[_SizeStateRow])
SPEND_LOG_SIZE_CLEANUP_STATE_PARAM: Final = "spend_logs_size_cleanup_state"
_LOAD_SIZE_STATE_SQL: Final = 'SELECT param_value::text AS state_json FROM "LiteLLM_Config" WHERE param_name = $1'
_SAVE_SIZE_STATE_SQL: Final = """
    INSERT INTO "LiteLLM_Config" (param_name, param_value) VALUES ($1, $2::jsonb)
    ON CONFLICT (param_name) DO UPDATE SET param_value = EXCLUDED.param_value
    """


class _OffsetRow(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True)
    startTime: datetime


_OFFSET_ROWS: Final = TypeAdapter(list[_OffsetRow])

_SPEND_LOG_TABLES: Final = ("LiteLLM_SpendLogs", "LiteLLM_SpendLogToolIndex")

# Partition children are counted with their parent: a partitioned parent has no storage of its own.
_STORAGE_SQL: Final = """
    SELECT coalesce(sum(pg_total_relation_size(s.relid)), 0)::bigint AS allocated_bytes,
           coalesce(sum(s.n_live_tup), 0)::bigint AS live_rows,
           coalesce(sum(s.n_dead_tup), 0)::bigint AS dead_rows,
           coalesce(sum(s.n_tup_ins), 0)::bigint AS inserted_total,
           coalesce(sum(s.n_tup_del), 0)::bigint AS deleted_total
    FROM pg_stat_user_tables s
    WHERE s.schemaname = current_schema()
      AND (
        s.relname = ANY($1::text[])
        OR s.relid IN (
          SELECT i.inhrelid FROM pg_inherits i
          JOIN pg_class p ON p.oid = i.inhparent
          JOIN pg_namespace n ON n.oid = p.relnamespace
          WHERE p.relname = ANY($1::text[]) AND n.nspname = current_schema()
        )
      )
    """

# The age job and the size job are scheduled on separate SpendLogCleanup instances, so the lock is per process.
_PASS_LOCK: Final = asyncio.Lock()

_ANALYZE_SQL: Final = 'ANALYZE "LiteLLM_SpendLogs", "LiteLLM_SpendLogToolIndex"'
# pg_stat_user_tables publishes what ANALYZE counted up to a second later (the stats collector on Postgres 14).
_STATS_SETTLE_SECONDS: Final = 1.0
_STATS_SETTLE_ATTEMPTS: Final = 5

SPEND_LOG_CLEANUP_BOUND_SETTINGS: Final = (
    "maximum_spend_logs_cleanup_batch_size",
    "maximum_spend_logs_cleanup_max_batches",
    "maximum_spend_logs_cleanup_run_budget",
    "maximum_spend_logs_cleanup_batch_timeout",
)


class SpendLogCleanup:
    """
    Handles cleaning up old spend logs based on maximum retention period.

    When LiteLLM_SpendLogs is range-partitioned, expired data is reclaimed by
    dropping whole partitions (instant, frees disk immediately). Otherwise it
    falls back to deleting logs in batches.
    Uses PodLockManager to ensure only one pod runs cleanup in multi-pod deployments.

    Every run is bounded so it can never monopolise the database: a wall-clock
    budget shared across all tables, a per-table batch cap, and a Postgres
    statement/lock timeout on every statement the job issues, deletes and the
    outstanding-rows probe alike. A run that hits a bound stops cleanly and the
    next run resumes from where it left off, because the cutoff is recomputed
    and deleted rows are gone.

    The budget is a hard wall clock, not an advisory one. Every statement this
    job issues, deletes, the outstanding-rows probe and partition DDL alike, is
    issued with a timeout clamped to the budget that is still left, so one
    started just under the deadline is cancelled by Postgres at the deadline
    rather than running a further batch timeout past it. No statement is issued
    at all once the budget is spent, which is why the probe is skipped on that
    path. Partition DDL additionally carries a lock_timeout, because it takes an
    ACCESS EXCLUSIVE lock and would otherwise queue behind a long-running reader
    for as long as that reader lives; a partition this run cannot get is left
    for the next one.
    """

    def __init__(
        self,
        general_settings=None,
        redis_cache: RedisCache | None = None,
        partition_manager: SpendLogsPartitionManager | None = None,
    ):
        self.retention_seconds: int | None = None
        self.partition_manager = partition_manager or SpendLogsPartitionManager()
        from litellm.proxy.proxy_server import general_settings as default_settings

        self.general_settings = general_settings or default_settings
        self._refresh_bounds()
        from litellm.proxy.proxy_server import proxy_logging_obj

        pod_lock_manager: Final = proxy_logging_obj.db_spend_update_writer.pod_lock_manager
        self.pod_lock_manager = pod_lock_manager
        verbose_proxy_logger.info(
            "SpendLogCleanup initialized: batch_size=%s max_batches=%s run_budget=%ss batch_timeout=%ss",
            self.batch_size,
            self.max_batches,
            self.run_budget_seconds,
            self.batch_timeout_seconds,
        )

    def _refresh_bounds(self) -> None:
        """
        Re-read every bound in SPEND_LOG_CLEANUP_BOUND_SETTINGS from settings.

        The scheduler holds one long-lived instance, so a bound captured at
        construction would never reflect a dashboard change. general_settings is
        the same dict the periodic config reload mutates in place, so reading it
        per run is what makes these knobs live. Every bound falls back to its
        shipped default, so clearing a field restores that default.
        """
        self.batch_size: int = self._positive_int_setting(
            "maximum_spend_logs_cleanup_batch_size", SPEND_LOG_CLEANUP_BATCH_SIZE
        )
        self.max_batches: int = self._positive_int_setting(
            "maximum_spend_logs_cleanup_max_batches", SPEND_LOG_RUN_LOOPS
        )
        self.run_budget_seconds: float = self._duration_setting(
            "maximum_spend_logs_cleanup_run_budget", SPEND_LOG_CLEANUP_RUN_BUDGET_SECONDS
        )
        self.batch_timeout_seconds: float = self._duration_setting(
            "maximum_spend_logs_cleanup_batch_timeout", SPEND_LOG_CLEANUP_BATCH_TIMEOUT_SECONDS
        )

    def _positive_int_setting(self, setting_name: str, default: int) -> int:
        """
        Read a positive-integer knob, falling back to the default when unset or unusable.
        """
        raw: Final = self.general_settings.get(setting_name)
        if raw is None:
            return default
        try:
            parsed: Final = int(raw)
        except (TypeError, ValueError):
            verbose_proxy_logger.warning("Invalid %s value: %s, using default %s", setting_name, raw, default)
            return default
        if parsed <= 0:
            verbose_proxy_logger.warning("%s must be positive, got %s, using default %s", setting_name, parsed, default)
            return default
        return parsed

    def _duration_setting(self, setting_name: str, default_seconds: float) -> float:
        """
        Read a duration knob (e.g. '5m'), falling back to the default when unset or unusable.

        The knob must never be able to remove the bound it exists to enforce, so
        anything the parser rejects (including the non-finite spellings 'inf' and
        'nan') and anything non-positive falls back rather than being honoured.
        """
        raw: Final = self.general_settings.get(setting_name)
        if raw is None:
            return default_seconds
        try:
            parsed: Final = float(duration_in_seconds(str(raw)))
        except (ValueError, TypeError) as e:
            verbose_proxy_logger.warning(
                "Invalid %s value: %s (%s), using default %ss", setting_name, raw, e, default_seconds
            )
            return default_seconds
        if parsed <= 0:
            verbose_proxy_logger.warning(
                "%s must be a positive duration, got %s, using default %ss", setting_name, raw, default_seconds
            )
            return default_seconds
        return parsed

    def _retention_seconds_for(self, setting_name: str) -> int | None:
        """
        Parse one retention setting into seconds, or None when unset or invalid.
        """
        retention_setting = self.general_settings.get(setting_name)
        verbose_proxy_logger.info("Checking %s: %s", setting_name, retention_setting)

        if retention_setting is None:
            return None

        try:
            if isinstance(retention_setting, int):
                verbose_proxy_logger.warning(
                    "%s is an integer (%s); treating as days. Use a string like '3d' to be explicit.",
                    setting_name,
                    retention_setting,
                )
                retention_setting = f"{retention_setting}d"
            retention_seconds: Final = duration_in_seconds(retention_setting)
        except ValueError as e:
            verbose_proxy_logger.warning("Invalid %s value: %s, error: %s", setting_name, retention_setting, e)
            return None
        verbose_proxy_logger.info("%s set to %s seconds", setting_name, retention_seconds)
        return retention_seconds

    def _should_delete_spend_logs(self) -> bool:
        """
        Determines if logs should be deleted based on the max retention period in settings.
        """
        self.retention_seconds = self._retention_seconds_for("maximum_spend_logs_retention_period")
        return self.retention_seconds is not None

    def _setting(self, setting_name: str) -> object:
        return self.general_settings.get(setting_name)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # untyped settings dict

    def _size_bytes_setting(self, setting_name: str) -> int | None:
        """Parse one size setting into bytes, or None when unset or unusable."""
        raw: Final = self._setting(setting_name)
        if raw is None:
            return None
        if not isinstance(raw, (str, int)):
            verbose_proxy_logger.warning("Invalid %s value: %r, ignoring it", setting_name, raw)
            return None
        try:
            return size_in_bytes(raw)
        except ValueError as e:
            verbose_proxy_logger.warning("Invalid %s value: %r (%s), ignoring it", setting_name, raw, e)
            return None

    def _size_rule(self) -> SizeRule | None:
        """
        The size rule from settings, or None when maximum_spend_logs_size is unset.

        A target at or above the limit cannot be what the operator meant, so it is reported and the run
        cleans only until it is back under the limit, which is also what an unset target means.
        """
        limit_bytes: Final = self._size_bytes_setting("maximum_spend_logs_size")
        if limit_bytes is None:
            return None
        target_bytes: Final = self._size_bytes_setting("maximum_spend_logs_cleanup_target_size")
        if target_bytes is None:
            return SizeRule(limit_bytes=limit_bytes, target_bytes=limit_bytes)
        if target_bytes >= limit_bytes:
            verbose_proxy_logger.error(
                "maximum_spend_logs_cleanup_target_size (%s bytes) must be below maximum_spend_logs_size (%s bytes); "
                "cleaning down to maximum_spend_logs_size instead",
                target_bytes,
                limit_bytes,
            )
            return SizeRule(limit_bytes=limit_bytes, target_bytes=limit_bytes)
        return SizeRule(limit_bytes=limit_bytes, target_bytes=target_bytes)

    def _size_pass_runs_with_age_pass(self) -> bool:
        """
        With maximum_spend_logs_cleanup_cron set there is one schedule for both rules, so the age run carries
        the size pass; otherwise the size pass has its own job on maximum_spend_logs_size_check_interval.
        """
        return bool(self._setting("maximum_spend_logs_cleanup_cron"))

    async def _probe_storage(self, prisma_client: PrismaClient, deadline: float) -> StorageProbe | None:
        """Read what the spend-log tables weigh from the catalog, or None when the probe fails or has no budget."""
        if time.monotonic() >= deadline:
            return None
        timeout_ms: Final = self._timeout_ms(deadline)
        try:
            async with db_span("measure_storage", "LiteLLM_SpendLogs"), bounded_tx(prisma_client, timeout_ms) as tx:
                await tx.execute_raw(trusted_sql(f"SET LOCAL statement_timeout = {timeout_ms}"))
                rows: Final = _STORAGE_ROWS.validate_python(
                    await tx.query_raw(trusted_sql(_STORAGE_SQL), list(_SPEND_LOG_TABLES))
                )
        except Exception as e:  # noqa: BLE001 - a failed measurement skips the size pass, it must not fail the run
            verbose_proxy_logger.warning("Could not measure spend-log storage: %s: %s", type(e).__name__, e)
            return None
        if not rows:
            return None
        row: Final = rows[0]
        return StorageProbe(
            allocated_bytes=row.allocated_bytes,
            live_rows=row.live_rows,
            dead_rows=row.dead_rows,
            inserted_total=row.inserted_total,
            deleted_total=row.deleted_total,
        )

    async def _reprobe_after_analyze(
        self, prisma_client: PrismaClient, deadline: float, stale: StorageProbe
    ) -> StorageProbe | None:
        """
        Row counters that are all zero under a file with data were reset, e.g. by a crash. ANALYZE rebuilds
        n_live_tup and n_dead_tup from a 30,000-row sample in seconds and blocks nothing, so the pass runs it and
        reads the catalog again rather than trusting an estimate of zero or waiting a day for autoanalyze. The
        new counts show up in pg_stat_user_tables a moment after ANALYZE returns, hence the settled re-reads.
        """
        verbose_proxy_logger.warning(
            "Spend-log row statistics are stale (allocated=%s bytes, live_rows=0, dead_rows=0); running %s before "
            "the size pass",
            stale.allocated_bytes,
            _ANALYZE_SQL,
        )
        if time.monotonic() >= deadline:
            return None
        timeout_ms: Final = self._timeout_ms(deadline)
        try:
            async with db_span("analyze", "LiteLLM_SpendLogs"), bounded_tx(prisma_client, timeout_ms) as tx:
                await tx.execute_raw(trusted_sql(f"SET LOCAL statement_timeout = {timeout_ms}"))
                await tx.execute_raw(trusted_sql(_ANALYZE_SQL))
        except Exception as e:  # noqa: BLE001 - a failed ANALYZE skips the size pass, it must not fail the run
            verbose_proxy_logger.warning("Could not ANALYZE the spend-log tables: %s: %s", type(e).__name__, e)
            return None
        refreshed: Final = await self._probe_once_settled(prisma_client, deadline, _STATS_SETTLE_ATTEMPTS)
        if refreshed is None or refreshed.stats_are_stale:
            verbose_proxy_logger.warning(
                "Spend-log row statistics are still stale after ANALYZE; skipping the size pass"
            )
            return None
        return refreshed

    async def _probe_once_settled(
        self, prisma_client: PrismaClient, deadline: float, attempts_left: int
    ) -> StorageProbe | None:
        await asyncio.sleep(_STATS_SETTLE_SECONDS)
        refreshed: Final = await self._probe_storage(prisma_client, deadline)
        if refreshed is None or not refreshed.stats_are_stale or attempts_left <= 1:
            return refreshed
        return await self._probe_once_settled(prisma_client, deadline, attempts_left - 1)

    async def _load_size_state(self, prisma_client: PrismaClient, deadline: float) -> SizeCleanupState | None:
        """The reclaimable count saved by the last check, or None on the first check or when it cannot be read."""
        if time.monotonic() >= deadline:
            return None
        timeout_ms: Final = self._timeout_ms(deadline)
        try:
            async with db_span("load_size_state", "LiteLLM_Config"), bounded_tx(prisma_client, timeout_ms) as tx:
                await tx.execute_raw(trusted_sql(f"SET LOCAL statement_timeout = {timeout_ms}"))
                rows: Final = _SIZE_STATE_ROWS.validate_python(
                    await tx.query_raw(trusted_sql(_LOAD_SIZE_STATE_SQL), SPEND_LOG_SIZE_CLEANUP_STATE_PARAM)
                )
            return SizeCleanupState.model_validate_json(rows[0].state_json) if rows else None
        except Exception as e:  # noqa: BLE001 - an unreadable count restarts from the catalog, it must not fail the run
            verbose_proxy_logger.warning("Could not read the spend-log size cleanup state: %s: %s", type(e).__name__, e)
            return None

    async def _save_size_state(self, prisma_client: PrismaClient, deadline: float, state: SizeCleanupState) -> None:
        timeout_ms: Final = max(self._timeout_ms(deadline), 1000)
        try:
            async with db_span("save_size_state", "LiteLLM_Config"), bounded_tx(prisma_client, timeout_ms) as tx:
                await tx.execute_raw(trusted_sql(f"SET LOCAL statement_timeout = {timeout_ms}"))
                await tx.execute_raw(
                    trusted_sql(_SAVE_SIZE_STATE_SQL), SPEND_LOG_SIZE_CLEANUP_STATE_PARAM, state.model_dump_json()
                )
        except Exception as e:  # noqa: BLE001 - a lost save costs one re-seed from the catalog, it must not fail the run
            verbose_proxy_logger.warning("Could not save the spend-log size cleanup state: %s: %s", type(e).__name__, e)

    async def _size_step_cutoff(self, prisma_client: PrismaClient, deadline: float, rows: int) -> datetime | None:
        """
        The startTime that puts the `rows` oldest rows behind it.

        Expressing a size step as a cutoff lets the size pass reuse the retention DELETE unchanged, including
        the tool-index rows that expire with their spend logs. When fewer rows than that remain, the step is
        everything up to now.
        """
        if time.monotonic() >= deadline:
            return None
        timeout_ms: Final = self._timeout_ms(deadline)
        async with db_span("size_step_cutoff", "LiteLLM_SpendLogs"), bounded_tx(prisma_client, timeout_ms) as tx:
            await tx.execute_raw(trusted_sql(f"SET LOCAL statement_timeout = {timeout_ms}"))
            found: Final = _OFFSET_ROWS.validate_python(
                await tx.query_raw(
                    trusted_sql(
                        'SELECT "startTime" FROM "LiteLLM_SpendLogs" ORDER BY "startTime" ASC OFFSET $1 LIMIT 1'
                    ),
                    rows,
                )
            )
        if not found:
            return datetime.now(timezone.utc)
        start_time: Final = found[0].startTime
        return start_time if start_time.tzinfo is not None else start_time.replace(tzinfo=timezone.utc)

    def _timeout_ms(self, deadline: float) -> int:
        """
        The per-statement bound in milliseconds: the batch timeout, or whatever
        is left of the run budget, whichever is smaller.

        Clamping to the remaining budget is what makes the budget a real
        wall-clock bound rather than an advisory one. Postgres offers no "stop
        at time T", only a per-statement duration, so a statement issued just
        under the deadline would otherwise run a full batch timeout past it, and
        with several tables those overruns stack.

        Interpolating this into SQL is safe by construction: an int cannot carry
        SQL, and SET does not accept a bind parameter.
        """
        remaining_ms: Final = int((deadline - time.monotonic()) * 1000)
        return max(1, min(int(self.batch_timeout_seconds * 1000), remaining_ms))

    @staticmethod
    def _group_deadline(overall_deadline: float, groups_remaining: int) -> float:
        """
        Give each pending cleanup group an equal share of the time left.

        A single group keeps the whole run budget, while a persistent backlog
        on an earlier group cannot starve a later group.
        """
        if groups_remaining == 1:
            return overall_deadline
        current_time: Final = time.monotonic()
        if current_time >= overall_deadline:
            return overall_deadline
        return current_time + (overall_deadline - current_time) / groups_remaining

    def _remaining_timeout_ms(self, deadline: float) -> RemainingTimeoutMs:
        """
        The per-statement bound for work this job delegates, as a callable.

        Partition maintenance issues one statement per partition, so handing it a
        number would bound each statement by the budget that was left before the
        FIRST one and never by what remains. Re-evaluating per statement is what
        makes the loop itself bounded, and None tells the callee to stop rather
        than issue a statement it has no budget for.
        """

        def remaining() -> int | None:
            return None if time.monotonic() >= deadline else self._timeout_ms(deadline)

        return remaining

    async def _execute_delete_batch(
        self, prisma_client: PrismaClient, delete_sql: str, cutoff_date: Cutoff, table_name: str, deadline: float
    ) -> int | None:
        """
        Run one delete batch under a Postgres statement and lock timeout.

        The timeouts are what actually bound the work: a Prisma transaction
        timeout cannot interrupt a statement that is already executing, so
        without these a single batch blocked behind a lock would hold its
        connection, and the row locks it already took, indefinitely. SET LOCAL
        scopes both to this transaction so the pooled connection is unaffected.

        Returns the row count, or None when the driver returned something that
        is not a row count. That is a contract violation rather than a transient
        fault, so the caller stops instead of retrying.
        """
        timeout_ms: Final = self._timeout_ms(deadline)
        async with db_span("cleanup_expired_rows", table_name), bounded_tx(prisma_client, timeout_ms) as tx:
            await tx.execute_raw(trusted_sql(f"SET LOCAL statement_timeout = {timeout_ms}"))
            await tx.execute_raw(trusted_sql(f"SET LOCAL lock_timeout = {timeout_ms}"))
            deleted_result: Final = await tx.execute_raw(trusted_sql(delete_sql), cutoff_date, self.batch_size)
        return (
            deleted_result
            if isinstance(deleted_result, int)  # pyright: ignore[reportUnnecessaryIsInstance]  # driver may return None
            else None
        )

    async def _count_remaining(
        self, prisma_client: PrismaClient, cutoff_date: Cutoff, table_name: str, time_column: str, deadline: float
    ) -> int | None:
        """
        Count expired rows still outstanding, stopping at a cap.

        An uncapped COUNT(*) over an expired backlog would itself be the kind of
        long scan this job exists to avoid, so the probe reads at most
        SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP index entries. A result equal to
        the cap means "at least this many".
        """
        count_sql: Final = f"""
            SELECT count(*)::int AS remaining FROM (
                SELECT 1 FROM "{table_name}"
                WHERE "{time_column}" < $1::{_cutoff_cast(cutoff_date)}
                LIMIT $2
            ) capped
            """
        timeout_ms: Final = self._timeout_ms(deadline)
        try:
            async with db_span("count_expired_rows", table_name), bounded_tx(prisma_client, timeout_ms) as tx:
                await tx.execute_raw(trusted_sql(f"SET LOCAL statement_timeout = {timeout_ms}"))
                rows: Final = _REMAINING_ROWS.validate_python(
                    await tx.query_raw(trusted_sql(count_sql), cutoff_date, SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP)
                )
        except Exception as e:  # noqa: BLE001 - an observability probe must never fail the cleanup run
            verbose_proxy_logger.warning("Could not count remaining %s rows: %s", table_name, e)
            return None
        return rows[0].remaining if rows else None

    async def _delete_old_rows_batched(
        self,
        prisma_client: PrismaClient,
        cutoff_date: Cutoff,
        table_name: str,
        key_columns: tuple[str, ...],
        time_column: str,
        deadline: float,
        batch_cap: int | None = None,
    ) -> TableCleanupResult:
        """
        Delete a table's rows older than the cutoff in batches.

        Stops at whichever bound is reached first: the backlog running out, the
        shared wall-clock deadline, the per-table batch cap (`batch_cap` when a
        caller spreads one table's cap over several calls, else the configured
        one), or too many consecutive batch failures.
        """
        max_batches: Final = self.max_batches if batch_cap is None else batch_cap
        key_list: Final = ", ".join(f'"{col}"' for col in key_columns)
        delete_sql: Final = f"""
            DELETE FROM "{table_name}"
            WHERE ({key_list}) IN (
                SELECT {key_list} FROM "{table_name}"
                WHERE "{time_column}" < $1::{_cutoff_cast(cutoff_date)}
                LIMIT $2
            )
            """
        total_deleted = 0
        run_count = 0
        consecutive_failures = 0
        while True:
            if time.monotonic() >= deadline:
                verbose_proxy_logger.info(
                    "Run budget exhausted during %s cleanup after %d rows; the next run resumes from here",
                    table_name,
                    total_deleted,
                )
                return await self._finish_table(
                    prisma_client, cutoff_date, table_name, time_column, total_deleted, "budget_exhausted", deadline
                )
            if run_count >= max_batches:
                verbose_proxy_logger.info(
                    "Max batches reached for %s cleanup, remaining rows will be deleted in next run", table_name
                )
                return await self._finish_table(
                    prisma_client, cutoff_date, table_name, time_column, total_deleted, "batch_cap_reached", deadline
                )
            # Find rows and delete them in one go without fetching to application
            batch_started_at = time.monotonic()
            try:
                batch_result = await self._execute_delete_batch(
                    prisma_client, delete_sql, cutoff_date, table_name, deadline
                )
            except Exception as batch_exc:
                if time.monotonic() >= deadline:
                    # The statement timeout was clamped to the budget that was
                    # left, so this batch was cancelled by the deadline itself.
                    # That is the bound working, not a database fault, and
                    # counting it would both inflate the failure metric and push
                    # every budget-exhausted run toward the abort threshold.
                    verbose_proxy_logger.info(
                        "Run budget exhausted mid-batch during %s cleanup after %d rows; "
                        "the next run resumes from here",
                        table_name,
                        total_deleted,
                    )
                    return await self._finish_table(
                        prisma_client, cutoff_date, table_name, time_column, total_deleted, "budget_exhausted", deadline
                    )
                # A single batch failure (e.g. Prisma/DB timeout) must not abort
                # the whole run — subsequent batches may still succeed.
                consecutive_failures += 1
                SpendLogCleanupMetrics.record_batch_failure(table_name)
                verbose_proxy_logger.exception(
                    "%s cleanup batch failed "
                    "(run_count=%d, consecutive_failures=%d, batch_size=%d, "
                    "cutoff=%s, total_deleted_so_far=%d): %s: %s",
                    table_name,
                    run_count,
                    consecutive_failures,
                    self.batch_size,
                    _cutoff_text(cutoff_date),
                    total_deleted,
                    type(batch_exc).__name__,
                    batch_exc,
                )
                if consecutive_failures >= SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES:
                    verbose_proxy_logger.error(
                        "Aborting %s cleanup after %d consecutive batch failures; total deleted before abort: %d",
                        table_name,
                        consecutive_failures,
                        total_deleted,
                    )
                    return await self._finish_table(
                        prisma_client, cutoff_date, table_name, time_column, total_deleted, "aborted", deadline
                    )
                await asyncio.sleep(SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS)
                continue

            if batch_result is None:
                verbose_proxy_logger.error(
                    "Unexpected execute_raw return type for %s cleanup; aborting cleanup to avoid infinite loop",
                    table_name,
                )
                return await self._finish_table(
                    prisma_client, cutoff_date, table_name, time_column, total_deleted, "aborted", deadline
                )

            consecutive_failures = 0
            deleted_count = batch_result
            SpendLogCleanupMetrics.record_batch(table_name, deleted_count, time.monotonic() - batch_started_at)
            verbose_proxy_logger.info("Deleted %s %s rows in this batch", deleted_count, table_name)

            if deleted_count == 0:
                verbose_proxy_logger.info("No more %s rows to delete. Total deleted: %s", table_name, total_deleted)
                return await self._finish_table(
                    prisma_client, cutoff_date, table_name, time_column, total_deleted, "exhausted", deadline
                )

            total_deleted += deleted_count
            run_count += 1
            _record_run_batch(deleted_count)

            # Add a small sleep to prevent overwhelming the database
            await asyncio.sleep(0.1)

    async def _finish_table(
        self,
        prisma_client: PrismaClient,
        cutoff_date: Cutoff,
        table_name: str,
        time_column: str,
        rows_deleted: int,
        stop_reason: StopReason,
        deadline: float,
    ) -> TableCleanupResult:
        """
        Publish how much of this table is still outstanding, then report the run's result.

        The probe is skipped once the budget is spent. It is the one piece of
        work that would otherwise be ISSUED after the deadline, and every table
        exits through here, including the ones a spent run never started, so
        keeping it would put one more statement per table past the bound. A run
        that ends this way already reports "budget_exhausted", which tells an
        operator the backlog was not drained; the gauge simply keeps its value
        from the last run that finished inside its budget.
        """
        if time.monotonic() >= deadline:
            return TableCleanupResult(table_name=table_name, rows_deleted=rows_deleted, stop_reason=stop_reason)
        remaining: Final = await self._count_remaining(prisma_client, cutoff_date, table_name, time_column, deadline)
        if remaining is not None:
            SpendLogCleanupMetrics.set_rows_remaining(table_name, remaining)
        return TableCleanupResult(table_name=table_name, rows_deleted=rows_deleted, stop_reason=stop_reason)

    async def _delete_old_logs(
        self, prisma_client: PrismaClient, cutoff_date: datetime, deadline: float
    ) -> TableCleanupResult:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_SpendLogs",
            key_columns=("request_id", "startTime"),
            time_column="startTime",
            deadline=deadline,
        )

    async def _delete_old_tool_index_rows(
        self, prisma_client: PrismaClient, cutoff_date: datetime, deadline: float
    ) -> TableCleanupResult:
        # SpendLogToolIndex rows are derived from spend logs, so they expire on the
        # same cutoff; rows older than retention point at already-deleted logs.
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_SpendLogToolIndex",
            key_columns=("request_id", "tool_name"),
            time_column="start_time",
            deadline=deadline,
        )

    async def _delete_old_autorouter_session_rows(
        self, prisma_client: PrismaClient, cutoff_date: datetime, deadline: float
    ) -> TableCleanupResult:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_AutoRouterSession",
            key_columns=("api_key", "session_id", "router_name"),
            time_column="last_turn_at",
            deadline=deadline,
        )

    async def _delete_old_autorouter_user_session_rows(
        self, prisma_client: PrismaClient, cutoff_date: datetime, deadline: float
    ) -> TableCleanupResult:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_AutoRouterUserSession",
            key_columns=("user_id", "api_key", "session_id", "router_name"),
            time_column="last_turn_at",
            deadline=deadline,
        )

    async def _delete_old_autorouter_daily_rows(
        self, prisma_client: PrismaClient, cutoff_day: str, deadline: float
    ) -> TableCleanupResult:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_day,
            table_name="LiteLLM_AutoRouterDailySpend",
            key_columns=("date", "api_key", "user_id", "router_name", "router_type"),
            time_column="date",
            deadline=deadline,
        )

    async def _delete_old_health_check_rows(
        self, prisma_client: PrismaClient, cutoff_date: datetime, deadline: float
    ) -> TableCleanupResult:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_HealthCheckTable",
            key_columns=("health_check_id",),
            time_column="checked_at",
            deadline=deadline,
        )

    async def _delete_old_daily_tag_spend_rows(
        self, prisma_client: PrismaClient, cutoff_day: str, deadline: float
    ) -> TableCleanupResult:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_day,
            table_name="LiteLLM_DailyTagSpend",
            key_columns=("id",),
            time_column="date",
            deadline=deadline,
        )

    async def _clean_spend_log_tables(
        self, prisma_client: PrismaClient, deadline: float
    ) -> tuple[TableCleanupResult, ...]:
        """
        Prune the spend logs and the tool index rows derived from them.

        When the table is range-partitioned, whole expired partitions are dropped
        first because that reclaims disk immediately. Expired rows can still sit in
        the DEFAULT partition (backfill, coverage gaps) or in a partition that spans
        the cutoff, so retention still deletes those stragglers row-wise.
        """
        cutoff_date: Final = datetime.now(timezone.utc) - timedelta(seconds=float(self.retention_seconds or 0))
        verbose_proxy_logger.info("Removing logs older than %s", cutoff_date.isoformat())

        # Partition maintenance is DDL taking an ACCESS EXCLUSIVE lock, so it is
        # only STARTED while the run still has budget, and each statement carries
        # the same timeouts the batches do. Without those, a DROP would queue
        # behind any long-running reader for as long as that reader lives, which
        # is the one way this job could still outlast its budget without bound.
        remaining_timeout_ms: Final = self._remaining_timeout_ms(deadline)
        if time.monotonic() >= deadline:
            verbose_proxy_logger.info("Run budget already spent, skipping partition maintenance this run")
        elif self.general_settings.get(
            "use_spend_logs_partitioning", False
        ) and await self.partition_manager.is_partitioned(prisma_client, remaining_timeout_ms):
            await self.partition_manager.ensure_partitions(prisma_client, remaining_timeout_ms)
            dropped: Final = await self.partition_manager.drop_partitions_older_than(
                prisma_client, cutoff_date, remaining_timeout_ms
            )
            verbose_proxy_logger.info("Dropped %d expired spend-log partitions: %s", len(dropped), dropped)

        logs_result: Final = await self._delete_old_logs(
            prisma_client, cutoff_date, self._group_deadline(deadline, groups_remaining=2)
        )
        verbose_proxy_logger.info("Deleted %s logs", logs_result.rows_deleted)

        index_result: Final = await self._delete_old_tool_index_rows(prisma_client, cutoff_date, deadline)
        verbose_proxy_logger.info("Deleted %s expired tool index rows", index_result.rows_deleted)
        return (logs_result, index_result)

    async def _clean_spend_logs_by_size(
        self, prisma_client: PrismaClient, rule: SizeRule, deadline: float
    ) -> SizePassReport | None:
        """
        Bring the spend-log tables under the size target, oldest rows first.

        Live bytes are the files on disk less the reclaimable count carried in LiteLLM_Config (see
        `SizeCleanupState`), rolled forward from the catalog's cumulative insert and delete counters. A
        partitioned table gives its oldest fully elapsed partitions back first, because a DROP returns disk at
        once, re-measuring after each. Then rows go in steps: the rows still needed to reach the target follow
        from bytes per row, and each step turns that many of the oldest rows (at most
        SPEND_LOG_SIZE_CLEANUP_RECHECK_BATCHES batches) into a cutoff and runs the same bounded DELETE the age
        rule uses, for the spend logs and the tool index rows derived from them. Progress is counted from the
        deletes themselves, not re-read from pg_stat_user_tables, whose counters trail a DELETE by up to a
        second. The run's batch cap and budget apply to the whole pass, so a pass that stops short reports how
        far above the target it still is and the next check continues from there.
        """
        measured: Final = await self._probe_storage(prisma_client, deadline)
        if measured is None:
            return None
        probe: Final = (
            await self._reprobe_after_analyze(prisma_client, deadline, measured)
            if measured.stats_are_stale
            else measured
        )
        if probe is None:
            return None
        state: Final = advance_size_state(await self._load_size_state(prisma_client, deadline), probe)
        before: Final = SpendLogStorage(
            allocated_bytes=probe.allocated_bytes,
            reclaimable_bytes=state.reclaimable_bytes,
            bytes_per_row=state.bytes_per_row,
            live_rows=probe.live_rows,
        )
        SpendLogCleanupMetrics.set_storage_bytes("allocated", before.allocated_bytes)
        SpendLogCleanupMetrics.set_storage_bytes("estimated_live", before.estimated_live_bytes)
        if before.estimated_live_bytes <= rule.limit_bytes:
            verbose_proxy_logger.info(
                "Spend-log storage is under maximum_spend_logs_size (estimated_live=%s allocated=%s reclaimable=%s "
                "limit=%s bytes)",
                before.estimated_live_bytes,
                before.allocated_bytes,
                before.reclaimable_bytes,
                rule.limit_bytes,
            )
            SpendLogCleanupMetrics.set_storage_bytes("above_target", 0)
            await self._save_size_state(prisma_client, deadline, state)
            return SizePassReport(rule=rule, before=before, after=before, partitions_dropped=(), results=())
        verbose_proxy_logger.warning(
            "Spend-log storage is over maximum_spend_logs_size (estimated_live=%s allocated=%s reclaimable=%s "
            "limit=%s bytes); deleting the oldest spend logs down to %s bytes",
            before.estimated_live_bytes,
            before.allocated_bytes,
            before.reclaimable_bytes,
            rule.limit_bytes,
            rule.target_bytes,
        )
        latest: Final[list[SpendLogStorage]] = [before]  # mutable-ok: last measurement, rewritten per partition drop
        baseline: Final[list[StorageProbe]] = [probe]  # mutable-ok: counters after the last drop

        remeasure_failed: Final[list[bool]] = [False]  # mutable-ok: set when a post-drop probe fails

        async def still_over_target() -> bool:
            measured: Final = await self._probe_storage(prisma_client, deadline)
            if measured is None:
                remeasure_failed[0] = True
                return False
            kept: Final = min(1.0, measured.allocated_bytes / max(latest[0].allocated_bytes, 1))
            latest[0] = SpendLogStorage(
                allocated_bytes=measured.allocated_bytes,
                reclaimable_bytes=min(measured.allocated_bytes, round(latest[0].reclaimable_bytes * kept)),
                bytes_per_row=latest[0].bytes_per_row,
                live_rows=measured.live_rows,
            )
            baseline[0] = measured
            return latest[0].estimated_live_bytes > rule.target_bytes

        partitions_dropped: Final = await self._drop_oldest_partitions_for_size(
            prisma_client, deadline, still_over_target
        )
        if remeasure_failed[0]:
            verbose_proxy_logger.warning(
                "Spend-log storage could not be re-measured after dropping partitions %s; the row step and the saved "
                "count wait for the next check rather than run on the pre-drop numbers",
                partitions_dropped,
            )
            return SizePassReport(
                rule=rule, before=before, after=latest[0], partitions_dropped=partitions_dropped, results=()
            )
        base: Final = latest[0]
        rows_to_delete: Final = max(0, math.ceil((base.estimated_live_bytes - rule.target_bytes) / base.bytes_per_row))

        def rows_still_needed(rows_deleted: int) -> int:
            return rows_to_delete - rows_deleted

        results: Final = await self._delete_oldest_rows_for_size(prisma_client, deadline, rows_still_needed)
        rows_deleted: Final = sum(result.rows_deleted for result in results)
        remeasured: Final = await self._probe_storage(prisma_client, deadline)
        after: Final = base.after_deleting(
            rows_deleted, remeasured.allocated_bytes if remeasured is not None else base.allocated_bytes
        )
        await self._save_size_state(
            prisma_client,
            deadline,
            after.to_state(
                inserted_total=baseline[0].inserted_total, deleted_total=baseline[0].deleted_total + rows_deleted
            ),
        )
        report: Final = SizePassReport(
            rule=rule, before=before, after=after, partitions_dropped=partitions_dropped, results=results
        )
        SpendLogCleanupMetrics.set_storage_bytes("allocated", after.allocated_bytes)
        SpendLogCleanupMetrics.set_storage_bytes("estimated_live", after.estimated_live_bytes)
        SpendLogCleanupMetrics.set_storage_bytes("above_target", report.above_target_bytes)
        return report

    async def _drop_oldest_partitions_for_size(
        self, prisma_client: PrismaClient, deadline: float, still_over_target: Callable[[], Awaitable[bool]]
    ) -> tuple[str, ...]:
        remaining_timeout_ms: Final = self._remaining_timeout_ms(deadline)
        if time.monotonic() >= deadline or not self._setting("use_spend_logs_partitioning"):
            return ()
        if not await self.partition_manager.is_partitioned(prisma_client, remaining_timeout_ms):
            return ()
        await self.partition_manager.ensure_partitions(prisma_client, remaining_timeout_ms)
        dropped: Final = await self.partition_manager.drop_oldest_partitions_while(
            prisma_client, datetime.now(timezone.utc), remaining_timeout_ms, still_over_target
        )
        verbose_proxy_logger.info("Dropped %d oldest spend-log partitions for the size rule: %s", len(dropped), dropped)
        return tuple(dropped)

    async def _delete_oldest_rows_for_size(
        self,
        prisma_client: PrismaClient,
        deadline: float,
        rows_still_needed: Callable[[int], int],
    ) -> tuple[TableCleanupResult, ...]:
        state = _SizeStepState(  # rebind-ok: one value threaded through the bounded step loop
            logs_deleted=0, index_deleted=0, batches_left=self.max_batches, stop_reason=None
        )
        for _ in range(self.max_batches // SPEND_LOG_SIZE_CLEANUP_RECHECK_BATCHES + 2):
            if state.stop_reason is not None:
                break
            state = await self._run_size_step(prisma_client, deadline, rows_still_needed, state)
        final_reason: Final[StopReason] = state.stop_reason or "batch_cap_reached"
        return (
            TableCleanupResult(
                table_name="LiteLLM_SpendLogs (size rule)", rows_deleted=state.logs_deleted, stop_reason=final_reason
            ),
            TableCleanupResult(
                table_name="LiteLLM_SpendLogToolIndex (size rule)",
                rows_deleted=state.index_deleted,
                stop_reason=final_reason,
            ),
        )

    async def _run_size_step(
        self,
        prisma_client: PrismaClient,
        deadline: float,
        rows_still_needed: Callable[[int], int],
        state: "_SizeStepState",
    ) -> "_SizeStepState":
        """One step: delete the rows still needed, at most SPEND_LOG_SIZE_CLEANUP_RECHECK_BATCHES batches of the oldest."""
        needed: Final = rows_still_needed(state.logs_deleted + state.index_deleted)
        if needed <= 0:
            return state.stopped("target_reached")
        if time.monotonic() >= deadline:
            return state.stopped("budget_exhausted")
        if state.batches_left <= 0:
            return state.stopped("batch_cap_reached")
        step_cap: Final = min(
            SPEND_LOG_SIZE_CLEANUP_RECHECK_BATCHES, state.batches_left, math.ceil(needed / self.batch_size)
        )
        cutoff: Final = await self._size_step_cutoff(prisma_client, deadline, step_cap * self.batch_size)
        if cutoff is None:
            return state.stopped("budget_exhausted")
        logs_result: Final = await self._delete_old_rows_batched(
            prisma_client,
            cutoff,
            table_name="LiteLLM_SpendLogs",
            key_columns=("request_id", "startTime"),
            time_column="startTime",
            deadline=deadline,
            batch_cap=step_cap,
        )
        index_result: Final = await self._delete_old_rows_batched(
            prisma_client,
            cutoff,
            table_name="LiteLLM_SpendLogToolIndex",
            key_columns=("request_id", "tool_name"),
            time_column="start_time",
            deadline=deadline,
            batch_cap=step_cap,
        )
        advanced: Final = _SizeStepState(
            logs_deleted=state.logs_deleted + logs_result.rows_deleted,
            index_deleted=state.index_deleted + index_result.rows_deleted,
            batches_left=state.batches_left - step_cap,
            stop_reason=None,
        )
        if rows_still_needed(advanced.logs_deleted + advanced.index_deleted) <= 0:
            return advanced.stopped("target_reached")
        if logs_result.stop_reason in ("aborted", "budget_exhausted") or index_result.stop_reason == "aborted":
            return advanced.stopped(logs_result.stop_reason if logs_result.stop_reason != "exhausted" else "aborted")
        if logs_result.rows_deleted == 0:
            return advanced.stopped("exhausted")
        return advanced

    async def _clean_session_rollup(
        self, prisma_client: PrismaClient, retention_seconds: int, deadline: float
    ) -> tuple[TableCleanupResult, ...]:
        """
        Prune auto-router session rollup rows, which carry their own retention horizon.
        """
        session_cutoff: Final = datetime.now(timezone.utc) - timedelta(seconds=float(retention_seconds))
        from litellm.proxy.db.baseline_accounting import BaselineAccountingStore

        if remaining_ms := self._remaining_timeout_ms(deadline)():
            try:
                await BaselineAccountingStore.for_client(prisma_client).retire_before(
                    session_cutoff,
                    self.batch_size,
                    remaining_ms,
                )
            except Exception:  # noqa: BLE001  # retained observations are retried by the next cleanup job
                verbose_proxy_logger.warning("Auto-router baseline retention remains pending")
        sessions_result: Final = await self._delete_old_autorouter_session_rows(
            prisma_client, session_cutoff, self._group_deadline(deadline, 3)
        )
        verbose_proxy_logger.info("Deleted %s expired auto-router session rollup rows", sessions_result.rows_deleted)
        user_sessions_result: Final = await self._delete_old_autorouter_user_session_rows(
            prisma_client, session_cutoff, self._group_deadline(deadline, 2)
        )
        verbose_proxy_logger.info(
            "Deleted %s expired auto-router user session rollup rows", user_sessions_result.rows_deleted
        )
        days_result: Final = await self._delete_old_autorouter_daily_rows(
            prisma_client, session_cutoff.date().isoformat(), deadline
        )
        verbose_proxy_logger.info("Deleted %s expired auto-router daily rollup rows", days_result.rows_deleted)
        return (sessions_result, user_sessions_result, days_result)

    async def _clean_health_checks(
        self, prisma_client: PrismaClient, retention_seconds: int, deadline: float
    ) -> tuple[TableCleanupResult, ...]:
        health_check_cutoff: Final = datetime.now(timezone.utc) - timedelta(seconds=float(retention_seconds))
        health_checks_result: Final = await self._delete_old_health_check_rows(
            prisma_client, health_check_cutoff, deadline
        )
        verbose_proxy_logger.info(
            "Deleted %s expired health-check rows",
            health_checks_result.rows_deleted,
        )
        return (health_checks_result,)

    async def _clean_daily_tag_spend(
        self, prisma_client: PrismaClient, retention_seconds: int, deadline: float
    ) -> tuple[TableCleanupResult, ...]:
        """
        Prune per-day tag spend rows whose ISO day sorts before the horizon day; the horizon day itself is kept.
        """
        horizon: Final = datetime.now(timezone.utc) - timedelta(seconds=float(retention_seconds))
        cutoff_day: Final = horizon.date().isoformat()
        result: Final = await self._delete_old_daily_tag_spend_rows(prisma_client, cutoff_day, deadline)
        verbose_proxy_logger.info("Deleted %s expired daily tag spend rows", result.rows_deleted)
        return (result,)

    @staticmethod
    def _run_outcome(results: tuple[TableCleanupResult, ...]) -> RunOutcome:
        """
        Report the most operationally significant reason the run stopped.

        A bound that was hit matters more than a table that simply ran dry, so
        those win over "completed", and an abort wins over everything.
        """
        reasons: Final = frozenset(result.stop_reason for result in results)
        if "aborted" in reasons:
            return "aborted"
        if "budget_exhausted" in reasons:
            return "budget_exhausted"
        if "batch_cap_reached" in reasons:
            return "batch_cap_reached"
        return "completed"

    @staticmethod
    def _storage_summary(report: SizePassReport | None) -> str:
        """
        The size pass in one clause: what was measured, where it ended against the target, and whether the
        operator still has to act to get the disk back.
        """
        if report is None:
            return ""
        rows_deleted: Final = sum(result.rows_deleted for result in report.results)
        reclaim_note: Final = (
            "; allocated bytes stay until autovacuum reuses them, only VACUUM FULL or pg_repack on the spend-log "
            "tables returns them to the filesystem"
            if rows_deleted > 0 and report.after.allocated_bytes >= report.before.allocated_bytes
            else ""
        )
        return (
            f" storage: allocated_before={report.before.allocated_bytes} "
            f"estimated_live_before={report.before.estimated_live_bytes} "
            f"allocated_after={report.after.allocated_bytes} estimated_live_after={report.after.estimated_live_bytes} "
            f"reclaimable_after={report.after.reclaimable_bytes} "
            f"limit={report.rule.limit_bytes} target={report.rule.target_bytes} "
            f"target_reached={str(report.target_reached).lower()} above_target={report.above_target_bytes} "
            f"partitions_dropped={len(report.partitions_dropped)}{reclaim_note}"
        )

    @classmethod
    def _log_run_summary(
        cls,
        outcome: RunOutcome,
        results: tuple[TableCleanupResult, ...],
        elapsed_seconds: float,
        size_report: SizePassReport | None = None,
    ) -> None:
        per_table: Final = ", ".join(
            f"{result.table_name}: deleted={result.rows_deleted} stop_reason={result.stop_reason}" for result in results
        )
        message: Final = "Spend log cleanup run finished: outcome=%s elapsed=%.1fs [%s]%s"
        storage: Final = cls._storage_summary(size_report)
        if outcome == "completed" and (size_report is None or size_report.target_reached):
            verbose_proxy_logger.info(message, outcome, elapsed_seconds, per_table, storage)
            return
        verbose_proxy_logger.warning(message, outcome, elapsed_seconds, per_table, storage)

    async def cleanup_old_spend_logs(self, prisma_client: PrismaClient) -> None:
        """
        The scheduled retention run: every age rule, plus the size rule when one cron drives both.
        If pod_lock_manager is available, ensures only one pod runs cleanup.
        If no pod_lock_manager, runs cleanup without distributed locking.
        """
        await self._run(prisma_client, age_rules=True, size_rule=self._size_pass_runs_with_age_pass())

    async def cleanup_spend_logs_by_size(self, prisma_client: PrismaClient) -> None:
        """The size-check run scheduled on maximum_spend_logs_size_check_interval: the size rule only."""
        await self._run(prisma_client, age_rules=False, size_rule=True)

    async def _run(self, prisma_client: PrismaClient, age_rules: bool, size_rule: bool) -> None:
        """
        One pass at a time per pod. The age job and the size job fire from separate schedules, and the pod
        lock only exists with Redis, so this is what keeps them apart in a pod without it. The later one waits
        rather than skips, because the other job applies a different rule and would not do its work for it.
        """
        if _PASS_LOCK.locked():
            verbose_proxy_logger.info("Another cleanup pass is running in this pod; waiting for it to finish")
        async with _PASS_LOCK:
            await self._run_passes(prisma_client, age_rules, size_rule)

    async def _run_passes(self, prisma_client: PrismaClient, age_rules: bool, size_rule: bool) -> None:
        lock_acquired = False
        run_started_at: Final = time.monotonic()
        progress: Final = _RunProgress()
        progress_token: Final = _run_progress.set(progress)
        try:
            verbose_proxy_logger.info("Cleanup job triggered at %s", datetime.now())
            self._refresh_bounds()

            delete_spend_logs: Final = age_rules and self._should_delete_spend_logs()
            active_size_rule: Final = self._size_rule() if size_rule else None
            autorouter_retention_seconds: Final = (
                self._retention_seconds_for("maximum_autorouter_session_retention_period") if age_rules else None
            )
            health_check_retention_seconds: Final = (
                self._retention_seconds_for("maximum_health_check_retention_period") if age_rules else None
            )
            daily_tag_spend_retention_seconds: Final = (
                self._retention_seconds_for("maximum_daily_tag_spend_retention_period") if age_rules else None
            )
            if (
                not delete_spend_logs
                and autorouter_retention_seconds is None
                and health_check_retention_seconds is None
                and daily_tag_spend_retention_seconds is None
                and active_size_rule is None
            ):
                SpendLogCleanupMetrics.record_run("skipped_disabled")
                return

            if delete_spend_logs and self.retention_seconds is None:
                verbose_proxy_logger.error("Retention seconds is None, cannot proceed with cleanup")
                SpendLogCleanupMetrics.record_run("skipped_disabled")
                return

            # If we have a pod lock manager, try to acquire the lock
            if self.pod_lock_manager and self.pod_lock_manager.redis_cache:
                lock_acquired = (
                    await self.pod_lock_manager.acquire_lock(
                        cronjob_id=SPEND_LOG_CLEANUP_JOB_NAME,
                    )
                    or False
                )
                verbose_proxy_logger.info(
                    "Lock acquisition attempt: %s  at %s", "successful" if lock_acquired else "failed", datetime.now()
                )

                if not lock_acquired:
                    verbose_proxy_logger.info("Another pod is already running cleanup")
                    SpendLogCleanupMetrics.record_run("skipped_locked")
                    return

            deadline: Final = time.monotonic() + self.run_budget_seconds
            size_group: Final = int(active_size_rule is not None)
            configured_group_count: Final = (
                int(delete_spend_logs and self.retention_seconds is not None)
                + int(autorouter_retention_seconds is not None)
                + int(health_check_retention_seconds is not None)
                + int(daily_tag_spend_retention_seconds is not None)
                + size_group
            )

            spend_log_results: Final = (
                await self._clean_spend_log_tables(
                    prisma_client,
                    self._group_deadline(deadline, configured_group_count),
                )
                if delete_spend_logs and self.retention_seconds is not None
                else ()
            )
            remaining_groups_after_spend_logs: Final = (
                int(autorouter_retention_seconds is not None)
                + int(health_check_retention_seconds is not None)
                + int(daily_tag_spend_retention_seconds is not None)
                + size_group
            )
            remaining_groups_after_sessions: Final = (
                int(health_check_retention_seconds is not None)
                + int(daily_tag_spend_retention_seconds is not None)
                + size_group
            )
            session_results: Final = (
                await self._clean_session_rollup(
                    prisma_client,
                    autorouter_retention_seconds,
                    self._group_deadline(deadline, remaining_groups_after_spend_logs),
                )
                if autorouter_retention_seconds is not None
                else ()
            )
            health_check_results: Final = (
                await self._clean_health_checks(
                    prisma_client,
                    health_check_retention_seconds,
                    self._group_deadline(deadline, remaining_groups_after_sessions),
                )
                if health_check_retention_seconds is not None
                else ()
            )
            daily_tag_spend_results: Final = (
                await self._clean_daily_tag_spend(
                    prisma_client,
                    daily_tag_spend_retention_seconds,
                    self._group_deadline(deadline, 1 + size_group),
                )
                if daily_tag_spend_retention_seconds is not None
                else ()
            )
            size_report: Final = (
                await self._clean_spend_logs_by_size(prisma_client, active_size_rule, deadline)
                if active_size_rule is not None
                else None
            )
            size_results: Final = size_report.results if size_report is not None else ()

            results: Final = (
                spend_log_results + session_results + health_check_results + daily_tag_spend_results + size_results
            )
            outcome: Final = self._run_outcome(results)
            SpendLogCleanupMetrics.record_run(outcome)
            self._log_run_summary(outcome, results, time.monotonic() - run_started_at, size_report)

        except asyncio.CancelledError:
            verbose_proxy_logger.error(
                "Spend log cleanup cancelled after %.2fs (rows_deleted=%d, batches=%d); the next run resumes from here",
                time.monotonic() - run_started_at,
                progress.rows_deleted,
                progress.batches,
            )
            SpendLogCleanupMetrics.record_run("aborted")
            raise
        except Exception as e:
            # .exception() captures the traceback; str(e) alone on a Prisma/DB
            # timeout is often empty and gives operators no signal to diagnose.
            verbose_proxy_logger.exception(
                "Error during spend log cleanup: %s: %s",
                type(e).__name__,
                e,
            )
            SpendLogCleanupMetrics.record_run("aborted")
            return  # Return after error handling
        finally:
            _run_progress.reset(progress_token)
            # Only release the lock if it was actually acquired
            if lock_acquired and self.pod_lock_manager and self.pod_lock_manager.redis_cache:
                await self.pod_lock_manager.release_lock(cronjob_id=SPEND_LOG_CLEANUP_JOB_NAME)
                verbose_proxy_logger.info("Released cleanup lock")
