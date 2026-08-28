import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final, Literal, TypeAlias

from pydantic import BaseModel, TypeAdapter

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
)
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.db.db_transaction_queue.spend_log_cleanup_metrics import (
    RunOutcome,
    SpendLogCleanupMetrics,
)
from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import (
    RemainingTimeoutMs,
    SpendLogsPartitionManager,
)
from litellm.proxy.utils import PrismaClient

StopReason: TypeAlias = Literal["exhausted", "budget_exhausted", "batch_cap_reached", "aborted"]


@dataclass(frozen=True, slots=True)
class TableCleanupResult:
    """Outcome of pruning one table, so the caller can report why a run ended."""

    rows_deleted: int
    stop_reason: StopReason


class _RemainingRow(BaseModel):
    """One row of the capped outstanding-rows probe, validated out of prisma's untyped result."""

    remaining: int


_REMAINING_ROWS: Final = TypeAdapter(list[_RemainingRow])

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
        self, prisma_client: PrismaClient, delete_sql: str, cutoff_date: datetime, deadline: float
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
        async with prisma_client.db.tx() as tx:
            await tx.execute_raw(f"SET LOCAL statement_timeout = {timeout_ms}")
            await tx.execute_raw(f"SET LOCAL lock_timeout = {timeout_ms}")
            deleted_result: Final = await tx.execute_raw(delete_sql, cutoff_date, self.batch_size)
        return deleted_result if isinstance(deleted_result, int) else None

    async def _count_remaining(
        self, prisma_client: PrismaClient, cutoff_date: datetime, table_name: str, time_column: str, deadline: float
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
                WHERE "{time_column}" < $1::timestamptz
                LIMIT $2
            ) capped
            """
        try:
            async with prisma_client.db.tx() as tx:
                await tx.execute_raw(f"SET LOCAL statement_timeout = {self._timeout_ms(deadline)}")
                rows: Final = _REMAINING_ROWS.validate_python(
                    await tx.query_raw(count_sql, cutoff_date, SPEND_LOG_CLEANUP_REMAINING_COUNT_CAP)
                )
        except Exception as e:  # noqa: BLE001 - an observability probe must never fail the cleanup run
            verbose_proxy_logger.warning("Could not count remaining %s rows: %s", table_name, e)
            return None
        return rows[0].remaining if rows else None

    async def _delete_old_rows_batched(
        self,
        prisma_client: PrismaClient,
        cutoff_date: datetime,
        table_name: str,
        key_columns: tuple[str, ...],
        time_column: str,
        deadline: float,
    ) -> TableCleanupResult:
        """
        Delete a table's rows older than the cutoff in batches.

        Stops at whichever bound is reached first: the backlog running out, the
        shared wall-clock deadline, the per-table batch cap, or too many
        consecutive batch failures.
        """
        key_list: Final = ", ".join(f'"{col}"' for col in key_columns)
        delete_sql: Final = f"""
            DELETE FROM "{table_name}"
            WHERE ({key_list}) IN (
                SELECT {key_list} FROM "{table_name}"
                WHERE "{time_column}" < $1::timestamptz
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
            if run_count >= self.max_batches:
                verbose_proxy_logger.info(
                    "Max batches reached for %s cleanup, remaining rows will be deleted in next run", table_name
                )
                return await self._finish_table(
                    prisma_client, cutoff_date, table_name, time_column, total_deleted, "batch_cap_reached", deadline
                )
            # Find rows and delete them in one go without fetching to application
            batch_started_at = time.monotonic()
            try:
                batch_result = await self._execute_delete_batch(prisma_client, delete_sql, cutoff_date, deadline)
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
                    cutoff_date.isoformat(),
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

            # Add a small sleep to prevent overwhelming the database
            await asyncio.sleep(0.1)

    async def _finish_table(
        self,
        prisma_client: PrismaClient,
        cutoff_date: datetime,
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
            return TableCleanupResult(rows_deleted=rows_deleted, stop_reason=stop_reason)
        remaining: Final = await self._count_remaining(prisma_client, cutoff_date, table_name, time_column, deadline)
        if remaining is not None:
            SpendLogCleanupMetrics.set_rows_remaining(table_name, remaining)
        return TableCleanupResult(rows_deleted=rows_deleted, stop_reason=stop_reason)

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

        logs_result: Final = await self._delete_old_logs(prisma_client, cutoff_date, deadline)
        verbose_proxy_logger.info("Deleted %s logs", logs_result.rows_deleted)

        index_result: Final = await self._delete_old_tool_index_rows(prisma_client, cutoff_date, deadline)
        verbose_proxy_logger.info("Deleted %s expired tool index rows", index_result.rows_deleted)
        return (logs_result, index_result)

    async def _clean_session_rollup(
        self, prisma_client: PrismaClient, retention_seconds: int, deadline: float
    ) -> tuple[TableCleanupResult, ...]:
        """
        Prune auto-router session rollup rows, which carry their own retention horizon.
        """
        session_cutoff: Final = datetime.now(timezone.utc) - timedelta(seconds=float(retention_seconds))
        sessions_result: Final = await self._delete_old_autorouter_session_rows(prisma_client, session_cutoff, deadline)
        verbose_proxy_logger.info("Deleted %s expired auto-router session rollup rows", sessions_result.rows_deleted)
        return (sessions_result,)

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

    async def cleanup_old_spend_logs(self, prisma_client: PrismaClient) -> None:
        """
        Main cleanup function. Deletes old spend logs in batches.
        If pod_lock_manager is available, ensures only one pod runs cleanup.
        If no pod_lock_manager, runs cleanup without distributed locking.
        """
        lock_acquired = False
        try:
            verbose_proxy_logger.info("Cleanup job triggered at %s", datetime.now())
            self._refresh_bounds()

            delete_spend_logs: Final = self._should_delete_spend_logs()
            autorouter_retention_seconds: Final = self._retention_seconds_for(
                "maximum_autorouter_session_retention_period"
            )
            health_check_retention_seconds: Final = self._retention_seconds_for("maximum_health_check_retention_period")
            if (
                not delete_spend_logs
                and autorouter_retention_seconds is None
                and health_check_retention_seconds is None
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
            configured_group_count: Final = (
                int(delete_spend_logs and self.retention_seconds is not None)
                + int(autorouter_retention_seconds is not None)
                + int(health_check_retention_seconds is not None)
            )

            spend_log_results: Final = (
                await self._clean_spend_log_tables(
                    prisma_client,
                    self._group_deadline(deadline, configured_group_count),
                )
                if delete_spend_logs and self.retention_seconds is not None
                else ()
            )
            remaining_groups_after_spend_logs: Final = int(autorouter_retention_seconds is not None) + int(
                health_check_retention_seconds is not None
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
                    deadline,
                )
                if health_check_retention_seconds is not None
                else ()
            )

            SpendLogCleanupMetrics.record_run(
                self._run_outcome(spend_log_results + session_results + health_check_results)
            )

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
            # Only release the lock if it was actually acquired
            if lock_acquired and self.pod_lock_manager and self.pod_lock_manager.redis_cache:
                await self.pod_lock_manager.release_lock(cronjob_id=SPEND_LOG_CLEANUP_JOB_NAME)
                verbose_proxy_logger.info("Released cleanup lock")
