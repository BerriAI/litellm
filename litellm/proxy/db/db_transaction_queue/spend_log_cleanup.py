import asyncio
from datetime import datetime, timedelta, timezone
from typing import Final

from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.constants import (
    SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS,
    SPEND_LOG_CLEANUP_BATCH_SIZE,
    SPEND_LOG_CLEANUP_JOB_NAME,
    SPEND_LOG_CLEANUP_MAX_CONSECUTIVE_BATCH_FAILURES,
    SPEND_LOG_RUN_LOOPS,
)
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm.proxy.db.db_transaction_queue.spend_logs_partition_manager import (
    SpendLogsPartitionManager,
)
from litellm.proxy.utils import PrismaClient


class SpendLogCleanup:
    """
    Handles cleaning up old spend logs based on maximum retention period.

    When LiteLLM_SpendLogs is range-partitioned, expired data is reclaimed by
    dropping whole partitions (instant, frees disk immediately). Otherwise it
    falls back to deleting logs in batches.
    Uses PodLockManager to ensure only one pod runs cleanup in multi-pod deployments.
    """

    def __init__(
        self,
        general_settings=None,
        redis_cache: RedisCache | None = None,
        partition_manager: SpendLogsPartitionManager | None = None,
    ):
        self.batch_size = SPEND_LOG_CLEANUP_BATCH_SIZE
        self.retention_seconds: int | None = None
        self.partition_manager = partition_manager or SpendLogsPartitionManager()
        from litellm.proxy.proxy_server import general_settings as default_settings

        self.general_settings = general_settings or default_settings
        from litellm.proxy.proxy_server import proxy_logging_obj

        pod_lock_manager: Final = proxy_logging_obj.db_spend_update_writer.pod_lock_manager
        self.pod_lock_manager = pod_lock_manager
        verbose_proxy_logger.info("SpendLogCleanup initialized with batch size: %s", self.batch_size)

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

    async def _delete_old_rows_batched(
        self,
        prisma_client: PrismaClient,
        cutoff_date: datetime,
        table_name: str,
        key_columns: tuple[str, ...],
        time_column: str,
    ) -> int:
        """
        Helper method to delete a table's rows older than the cutoff in batches.
        Returns the total number of rows deleted.
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
            if run_count > SPEND_LOG_RUN_LOOPS:
                verbose_proxy_logger.info(
                    "Max batches reached for %s cleanup, remaining rows will be deleted in next run", table_name
                )
                break
            # Step 1: Find rows and delete them in one go without fetching to application
            # Delete in batches, limited by self.batch_size
            try:
                deleted_result = await prisma_client.db.execute_raw(
                    delete_sql,
                    cutoff_date,
                    self.batch_size,
                )
            except Exception as batch_exc:
                # A single batch failure (e.g. Prisma/DB timeout) must not abort
                # the whole run — subsequent batches may still succeed.
                consecutive_failures += 1
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
                    break
                await asyncio.sleep(SPEND_LOG_CLEANUP_BATCH_FAILURE_BACKOFF_SECONDS)
                continue

            consecutive_failures = 0

            deleted_count = 0
            if isinstance(deleted_result, int):
                deleted_count = deleted_result
            else:
                verbose_proxy_logger.error(
                    "Unexpected execute_raw return type for %s cleanup: %s; aborting cleanup to avoid infinite loop",
                    table_name,
                    type(deleted_result),
                )
                break

            verbose_proxy_logger.info("Deleted %s %s rows in this batch", deleted_count, table_name)

            if deleted_count == 0:
                verbose_proxy_logger.info("No more %s rows to delete. Total deleted: %s", table_name, total_deleted)
                break

            total_deleted += deleted_count
            run_count += 1

            # Add a small sleep to prevent overwhelming the database
            await asyncio.sleep(0.1)

        return total_deleted

    async def _delete_old_logs(self, prisma_client: PrismaClient, cutoff_date: datetime) -> int:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_SpendLogs",
            key_columns=("request_id", "startTime"),
            time_column="startTime",
        )

    async def _delete_old_tool_index_rows(self, prisma_client: PrismaClient, cutoff_date: datetime) -> int:
        # SpendLogToolIndex rows are derived from spend logs, so they expire on the
        # same cutoff; rows older than retention point at already-deleted logs.
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_SpendLogToolIndex",
            key_columns=("request_id", "tool_name"),
            time_column="start_time",
        )

    async def _delete_old_autorouter_session_rows(self, prisma_client: PrismaClient, cutoff_date: datetime) -> int:
        return await self._delete_old_rows_batched(
            prisma_client,
            cutoff_date,
            table_name="LiteLLM_AutoRouterSession",
            key_columns=("api_key", "session_id", "router_name"),
            time_column="last_turn_at",
        )

    async def cleanup_old_spend_logs(self, prisma_client: PrismaClient) -> None:
        """
        Main cleanup function. Deletes old spend logs in batches.
        If pod_lock_manager is available, ensures only one pod runs cleanup.
        If no pod_lock_manager, runs cleanup without distributed locking.
        """
        lock_acquired = False
        try:
            verbose_proxy_logger.info("Cleanup job triggered at %s", datetime.now())

            delete_spend_logs: Final = self._should_delete_spend_logs()
            autorouter_retention_seconds: Final = self._retention_seconds_for(
                "maximum_autorouter_session_retention_period"
            )
            if not delete_spend_logs and autorouter_retention_seconds is None:
                return

            if delete_spend_logs and self.retention_seconds is None:
                verbose_proxy_logger.error("Retention seconds is None, cannot proceed with cleanup")
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
                    return

            if delete_spend_logs and self.retention_seconds is not None:
                cutoff_date: Final = datetime.now(timezone.utc) - timedelta(seconds=float(self.retention_seconds))
                verbose_proxy_logger.info("Removing logs older than %s", cutoff_date.isoformat())

                if self.general_settings.get(
                    "use_spend_logs_partitioning", False
                ) and await self.partition_manager.is_partitioned(prisma_client):
                    await self.partition_manager.ensure_partitions(prisma_client)
                    dropped: Final = await self.partition_manager.drop_partitions_older_than(prisma_client, cutoff_date)
                    verbose_proxy_logger.info(
                        "Dropped %d expired spend-log partitions: %s",
                        len(dropped),
                        dropped,
                    )
                    # DROP only reclaims whole expired partitions. Expired rows can
                    # still sit in the DEFAULT partition (backfill, coverage gaps)
                    # or in a partition that spans the cutoff, so retention must
                    # also delete those stragglers row-wise.
                    total_deleted = await self._delete_old_logs(prisma_client, cutoff_date)
                    verbose_proxy_logger.info(
                        "Deleted %s expired logs not covered by dropped partitions", total_deleted
                    )
                else:
                    total_deleted = await self._delete_old_logs(prisma_client, cutoff_date)
                    verbose_proxy_logger.info("Deleted %s logs", total_deleted)

                index_deleted: Final = await self._delete_old_tool_index_rows(prisma_client, cutoff_date)
                verbose_proxy_logger.info("Deleted %s expired tool index rows", index_deleted)

            if autorouter_retention_seconds is not None:
                session_cutoff: Final = datetime.now(timezone.utc) - timedelta(
                    seconds=float(autorouter_retention_seconds)
                )
                sessions_deleted: Final = await self._delete_old_autorouter_session_rows(prisma_client, session_cutoff)
                verbose_proxy_logger.info("Deleted %s expired auto-router session rollup rows", sessions_deleted)

        except Exception as e:
            # .exception() captures the traceback; str(e) alone on a Prisma/DB
            # timeout is often empty and gives operators no signal to diagnose.
            verbose_proxy_logger.exception(
                "Error during spend log cleanup: %s: %s",
                type(e).__name__,
                e,
            )
            return  # Return after error handling
        finally:
            # Only release the lock if it was actually acquired
            if lock_acquired and self.pod_lock_manager and self.pod_lock_manager.redis_cache:
                await self.pod_lock_manager.release_lock(cronjob_id=SPEND_LOG_CLEANUP_JOB_NAME)
                verbose_proxy_logger.info("Released cleanup lock")
