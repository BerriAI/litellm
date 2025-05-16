from datetime import datetime, timedelta, UTC
from typing import Optional
from litellm.proxy.utils import PrismaClient
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm._logging import verbose_proxy_logger
from litellm.caching import RedisCache
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from litellm.constants import SPEND_LOG_CLEANUP_JOB_NAME
import os


class SpendLogCleanup:
    """
    Handles cleaning up old spend logs based on maximum retention period.
    Deletes logs in batches to prevent timeouts.
    Uses PodLockManager to ensure only one pod runs cleanup in multi-pod deployments.
    """

    def __init__(self, general_settings=None, redis_cache: Optional[RedisCache] = None):
        self.batch_size = 1000
        self.retention_seconds: Optional[int] = None
        from litellm.proxy.proxy_server import general_settings as default_settings
        self.general_settings = general_settings or default_settings
        from litellm.proxy.proxy_server import proxy_logging_obj

        pod_lock_manager = proxy_logging_obj.db_spend_update_writer.pod_lock_manager
        self.pod_lock_manager = pod_lock_manager
        verbose_proxy_logger.info(f"SpendLogCleanup initialized with batch size: {self.batch_size}")

    def _should_delete_spend_logs(self) -> bool:
        """
        Determines if logs should be deleted based on the max retention period in settings.
        """
        retention_setting = self.general_settings.get("maximum_spend_logs_retention_period")
        verbose_proxy_logger.info(f"Checking retention setting: {retention_setting}")

        if retention_setting is None:
            verbose_proxy_logger.info(f"No retention setting found")
            return False

        try:
            if isinstance(retention_setting, int):
                retention_setting = str(retention_setting)
            self.retention_seconds = duration_in_seconds(retention_setting)
            verbose_proxy_logger.info(f"Retention period set to {self.retention_seconds} seconds")
            return True
        except ValueError as e:
            verbose_proxy_logger.error(
                f"Invalid maximum_spend_logs_retention_period value: {retention_setting}, error: {str(e)}"
            )
            return False

    async def cleanup_old_spend_logs(self, prisma_client: PrismaClient) -> None:
        """
        Main cleanup function. Deletes old spend logs in batches.
        Only runs on the pod that acquires the distributed lock.
        """
        try:
            verbose_proxy_logger.info(f"Cleanup job triggered at {datetime.now()}")

            if not self._should_delete_spend_logs():
                verbose_proxy_logger.info(f"Skipping cleanup — invalid or missing retention setting.")
                return

            if self.retention_seconds is None:
                verbose_proxy_logger.error(f"Retention seconds is None, cannot proceed with cleanup")
                return

            # Check if pod_lock_manager and redis_cache exist
            if not self.pod_lock_manager or not self.pod_lock_manager.redis_cache:
                verbose_proxy_logger.info(f"Pod lock manager or redis cache not initialized, skipping cleanup")
                return

            # Try to acquire the distributed lock
            lock_acquired = await self.pod_lock_manager.acquire_lock(cronjob_id=SPEND_LOG_CLEANUP_JOB_NAME)
            verbose_proxy_logger.info(f"Lock acquisition attempt: {'successful' if lock_acquired else 'failed'}")
            
            if not lock_acquired:
                verbose_proxy_logger.info(f"Another pod is already running cleanup")
                return

            try:
                cutoff_date = datetime.now(UTC) - timedelta(seconds=float(self.retention_seconds))
                verbose_proxy_logger.info(f"Deleting logs older than {cutoff_date.isoformat()}")

                total_deleted = 0
                run_count = 0
                while True:
                    if run_count > 100:
                        verbose_proxy_logger.info(f"Max logs deleted - 1,00,000, rest of the logs will be deleted in next run")
                        break
                    # Step 1: Find logs to delete
                    logs_to_delete = await prisma_client.db.litellm_spendlogs.find_many(
                        where={"startTime": {"lt": cutoff_date}},
                        take=self.batch_size,
                    )
                    verbose_proxy_logger.info(f"🗑️ Found {len(logs_to_delete)} logs in this batch")

                    if not logs_to_delete:
                        verbose_proxy_logger.info(f"No more logs to delete. Total deleted: {total_deleted}")
                        break

                    request_ids = [log.request_id for log in logs_to_delete]

                    # Step 2: Delete them in one go
                    await prisma_client.db.litellm_spendlogs.delete_many(
                        where={"request_id": {"in": request_ids}}
                    )

                    total_deleted += len(logs_to_delete)
                    verbose_proxy_logger.info(f"Deleted {len(logs_to_delete)} logs in this batch")
                    run_count += 1

                # After cleanup is complete, release the lock and return
                await self.pod_lock_manager.release_lock(cronjob_id=SPEND_LOG_CLEANUP_JOB_NAME)
                verbose_proxy_logger.info(f"Released cleanup lock")
                return  # Explicitly return after cleanup is complete

            except Exception as e:
                verbose_proxy_logger.error(f"Error during cleanup: {str(e)}")
                # Ensure lock is released even if an error occurs
                await self.pod_lock_manager.release_lock(cronjob_id=SPEND_LOG_CLEANUP_JOB_NAME)
                verbose_proxy_logger.info(f"Released cleanup lock after error")
                return  # Return after error handling

        except Exception as e:
            verbose_proxy_logger.error(f"Error during cleanup: {str(e)}")
            return  # Return after error handling
