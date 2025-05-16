from datetime import datetime, timedelta, UTC
from typing import Optional
from litellm.proxy.utils import PrismaClient
from litellm.litellm_core_utils.duration_parser import duration_in_seconds
from litellm._logging import verbose_proxy_logger


class SpendLogCleanup:
    """
    Handles cleaning up old spend logs based on maximum retention period.
    Deletes logs in batches to prevent timeouts.
    """

    def __init__(self, general_settings=None):
        self.batch_size = 1000
        self.retention_seconds: Optional[int] = None
        from litellm.proxy.proxy_server import general_settings as default_settings
        self.general_settings = general_settings or default_settings
        verbose_proxy_logger.info("SpendLogCleanup initialized with batch size: %d", self.batch_size)

    def _should_delete_spend_logs(self) -> bool:
        """
        Determines if logs should be deleted based on the max retention period in settings.
        """
        retention_setting = self.general_settings.get("maximum_spend_logs_retention_period")
        verbose_proxy_logger.info("Checking retention setting: %s", retention_setting)

        if retention_setting is None:
            verbose_proxy_logger.info("No retention setting found")
            return False

        try:
            if isinstance(retention_setting, int):
                retention_setting = str(retention_setting)
            self.retention_seconds = duration_in_seconds(retention_setting)
            verbose_proxy_logger.info("Retention period set to %d seconds", self.retention_seconds)
            return True
        except ValueError as e:
            verbose_proxy_logger.error(
                f"Invalid maximum_spend_logs_retention_period value: {retention_setting}, error: {str(e)}"
            )
            return False

    async def cleanup_old_spend_logs(self, prisma_client: PrismaClient) -> None:
        """
        Main cleanup function. Deletes old spend logs in batches.
        """
        try:
            verbose_proxy_logger.info(f"Cleanup job triggered at {datetime.now()}")

            if not self._should_delete_spend_logs():
                verbose_proxy_logger.info("Skipping cleanup — invalid or missing retention setting.")
                return

            if self.retention_seconds is None:
                verbose_proxy_logger.error("Retention seconds is None, cannot proceed with cleanup")
                return

            cutoff_date = datetime.now(UTC) - timedelta(seconds=float(self.retention_seconds))
            verbose_proxy_logger.info(f"Deleting logs older than {cutoff_date.isoformat()}")

            total_deleted = 0
            run_count = 0
            while True:
                if run_count > 100:
                    verbose_proxy_logger.info("Max logs deleted - 1,00,000, rest of the logs will be deleted in next run")
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
        except Exception as e:
            verbose_proxy_logger.error(f"Error during cleanup: {str(e)}")
