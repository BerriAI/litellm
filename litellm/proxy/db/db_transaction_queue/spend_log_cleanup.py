import asyncio
import logging
from datetime import datetime, timedelta, UTC
import os
from typing import Optional
from litellm.proxy.utils import PrismaClient
from litellm.litellm_core_utils.duration_parser import duration_in_seconds

logger = logging.getLogger(__name__)

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

    def _should_delete_spend_logs(self) -> bool:
        """
        Determines if logs should be deleted based on the max retention period in settings.
        """
        retention_setting = self.general_settings.get("maximum_spend_logs_retention_period")

        if retention_setting is None:
            return False

        try:
            if isinstance(retention_setting, int):
                retention_setting = str(retention_setting)
            self.retention_seconds = duration_in_seconds(retention_setting)
            return True
        except ValueError as e:
            logger.error(
                f"Invalid maximum_spend_logs_retention_period value: {retention_setting}, error: {str(e)}"
            )
            return False

    async def cleanup_old_spend_logs(self, prisma_client: PrismaClient) -> None:
        """
        Main cleanup function. Deletes old spend logs in batches.
        """
        try:
            print("Cleanup DB URL:", os.environ.get("DATABASE_URL"))

            print("Cleanup job triggered at", datetime.now())

            if not self._should_delete_spend_logs():
                print("Skipping cleanup — invalid or missing retention setting.")
                return

            cutoff_date = datetime.now(UTC) - timedelta(seconds=self.retention_seconds)
            print(f"🧹 Deleting logs older than {cutoff_date.isoformat()}")

            total_deleted = 0
            while True:
                old_logs = await prisma_client.db.litellm_spendlogs.find_many(
                    where={"startTime": {"lt": cutoff_date}},
                    take=self.batch_size,
                )

                if not old_logs:
                    print(f"✅ No more logs to delete. Total deleted: {total_deleted}")
                    break

                for log in old_logs:
                    await prisma_client.db.litellm_spendlogs.delete(
                        where={"request_id": log.request_id}
                    )
                    total_deleted += 1

                print(f"🗑️ Deleted {len(old_logs)} logs in this batch")

        except Exception as e:
            print(f"🔥 Error during cleanup: {str(e)}")
