import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.constants import DEFAULT_CRON_JOB_LOCK_TTL_SECONDS

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient, ProxyLogging
else:
    PrismaClient = Any
    ProxyLogging = Any


class PodLockManager:
    """
    Manager for acquiring and releasing locks for cron jobs.

    Ensures that only one pod can run a cron job at a time.
    """

    def __init__(self, cronjob_id: str):
        self.pod_id = str(uuid.uuid4())
        self.cronjob_id = cronjob_id

    async def acquire_lock(self) -> bool:
        """
        Attempt to acquire the lock for a specific cron job.
        """
        from litellm.proxy.proxy_server import prisma_client

        verbose_proxy_logger.debug("acquiring lock for cronjob_id=%s", self.cronjob_id)
        if not prisma_client:
            verbose_proxy_logger.debug("prisma is None, returning False")
            return False
        try:
            current_time = datetime.now(timezone.utc)
            # Lease expiry time
            ttl_expiry = current_time + timedelta(
                seconds=DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
            )

            # Attempt to acquire the lock by upserting the record in the `cronjob_locks` table
            cronjob_lock = await prisma_client.db.cronjob.upsert(
                where={"cronjob_id": self.cronjob_id},
                data={
                    "create": {
                        "cronjob_id": self.cronjob_id,
                        "pod_id": self.pod_id,
                        "status": "ACTIVE",
                        "last_updated": current_time,
                        "ttl": ttl_expiry,
                    },
                    "update": {
                        "status": "ACTIVE",
                        "last_updated": current_time,
                        "ttl": ttl_expiry,
                    },
                },
            )
            verbose_proxy_logger.debug("cronjob_lock=%s", cronjob_lock)

            if cronjob_lock.status == "ACTIVE" and cronjob_lock.pod_id == self.pod_id:
                verbose_proxy_logger.debug(
                    f"Pod {self.pod_id} has acquired the lock for {self.cronjob_id}."
                )
                return True  # Lock successfully acquired
            return False
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error acquiring the lock for {self.cronjob_id}: {e}"
            )
            return False

    async def renew_lock(self):
        """
        Renew the lock (update the TTL) for the pod holding the lock.
        """
        from litellm.proxy.proxy_server import prisma_client

        if not prisma_client:
            return False
        try:
            verbose_proxy_logger.debug(
                "renewing lock for cronjob_id=%s", self.cronjob_id
            )
            current_time = datetime.now(timezone.utc)
            # Extend the TTL for another DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
            ttl_expiry = current_time + timedelta(
                seconds=DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
            )

            await prisma_client.db.cronjob.update(
                where={"cronjob_id": self.cronjob_id, "pod_id": self.pod_id},
                data={"ttl": ttl_expiry, "last_updated": current_time},
            )
            verbose_proxy_logger.info(
                f"Renewed the lock for Pod {self.pod_id} for {self.cronjob_id}"
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error renewing the lock for {self.cronjob_id}: {e}"
            )

    async def release_lock(self):
        """
        Release the lock and mark the pod as inactive.
        """
        from litellm.proxy.proxy_server import prisma_client

        if not prisma_client:
            return False
        try:
            verbose_proxy_logger.debug(
                "releasing lock for cronjob_id=%s", self.cronjob_id
            )
            await prisma_client.db.cronjob.update(
                where={"cronjob_id": self.cronjob_id, "pod_id": self.pod_id},
                data={"status": "INACTIVE"},
            )
            verbose_proxy_logger.info(
                f"Pod {self.pod_id} has released the lock for {self.cronjob_id}."
            )
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error releasing the lock for {self.cronjob_id}: {e}"
            )
