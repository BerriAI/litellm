import uuid
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any

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
        Attempt to acquire the lock for a specific cron job using database locking.
        """
        from litellm.proxy.proxy_server import prisma_client

        verbose_proxy_logger.debug(
            "Pod %s acquiring lock for cronjob_id=%s", self.pod_id, self.cronjob_id
        )
        if not prisma_client:
            verbose_proxy_logger.debug("prisma is None, returning False")
            return False

        try:
            current_time = datetime.now(timezone.utc)
            ttl_expiry = current_time + timedelta(
                seconds=DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
            )

            # Use Prisma's findUnique with FOR UPDATE lock to prevent race conditions
            lock_record = await prisma_client.db.litellm_cronjob.find_unique(
                where={"cronjob_id": self.cronjob_id},
            )

            if lock_record:
                # If record exists, only update if it's inactive or expired
                if lock_record.status == "ACTIVE" and lock_record.ttl > current_time:
                    return lock_record.pod_id == self.pod_id

                # Update existing record
                updated_lock = await prisma_client.db.litellm_cronjob.update(
                    where={"cronjob_id": self.cronjob_id},
                    data={
                        "pod_id": self.pod_id,
                        "status": "ACTIVE",
                        "last_updated": current_time,
                        "ttl": ttl_expiry,
                    },
                )
            else:
                # Create new record if none exists
                updated_lock = await prisma_client.db.litellm_cronjob.create(
                    data={
                        "cronjob_id": self.cronjob_id,
                        "pod_id": self.pod_id,
                        "status": "ACTIVE",
                        "last_updated": current_time,
                        "ttl": ttl_expiry,
                    }
                )

            return updated_lock.pod_id == self.pod_id

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

            await prisma_client.db.litellm_cronjob.update(
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
                "Pod %s releasing lock for cronjob_id=%s", self.pod_id, self.cronjob_id
            )
            await prisma_client.db.litellm_cronjob.update(
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
