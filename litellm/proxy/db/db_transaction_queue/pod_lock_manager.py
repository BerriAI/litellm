import asyncio
import uuid
from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
from litellm.proxy.db.db_transaction_queue.base_update_queue import service_logger_obj
from litellm.types.services import ServiceTypes

if TYPE_CHECKING:
    ProxyLogging = Any
else:
    ProxyLogging = Any


class PodLockManager:
    """
    Manager for acquiring and releasing locks for cron jobs using Redis.

    Ensures that only one pod can run a cron job at a time.
    """

    def __init__(self, redis_cache: Optional[RedisCache] = None):
        self.pod_id = str(uuid.uuid4())
        self.redis_cache = redis_cache

    @staticmethod
    def get_redis_lock_key(cronjob_id: str) -> str:
        return f"cronjob_lock:{cronjob_id}"

    async def acquire_lock(
        self,
        cronjob_id: str,
    ) -> Optional[bool]:
        """
        Attempt to acquire the lock for a specific cron job using Redis.
        Uses the SET command with NX and EX options to ensure atomicity.
        
        Args:
            cronjob_id: The ID of the cron job to lock
        """
        if self.redis_cache is None:
            verbose_proxy_logger.debug("redis_cache is None, skipping acquire_lock")
            return None
        try:
            verbose_proxy_logger.debug(
                "Pod %s attempting to acquire Redis lock for cronjob_id=%s",
                self.pod_id,
                cronjob_id,
            )
            # Try to set the lock key with the pod_id as its value, only if it doesn't exist (NX)
            # and with an expiration (EX) to avoid deadlocks.
            lock_key = PodLockManager.get_redis_lock_key(cronjob_id)
            acquired = await self.redis_cache.async_set_cache(
                lock_key,
                self.pod_id,
                nx=True,
                ttl=DEFAULT_CRON_JOB_LOCK_TTL_SECONDS,
            )
            if acquired:
                verbose_proxy_logger.info(
                    "Pod %s successfully acquired Redis lock for cronjob_id=%s",
                    self.pod_id,
                    cronjob_id,
                )

                return True
            else:
                # Check if the current pod already holds the lock
                current_value = await self.redis_cache.async_get_cache(lock_key)
                if current_value is not None:
                    if isinstance(current_value, bytes):
                        current_value = current_value.decode("utf-8")
                    if current_value == self.pod_id:
                        verbose_proxy_logger.info(
                            "Pod %s already holds the Redis lock for cronjob_id=%s",
                            self.pod_id,
                            cronjob_id,
                        )
                        self._emit_acquired_lock_event(cronjob_id, self.pod_id)
                        return True
            return False
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error acquiring Redis lock for {cronjob_id}: {e}"
            )
            return False

    async def release_lock(
        self,
        cronjob_id: str,
    ):
        """
        Release the lock if the current pod holds it.
        Uses get and delete commands to ensure that only the owner can release the lock.
        """
        if self.redis_cache is None:
            verbose_proxy_logger.debug("redis_cache is None, skipping release_lock")
            return
        try:
            cronjob_id = cronjob_id
            verbose_proxy_logger.debug(
                "Pod %s attempting to release Redis lock for cronjob_id=%s",
                self.pod_id,
                cronjob_id,
            )
            lock_key = PodLockManager.get_redis_lock_key(cronjob_id)

            current_value = await self.redis_cache.async_get_cache(lock_key)
            if current_value is not None:
                if isinstance(current_value, bytes):
                    current_value = current_value.decode("utf-8")
                if current_value == self.pod_id:
                    result = await self.redis_cache.async_delete_cache(lock_key)
                    if result == 1:
                        verbose_proxy_logger.info(
                            "Pod %s successfully released Redis lock for cronjob_id=%s",
                            self.pod_id,
                            cronjob_id,
                        )
                        self._emit_released_lock_event(
                            cronjob_id=cronjob_id,
                            pod_id=self.pod_id,
                        )
                    else:
                        verbose_proxy_logger.debug(
                            "Pod %s failed to release Redis lock for cronjob_id=%s",
                            self.pod_id,
                            cronjob_id,
                        )
                else:
                    verbose_proxy_logger.debug(
                        "Pod %s cannot release Redis lock for cronjob_id=%s because it is held by pod %s",
                        self.pod_id,
                        cronjob_id,
                        current_value,
                    )
            else:
                verbose_proxy_logger.debug(
                    "Pod %s attempted to release Redis lock for cronjob_id=%s, but no lock was found",
                    self.pod_id,
                    cronjob_id,
                )
        except Exception as e:
            verbose_proxy_logger.error(
                f"Error releasing Redis lock for {cronjob_id}: {e}"
            )

    @staticmethod
    def _emit_acquired_lock_event(cronjob_id: str, pod_id: str):
        asyncio.create_task(
            service_logger_obj.async_service_success_hook(
                service=ServiceTypes.POD_LOCK_MANAGER,
                duration=DEFAULT_CRON_JOB_LOCK_TTL_SECONDS,
                call_type="_emit_acquired_lock_event",
                event_metadata={
                    "gauge_labels": f"{cronjob_id}:{pod_id}",
                    "gauge_value": 1,
                },
            )
        )

    @staticmethod
    def _emit_released_lock_event(cronjob_id: str, pod_id: str):
        asyncio.create_task(
            service_logger_obj.async_service_success_hook(
                service=ServiceTypes.POD_LOCK_MANAGER,
                duration=DEFAULT_CRON_JOB_LOCK_TTL_SECONDS,
                call_type="_emit_released_lock_event",
                event_metadata={
                    "gauge_labels": f"{cronjob_id}:{pod_id}",
                    "gauge_value": 0,
                },
            )
        )
