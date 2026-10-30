import asyncio
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Final, TypeAlias

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
from litellm.proxy.db.db_transaction_queue.base_update_queue import service_logger_obj
from litellm.redis.stores import pod_lock
from litellm.redis.stores.pod_lock import Acquired, HeldBy, PodLockStore, TransportPodLockStore, Vacant
from litellm.redis.transport.contract import RedisFailure
from litellm.types.services import ServiceTypes

POD_LOCK_TARGET: Final = "pod_lock"

if TYPE_CHECKING:
    from litellm.proxy.utils import ProxyLogging as _ProxyLogging

    ProxyLogging: TypeAlias = _ProxyLogging
else:
    ProxyLogging = Any


class PodLockManager:
    """
    Manager for acquiring and releasing locks for cron jobs using Redis.

    Ensures that only one pod can run a cron job at a time.
    """

    def __init__(self, redis_cache: RedisCache | None = None):
        self.pod_id = str(uuid.uuid4())
        self.redis_cache = redis_cache

    @staticmethod
    def get_redis_lock_key(cronjob_id: str) -> str:
        return pod_lock.lock_key(cronjob_id)

    def _store(self) -> PodLockStore | None:
        return None if self.redis_cache is None else TransportPodLockStore(self.redis_cache.transport())

    @with_service_target(POD_LOCK_TARGET)
    async def acquire_lock(
        self,
        cronjob_id: str,
        ttl: int | None = None,
        allow_reentrant: bool = True,
    ) -> bool | None:
        """
        Attempt to acquire the lock for a specific cron job using Redis.

        Args:
            cronjob_id: The ID of the cron job to lock
            ttl: Optional custom TTL in seconds. Defaults to DEFAULT_CRON_JOB_LOCK_TTL_SECONDS.
                 Use a longer TTL for jobs that may take longer than the default 60s
                 (e.g. key rotation with many keys).
            allow_reentrant: With the default True, a pod that already holds the lock
                 acquires it again (leader election semantics). Pass False when the live
                 lock marks work as already done for this window, so not even the holder
                 may redo it before the TTL expires.
        """
        store: Final = self._store()
        if store is None:
            verbose_proxy_logger.debug("redis_cache is None, skipping acquire_lock")
            return None
        lock_ttl: Final = ttl or DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
        verbose_proxy_logger.debug(
            "Pod %s attempting to acquire Redis lock for cronjob_id=%s (ttl=%ds)",
            self.pod_id,
            cronjob_id,
            lock_ttl,
        )
        match await store.acquire(cronjob_id, self.pod_id, ttl=timedelta(seconds=lock_ttl)):
            case Acquired():
                verbose_proxy_logger.info(
                    "Pod %s successfully acquired Redis lock for cronjob_id=%s", self.pod_id, cronjob_id
                )
                return True
            case HeldBy(owner=owner) if owner == self.pod_id and allow_reentrant:
                verbose_proxy_logger.info(
                    "Pod %s already holds the Redis lock for cronjob_id=%s", self.pod_id, cronjob_id
                )
                self._emit_acquired_lock_event(cronjob_id, self.pod_id)
                return True
            case HeldBy(owner=owner):
                verbose_proxy_logger.info(
                    "Pod %s could not acquire lock for cronjob_id=%s, held by pod %s.", self.pod_id, cronjob_id, owner
                )
                return False
            case Vacant() | RedisFailure():
                return False

    @with_service_target(POD_LOCK_TARGET)
    async def lock_holder(self, cronjob_id: str) -> HeldBy | None | RedisFailure:
        store: Final = self._store()
        return None if store is None else await store.holder(cronjob_id)

    @with_service_target(POD_LOCK_TARGET)
    async def is_held(self, cronjob_id: str) -> bool:
        """An unreadable lock reports as unheld, so callers run a job twice rather than skip it."""
        match await self.lock_holder(cronjob_id):
            case HeldBy():
                return True
            case None:
                return False
            case RedisFailure(message=message):
                verbose_proxy_logger.warning("Could not read the lock for cronjob_id=%s: %s", cronjob_id, message)
                return False

    @with_service_target(POD_LOCK_TARGET)
    async def release_lock(
        self,
        cronjob_id: str,
    ) -> None:
        """
        Release the lock if the current pod holds it, with an atomic compare-and-delete
        so a stale owner can never delete a lock another pod has since taken.
        """
        store: Final = self._store()
        if store is None:
            verbose_proxy_logger.debug("redis_cache is None, skipping release_lock")
            return
        verbose_proxy_logger.debug("Pod %s attempting to release Redis lock for cronjob_id=%s", self.pod_id, cronjob_id)
        released: Final = await self._release(store, cronjob_id)
        if isinstance(released, RedisFailure):
            verbose_proxy_logger.error("Error releasing Redis lock for %s: %s", cronjob_id, released.message)
        elif released:
            verbose_proxy_logger.info(
                "Pod %s successfully released Redis lock for cronjob_id=%s", self.pod_id, cronjob_id
            )
            self._emit_released_lock_event(cronjob_id=cronjob_id, pod_id=self.pod_id)
        else:
            verbose_proxy_logger.debug(
                "Pod %s failed to release Redis lock for cronjob_id=%s (lock missing or held by another pod)",
                self.pod_id,
                cronjob_id,
            )

    async def _release(self, store: PodLockStore, cronjob_id: str) -> bool | RedisFailure:
        released: Final = await store.release(cronjob_id, self.pod_id)
        if not isinstance(released, RedisFailure):
            return released
        verbose_proxy_logger.warning(
            "Lua compare-and-delete failed for lock_key=%s, falling back to GET+DEL",
            self.get_redis_lock_key(cronjob_id),
        )
        match await store.holder(cronjob_id):
            case RedisFailure(kind="breaker_open") as refused:
                return refused
            case HeldBy(owner=owner) if owner == self.pod_id:
                return await store.remove(cronjob_id)
            case _:
                return False

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
