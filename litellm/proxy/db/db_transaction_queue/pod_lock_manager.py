import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from contextlib import suppress
from typing import TYPE_CHECKING, Any, Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm._uuid import uuid
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
from litellm.proxy.db.db_transaction_queue.base_update_queue import service_logger_obj
from litellm.types.services import ServiceTypes

if TYPE_CHECKING:
    ProxyLogging = Any
else:
    ProxyLogging = Any


class _RegisteredScript(Protocol):
    """A Lua script already registered with Redis, as returned by async_register_script."""

    def __call__(self, keys: Sequence[str], args: Sequence[object]) -> Awaitable[object]: ...


class PodLockManager:
    """
    Manager for acquiring and releasing locks for cron jobs using Redis.

    Ensures that only one pod can run a cron job at a time.
    """

    _COMPARE_AND_DELETE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

    _COMPARE_AND_EXPIRE_LOCK_SCRIPT = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("expire", KEYS[1], ARGV[2])
else
    return 0
end
"""

    def __init__(self, redis_cache: RedisCache | None = None):
        self.pod_id = str(uuid.uuid4())
        self.redis_cache = redis_cache
        self._release_lock_script: _RegisteredScript | None = None
        self._renew_lock_script: _RegisteredScript | None = None

    @staticmethod
    def get_redis_lock_key(cronjob_id: str) -> str:
        return f"cronjob_lock:{cronjob_id}"

    async def acquire_lock(
        self,
        cronjob_id: str,
        ttl: int | None = None,
        allow_reentrant: bool = True,
    ) -> bool | None:
        """
        Attempt to acquire the lock for a specific cron job using Redis.
        Uses the SET command with NX and EX options to ensure atomicity.

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
        if self.redis_cache is None:
            verbose_proxy_logger.debug("redis_cache is None, skipping acquire_lock")
            return None
        try:
            lock_ttl: Final = ttl or DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
            verbose_proxy_logger.debug(
                "Pod %s attempting to acquire Redis lock for cronjob_id=%s (ttl=%ds)",
                self.pod_id,
                cronjob_id,
                lock_ttl,
            )
            # Try to set the lock key with the pod_id as its value, only if it doesn't exist (NX)
            # and with an expiration (EX) to avoid deadlocks.
            lock_key: Final = PodLockManager.get_redis_lock_key(cronjob_id)
            acquired: Final = await self.redis_cache.async_set_cache(
                lock_key,
                self.pod_id,
                nx=True,
                ttl=lock_ttl,
            )
            if acquired:
                verbose_proxy_logger.info(
                    "Pod %s successfully acquired Redis lock for cronjob_id=%s",
                    self.pod_id,
                    cronjob_id,
                )
                self._emit_acquired_lock_event(cronjob_id, self.pod_id)
                return True
            else:
                # Check if the current pod already holds the lock
                current_value = await self.redis_cache.async_get_cache(lock_key)
                if current_value is not None:
                    if isinstance(current_value, bytes):
                        current_value = current_value.decode("utf-8")
                    if current_value == self.pod_id and allow_reentrant:
                        verbose_proxy_logger.info(
                            "Pod %s already holds the Redis lock for cronjob_id=%s",
                            self.pod_id,
                            cronjob_id,
                        )
                        self._emit_acquired_lock_event(cronjob_id, self.pod_id)
                        return True
                    verbose_proxy_logger.info(
                        "Pod %s could not acquire lock for cronjob_id=%s, held by pod %s.",
                        self.pod_id,
                        cronjob_id,
                        current_value,
                    )
            return False
        except Exception as e:
            verbose_proxy_logger.error("Error acquiring Redis lock for %s: %s", cronjob_id, e)
            return False

    async def renew_lock(self, cronjob_id: str, ttl: int | None = None) -> bool:
        """Extend this pod's lock by another TTL, if it still owns it.

        Lets a lease outlive a run that takes longer than its TTL without making
        the TTL a failover deadline for the whole job. Returns False when the
        lock is gone or another pod took it, so the caller learns it is no
        longer the owner rather than extending someone else's lease.

        Renewal has no non-atomic fallback, unlike release. A GET-then-SET would
        write unconditionally, so a lease that lapsed and was taken over between
        the two calls would be handed back to the pod that lost it, leaving that
        pod and its successor both believing they own the job. Where the
        compare-and-expire cannot run, this reports False and the lease is left
        to expire into the failover it already describes.
        """
        cache: Final = self.redis_cache
        if cache is None:
            verbose_proxy_logger.debug("redis_cache is None, skipping renew_lock")
            return False
        lock_ttl: Final = ttl or DEFAULT_CRON_JOB_LOCK_TTL_SECONDS
        lock_key: Final = PodLockManager.get_redis_lock_key(cronjob_id)

        result, self._renew_lock_script = await self._act_if_owner(
            cache=cache,
            lock_key=lock_key,
            script=self._COMPARE_AND_EXPIRE_LOCK_SCRIPT,
            script_handle=self._renew_lock_script,
            script_args=(lock_ttl,),
            fallback=None,
        )
        if result:
            verbose_proxy_logger.debug(
                "Pod %s renewed Redis lock for cronjob_id=%s (ttl=%ds)",
                self.pod_id,
                cronjob_id,
                lock_ttl,
            )
        return bool(result)

    async def release_lock(
        self,
        cronjob_id: str,
    ):
        """
        Release the lock if the current pod holds it.
        Uses an atomic Lua compare-and-delete to prevent TOCTOU races where a
        stale owner could delete a newly reacquired lock.
        Falls back to GET + DEL for cache implementations that don't support
        script registration.
        """
        cache: Final = self.redis_cache
        if cache is None:
            verbose_proxy_logger.debug("redis_cache is None, skipping release_lock")
            return
        try:
            verbose_proxy_logger.debug(
                "Pod %s attempting to release Redis lock for cronjob_id=%s",
                self.pod_id,
                cronjob_id,
            )
            lock_key: Final = PodLockManager.get_redis_lock_key(cronjob_id)
            result: Final = await self._compare_and_delete_lock(cache=cache, lock_key=lock_key)
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
                    "Pod %s failed to release Redis lock for cronjob_id=%s (lock missing or held by another pod)",
                    self.pod_id,
                    cronjob_id,
                )
        except Exception as e:
            verbose_proxy_logger.error("Error releasing Redis lock for %s: %s", cronjob_id, e)

    async def _compare_and_delete_lock(self, cache: RedisCache, lock_key: str) -> int:
        """
        Atomically delete lock key only if current pod owns it.

        Falls back to get/delete for non-RedisCache implementations that do not
        expose Lua script registration.
        """

        async def _delete() -> int:
            return int(await cache.async_delete_cache(lock_key) or 0)

        result, self._release_lock_script = await self._act_if_owner(
            cache=cache,
            lock_key=lock_key,
            script=self._COMPARE_AND_DELETE_LOCK_SCRIPT,
            script_handle=self._release_lock_script,
            script_args=(),
            fallback=_delete,
        )
        return result

    async def _act_if_owner(
        self,
        *,
        cache: RedisCache,
        lock_key: str,
        script: str,
        script_handle: _RegisteredScript | None,
        script_args: Sequence[object],
        fallback: Callable[[], Awaitable[int]] | None,
    ) -> tuple[int, _RegisteredScript | None]:
        """Act on the lock only if this pod still owns it, atomically where possible.

        Returns the script's result and the handle to cache for next time, which is
        None whenever the GET-then-act fallback answered instead, so a Redis restart
        that cleared the loaded scripts re-registers rather than failing forever.

        ``fallback`` is None for actions with no safe non-atomic form, which report
        0 rather than racing the lock's owner.
        """
        script_register: Final = getattr(cache, "async_register_script", None)
        if callable(script_register):
            with suppress(Exception):
                # acquire_lock stores the pod_id via async_set_cache, which
                # JSON-encodes the value; compare against the same encoding so
                # the Lua equality check matches
                handle: Final = script_handle or script_register(script)
                result: Final = await handle(keys=(lock_key,), args=(json.dumps(self.pod_id), *script_args))
                return int(result or 0), handle
            # scripting is disabled, or a Redis restart cleared the loaded scripts
            verbose_proxy_logger.warning(
                "Lua compare-and-act failed for lock_key=%s, falling back to GET then act",
                lock_key,
            )

        if fallback is None:
            return 0, None

        current_value = await cache.async_get_cache(lock_key)
        if isinstance(current_value, bytes):
            current_value = current_value.decode("utf-8")
        if current_value != self.pod_id:
            return 0, None
        return await fallback(), None

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
