import asyncio
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta
from typing import TYPE_CHECKING, Final

from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL,
    DEFAULT_SHARED_HEALTH_CHECK_TTL,
)
from litellm.proxy.health_check import perform_health_check
from litellm.redis.stores.shared_health import (
    CachedHealthCheckResults,
    SharedHealthStore,
    TransportSharedHealthStore,
    lock_key,
    model_lock_key,
    model_results_key,
    results_key,
)
from litellm.redis.transport.contract import RedisFailure
from litellm.router_utils.health_state_cache import HEALTH_CHECKS_TARGET

if TYPE_CHECKING:
    from litellm.router import Router


class SharedHealthCheckManager:
    """
    Manager for coordinating health checks across multiple pods using Redis.

    This class implements a shared health check state mechanism that:
    - Prevents duplicate health checks across pods
    - Caches health check results with configurable TTL
    - Uses Redis locks to ensure only one pod runs health checks at a time
    - Allows other pods to read cached results instead of running redundant checks
    """

    def __init__(
        self,
        redis_cache: RedisCache | None = None,
        health_check_ttl: int = DEFAULT_SHARED_HEALTH_CHECK_TTL,
        lock_ttl: int = DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL,
    ):
        self.redis_cache = redis_cache
        self._store: SharedHealthStore | None = (
            TransportSharedHealthStore(redis_cache.transport()) if redis_cache is not None else None
        )
        self.health_check_ttl = health_check_ttl
        self.lock_ttl = lock_ttl
        self.pod_id = f"pod_{int(time.time() * 1000)}"

    @staticmethod
    def get_health_check_lock_key() -> str:
        """Get the Redis key for health check lock."""
        return lock_key()

    @staticmethod
    def get_health_check_cache_key() -> str:
        """Get the Redis key for health check results cache."""
        return results_key()

    @staticmethod
    def get_model_health_check_lock_key(model_name: str) -> str:
        """Get the Redis key for model-specific health check lock."""
        return model_lock_key(model_name)

    @staticmethod
    def get_model_health_check_cache_key(model_name: str) -> str:
        """Get the Redis key for model-specific health check results cache."""
        return model_results_key(model_name)

    @with_service_target(HEALTH_CHECKS_TARGET)
    async def acquire_health_check_lock(self) -> bool:
        """
        Attempt to acquire the global health check lock.

        Returns:
            bool: True if lock was acquired, False otherwise
        """
        if self._store is None:
            verbose_proxy_logger.debug("redis_cache is None, skipping lock acquisition")
            return False

        try:
            acquired: Final = await self._store.acquire(self.pod_id, ttl=timedelta(seconds=self.lock_ttl))
            if isinstance(acquired, RedisFailure):
                return False

            if acquired:
                verbose_proxy_logger.info("Pod %s acquired health check lock", self.pod_id)
            else:
                verbose_proxy_logger.debug("Pod %s failed to acquire health check lock", self.pod_id)

            return bool(acquired)
        except Exception:
            return False

    @with_service_target(HEALTH_CHECKS_TARGET)
    async def release_health_check_lock(self) -> None:
        """Release the global health check lock."""
        if self._store is None:
            return

        try:
            released: Final = await self._store.release(self.pod_id)
            if released is True:
                verbose_proxy_logger.info("Pod %s released health check lock", self.pod_id)
        except Exception:
            return

    @with_service_target(HEALTH_CHECKS_TARGET)
    async def get_cached_health_check_results(self) -> CachedHealthCheckResults | None:
        """
        Get cached health check results from Redis.

        Returns:
            Optional[Dict]: Cached health check results or None if not found/expired
        """
        if self._store is None:
            return None

        try:
            cached_results: Final = await self._store.get()
            if cached_results is None or isinstance(cached_results, RedisFailure):
                return None

            # Check if the cache is still valid
            cache_timestamp: Final = cached_results.get("timestamp", 0)
            current_time: Final = time.time()

            if current_time - cache_timestamp > self.health_check_ttl:
                verbose_proxy_logger.debug("Cached health check results expired")
                return None

            verbose_proxy_logger.debug("Using cached health check results")
            return cached_results

        except Exception:
            return None

    @with_service_target(HEALTH_CHECKS_TARGET)
    async def cache_health_check_results(
        self,
        healthy_endpoints: Sequence[Mapping[str, object]],
        unhealthy_endpoints: Sequence[Mapping[str, object]],
    ) -> None:
        """
        Cache health check results in Redis.

        Args:
            healthy_endpoints: List of healthy endpoints
            unhealthy_endpoints: List of unhealthy endpoints
        """
        if self._store is None:
            return

        try:
            cache_data: Final = {
                "healthy_endpoints": healthy_endpoints,
                "unhealthy_endpoints": unhealthy_endpoints,
                "healthy_count": len(healthy_endpoints),
                "unhealthy_count": len(unhealthy_endpoints),
                "timestamp": time.time(),
                "checked_by": self.pod_id,
            }

            written: Final = await self._store.put(cache_data, ttl=timedelta(seconds=self.health_check_ttl))
            if isinstance(written, RedisFailure):
                return

            verbose_proxy_logger.info(
                "Cached health check results for %d healthy and %d unhealthy endpoints",
                len(healthy_endpoints),
                len(unhealthy_endpoints),
            )

        except Exception:
            return

    @with_service_target(HEALTH_CHECKS_TARGET)
    async def perform_shared_health_check(
        self,
        model_list: list[Mapping[str, object]],
        details: bool = True,
        max_concurrency: int | None = None,
        health_check_skip_disabled_background_models: bool = False,
        router: "Router | None" = None,
    ) -> tuple[list[Mapping[str, object]], list[Mapping[str, object]], dict[str, object]]:
        """
        Perform health check with shared state coordination.

        This method:
        1. First checks if there are recent cached results
        2. If no recent cache, tries to acquire lock to run health check
        3. If lock acquired, runs health check and caches results
        4. If lock not acquired, waits briefly and tries to get cached results again
        5. Falls back to running health check locally if no cache available

        Args:
            model_list: List of models to check
            details: Whether to include detailed information
            max_concurrency: Optional limit on concurrent health check requests
            health_check_skip_disabled_background_models: Remove models with disable_background_health_check: true

        Returns:
            Tuple of (healthy_endpoints, unhealthy_endpoints)
        """
        # First, try to get cached results
        cached_results = await self.get_cached_health_check_results()
        if cached_results is not None:
            return (
                cached_results.get("healthy_endpoints", []),
                cached_results.get("unhealthy_endpoints", []),
                {},
            )

        # No recent cache, try to acquire lock
        lock_acquired: Final = await self.acquire_health_check_lock()

        if lock_acquired:
            try:
                # We have the lock, run health check
                verbose_proxy_logger.info(
                    "Pod %s running health check for %d models",
                    self.pod_id,
                    len(model_list),
                )

                (
                    healthy_endpoints,
                    unhealthy_endpoints,
                    exceptions_by_model_id,
                ) = await perform_health_check(
                    model_list=model_list,
                    details=details,
                    max_concurrency=max_concurrency,
                    health_check_skip_disabled_background_models=health_check_skip_disabled_background_models,
                    router=router,
                )

                # Cache the results
                await self.cache_health_check_results(healthy_endpoints, unhealthy_endpoints)

                return healthy_endpoints, unhealthy_endpoints, exceptions_by_model_id

            finally:
                # Always release the lock
                await self.release_health_check_lock()
        else:
            # If Redis is not configured, skip polling — there is no cache
            # to wait for.
            if self._store is None:
                return await perform_health_check(
                    model_list=model_list,
                    details=details,
                    max_concurrency=max_concurrency,
                    health_check_skip_disabled_background_models=health_check_skip_disabled_background_models,
                    router=router,
                )

            # Lock not acquired — poll for cached results until the lock
            # holder finishes or the lock expires, rather than falling back
            # to a redundant local health check after only 2 seconds.
            verbose_proxy_logger.debug("Pod %s waiting for other pod to complete health check", self.pod_id)

            poll_interval: Final = 5  # seconds between cache checks
            max_wait: Final = self.lock_ttl  # wait at most as long as the lock can live
            elapsed = 0

            while elapsed < max_wait:
                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

                cached_results = await self.get_cached_health_check_results()
                if cached_results is not None:
                    verbose_proxy_logger.info(
                        "Pod %s using cached health check results after waiting %ds",
                        self.pod_id,
                        elapsed,
                    )
                    return (
                        cached_results.get("healthy_endpoints", []),
                        cached_results.get("unhealthy_endpoints", []),
                        {},
                    )

                # Check if the lock is still held — if it was released without
                # caching (e.g. the holder crashed), stop waiting early.
                try:
                    if (
                        isinstance(current_owner := await self._store.owner(), RedisFailure)
                        and current_owner.kind == "breaker_open"
                    ):
                        continue
                    if current_owner is None or isinstance(current_owner, RedisFailure):
                        verbose_proxy_logger.debug(
                            "Pod %s detected lock released without cache, stopping wait",
                            self.pod_id,
                        )
                        break
                except Exception:
                    # Redis hiccup — continue polling rather than crashing out
                    pass

            # Exhausted wait — fall back to local health check
            verbose_proxy_logger.warning(
                "Pod %s falling back to local health check after waiting %ds (no cache available)",
                self.pod_id,
                elapsed,
            )

            return await perform_health_check(
                model_list=model_list,
                details=details,
                max_concurrency=max_concurrency,
                health_check_skip_disabled_background_models=health_check_skip_disabled_background_models,
                router=router,
            )

    @with_service_target(HEALTH_CHECKS_TARGET)
    async def is_health_check_in_progress(self) -> bool:
        """
        Check if a health check is currently in progress by another pod.

        Returns:
            bool: True if health check is in progress, False otherwise
        """
        if self._store is None:
            return False

        try:
            current_owner: Final = await self._store.owner()
            if isinstance(current_owner, RedisFailure):
                return False
            return current_owner is not None and current_owner.value != self.pod_id
        except Exception:
            return False

    @with_service_target(HEALTH_CHECKS_TARGET)
    async def get_health_check_status(self) -> dict[str, object]:
        """
        Get the current status of health check coordination.

        Returns:
            Dict containing status information
        """
        status: Final[dict[str, object]] = {
            "pod_id": self.pod_id,
            "redis_available": self.redis_cache is not None,
            "lock_ttl": self.lock_ttl,
            "cache_ttl": self.health_check_ttl,
        }

        if self._store is not None:
            try:
                owner: Final = await self._store.owner()
                if isinstance(owner, RedisFailure) and owner.kind == "breaker_open":
                    return {**status, "error": owner.message}
                current_owner: Final = None if owner is None or isinstance(owner, RedisFailure) else owner.value
                status["lock_owner"] = current_owner
                status["lock_in_progress"] = current_owner is not None

                # Check cache status
                cached_results: Final = await self.get_cached_health_check_results()
                status["cache_available"] = cached_results is not None
                if cached_results:
                    status["cache_age_seconds"] = time.time() - cached_results.get("timestamp", 0)
                    status["last_checked_by"] = cached_results.get("checked_by")

            except Exception as e:
                status["error"] = str(e)

        return status
