import asyncio
import json
import time
from typing import Any, Dict, List, Optional, Tuple

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_cache import RedisCache
from litellm.constants import (
    DEFAULT_SHARED_HEALTH_CHECK_TTL,
    DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL,
)
from litellm.proxy.health_check import perform_health_check


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
        redis_cache: Optional[RedisCache] = None,
        health_check_ttl: int = DEFAULT_SHARED_HEALTH_CHECK_TTL,
        lock_ttl: int = DEFAULT_SHARED_HEALTH_CHECK_LOCK_TTL,
    ):
        self.redis_cache = redis_cache
        self.health_check_ttl = health_check_ttl
        self.lock_ttl = lock_ttl
        self.pod_id = f"pod_{int(time.time() * 1000)}"

    @staticmethod
    def get_health_check_lock_key() -> str:
        """Get the Redis key for health check lock."""
        return "health_check_lock"

    @staticmethod
    def get_health_check_cache_key() -> str:
        """Get the Redis key for health check results cache."""
        return "health_check_results"

    @staticmethod
    def get_model_health_check_lock_key(model_name: str) -> str:
        """Get the Redis key for model-specific health check lock."""
        return f"health_check_lock:{model_name}"

    @staticmethod
    def get_model_health_check_cache_key(model_name: str) -> str:
        """Get the Redis key for model-specific health check results cache."""
        return f"health_check_results:{model_name}"

    async def acquire_health_check_lock(self) -> bool:
        """
        Attempt to acquire the global health check lock.
        
        Returns:
            bool: True if lock was acquired, False otherwise
        """
        if self.redis_cache is None:
            verbose_proxy_logger.debug("redis_cache is None, skipping lock acquisition")
            return False

        try:
            lock_key = self.get_health_check_lock_key()
            acquired = await self.redis_cache.async_set_cache(
                lock_key,
                self.pod_id,
                nx=True,  # Only set if key doesn't exist
                ttl=self.lock_ttl,
            )
            
            if acquired:
                verbose_proxy_logger.info(
                    "Pod %s acquired health check lock", self.pod_id
                )
            else:
                verbose_proxy_logger.debug(
                    "Pod %s failed to acquire health check lock", self.pod_id
                )
            
            return acquired
        except Exception as e:
            verbose_proxy_logger.error(
                "Error acquiring health check lock: %s", str(e)
            )
            return False

    async def release_health_check_lock(self) -> None:
        """Release the global health check lock."""
        if self.redis_cache is None:
            return

        try:
            lock_key = self.get_health_check_lock_key()
            # Only release if we own the lock
            current_owner = await self.redis_cache.async_get_cache(lock_key)
            if current_owner == self.pod_id:
                await self.redis_cache.async_delete_cache(lock_key)
                verbose_proxy_logger.info(
                    "Pod %s released health check lock", self.pod_id
                )
        except Exception as e:
            verbose_proxy_logger.error(
                "Error releasing health check lock: %s", str(e)
            )

    async def get_cached_health_check_results(self) -> Optional[Dict[str, Any]]:
        """
        Get cached health check results from Redis.
        
        Returns:
            Optional[Dict]: Cached health check results or None if not found/expired
        """
        if self.redis_cache is None:
            return None

        try:
            cache_key = self.get_health_check_cache_key()
            cached_data = await self.redis_cache.async_get_cache(cache_key)
            
            if cached_data is None:
                return None

            # Parse the cached data
            if isinstance(cached_data, str):
                cached_results = json.loads(cached_data)
            else:
                cached_results = cached_data

            # Check if the cache is still valid
            cache_timestamp = cached_results.get("timestamp", 0)
            current_time = time.time()
            
            if current_time - cache_timestamp > self.health_check_ttl:
                verbose_proxy_logger.debug("Cached health check results expired")
                return None

            verbose_proxy_logger.debug("Using cached health check results")
            return cached_results

        except Exception as e:
            verbose_proxy_logger.error(
                "Error getting cached health check results: %s", str(e)
            )
            return None

    async def cache_health_check_results(
        self, 
        healthy_endpoints: List[Dict[str, Any]], 
        unhealthy_endpoints: List[Dict[str, Any]]
    ) -> None:
        """
        Cache health check results in Redis.
        
        Args:
            healthy_endpoints: List of healthy endpoints
            unhealthy_endpoints: List of unhealthy endpoints
        """
        if self.redis_cache is None:
            return

        try:
            cache_data = {
                "healthy_endpoints": healthy_endpoints,
                "unhealthy_endpoints": unhealthy_endpoints,
                "healthy_count": len(healthy_endpoints),
                "unhealthy_count": len(unhealthy_endpoints),
                "timestamp": time.time(),
                "checked_by": self.pod_id,
            }

            cache_key = self.get_health_check_cache_key()
            await self.redis_cache.async_set_cache(
                cache_key,
                json.dumps(cache_data),
                ttl=self.health_check_ttl,
            )
            
            verbose_proxy_logger.info(
                "Cached health check results for %d healthy and %d unhealthy endpoints",
                len(healthy_endpoints),
                len(unhealthy_endpoints),
            )

        except Exception as e:
            verbose_proxy_logger.error(
                "Error caching health check results: %s", str(e)
            )

    async def perform_shared_health_check(
        self, 
        model_list: List[Dict[str, Any]], 
        details: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
            
        Returns:
            Tuple of (healthy_endpoints, unhealthy_endpoints)
        """
        # First, try to get cached results
        cached_results = await self.get_cached_health_check_results()
        if cached_results is not None:
            return (
                cached_results.get("healthy_endpoints", []),
                cached_results.get("unhealthy_endpoints", []),
            )

        # No recent cache, try to acquire lock
        lock_acquired = await self.acquire_health_check_lock()
        
        if lock_acquired:
            try:
                # We have the lock, run health check
                verbose_proxy_logger.info(
                    "Pod %s running health check for %d models", 
                    self.pod_id, 
                    len(model_list)
                )
                
                healthy_endpoints, unhealthy_endpoints = await perform_health_check(
                    model_list=model_list, details=details
                )
                
                # Cache the results
                await self.cache_health_check_results(
                    healthy_endpoints, unhealthy_endpoints
                )
                
                return healthy_endpoints, unhealthy_endpoints
                
            finally:
                # Always release the lock
                await self.release_health_check_lock()
        else:
            # Lock not acquired, wait briefly and try to get cached results
            verbose_proxy_logger.debug(
                "Pod %s waiting for other pod to complete health check", self.pod_id
            )
            
            # Wait a bit for the other pod to complete
            await asyncio.sleep(2)
            
            # Try to get cached results again
            cached_results = await self.get_cached_health_check_results()
            if cached_results is not None:
                return (
                    cached_results.get("healthy_endpoints", []),
                    cached_results.get("unhealthy_endpoints", []),
                )
            
            # Still no cache, fall back to local health check
            verbose_proxy_logger.warning(
                "Pod %s falling back to local health check (no cache available)", 
                self.pod_id
            )
            
            return await perform_health_check(model_list=model_list, details=details)

    async def is_health_check_in_progress(self) -> bool:
        """
        Check if a health check is currently in progress by another pod.
        
        Returns:
            bool: True if health check is in progress, False otherwise
        """
        if self.redis_cache is None:
            return False

        try:
            lock_key = self.get_health_check_lock_key()
            current_owner = await self.redis_cache.async_get_cache(lock_key)
            return current_owner is not None and current_owner != self.pod_id
        except Exception as e:
            verbose_proxy_logger.error(
                "Error checking health check lock status: %s", str(e)
            )
            return False

    async def get_health_check_status(self) -> Dict[str, Any]:
        """
        Get the current status of health check coordination.
        
        Returns:
            Dict containing status information
        """
        status = {
            "pod_id": self.pod_id,
            "redis_available": self.redis_cache is not None,
            "lock_ttl": self.lock_ttl,
            "cache_ttl": self.health_check_ttl,
        }

        if self.redis_cache is not None:
            try:
                # Check if there's a current lock
                lock_key = self.get_health_check_lock_key()
                current_owner = await self.redis_cache.async_get_cache(lock_key)
                status["lock_owner"] = current_owner
                status["lock_in_progress"] = current_owner is not None

                # Check cache status
                cached_results = await self.get_cached_health_check_results()
                status["cache_available"] = cached_results is not None
                if cached_results:
                    status["cache_age_seconds"] = time.time() - cached_results.get("timestamp", 0)
                    status["last_checked_by"] = cached_results.get("checked_by")

            except Exception as e:
                status["error"] = str(e)

        return status
