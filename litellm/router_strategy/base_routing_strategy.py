"""
Base class across routing strategies to abstract commmon functions like batch incrementing redis
"""

import asyncio
import threading
from abc import ABC
from typing import List, Optional, Set, Union

from litellm._logging import verbose_router_logger
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisPipelineIncrementOperation
from litellm.constants import DEFAULT_REDIS_SYNC_INTERVAL


class BaseRoutingStrategy(ABC):
    def __init__(
        self,
        dual_cache: DualCache,
        should_batch_redis_writes: bool,
        default_sync_interval: Optional[Union[int, float]],
    ):
        self.dual_cache = dual_cache
        self.redis_increment_operation_queue: List[RedisPipelineIncrementOperation] = []
        if should_batch_redis_writes:
            try:
                # Try to get existing event loop
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop exists and is running, create task in existing loop
                    loop.create_task(
                        self.periodic_sync_in_memory_spend_with_redis(
                            default_sync_interval=default_sync_interval
                        )
                    )
                else:
                    self._create_sync_thread(default_sync_interval)
            except RuntimeError:  # No event loop in current thread
                self._create_sync_thread(default_sync_interval)

        self.in_memory_keys_to_update: set[
            str
        ] = set()  # Set with max size of 1000 keys

    async def _increment_value_in_current_window(
        self, key: str, value: Union[int, float], ttl: int
    ):
        """
        Increment spend within existing budget window

        Runs once the budget start time exists in Redis Cache (on the 2nd and subsequent requests to the same provider)

        - Increments the spend in memory cache (so spend instantly updated in memory)
        - Queues the increment operation to Redis Pipeline (using batched pipeline to optimize performance. Using Redis for multi instance environment of LiteLLM)
        """
        result = await self.dual_cache.in_memory_cache.async_increment(
            key=key,
            value=value,
            ttl=ttl,
        )
        increment_op = RedisPipelineIncrementOperation(
            key=key,
            increment_value=value,
            ttl=ttl,
        )
        self.redis_increment_operation_queue.append(increment_op)
        self.add_to_in_memory_keys_to_update(key=key)
        return result

    async def periodic_sync_in_memory_spend_with_redis(
        self, default_sync_interval: Optional[Union[int, float]]
    ):
        """
        Handler that triggers sync_in_memory_spend_with_redis every DEFAULT_REDIS_SYNC_INTERVAL seconds

        Required for multi-instance environment usage of provider budgets
        """
        default_sync_interval = default_sync_interval or DEFAULT_REDIS_SYNC_INTERVAL
        while True:
            try:
                await self._sync_in_memory_spend_with_redis()
                await asyncio.sleep(
                    default_sync_interval
                )  # Wait for DEFAULT_REDIS_SYNC_INTERVAL seconds before next sync
            except Exception as e:
                verbose_router_logger.error(f"Error in periodic sync task: {str(e)}")
                await asyncio.sleep(
                    default_sync_interval
                )  # Still wait DEFAULT_REDIS_SYNC_INTERVAL seconds on error before retrying

    async def _push_in_memory_increments_to_redis(self):
        """
        How this works:
        - async_log_success_event collects all provider spend increments in `redis_increment_operation_queue`
        - This function pushes all increments to Redis in a batched pipeline to optimize performance

        Only runs if Redis is initialized
        """
        try:
            if not self.dual_cache.redis_cache:
                return  # Redis is not initialized

            verbose_router_logger.debug(
                "Pushing Redis Increment Pipeline for queue: %s",
                self.redis_increment_operation_queue,
            )
            if len(self.redis_increment_operation_queue) > 0:
                asyncio.create_task(
                    self.dual_cache.redis_cache.async_increment_pipeline(
                        increment_list=self.redis_increment_operation_queue,
                    )
                )

            self.redis_increment_operation_queue = []

        except Exception as e:
            verbose_router_logger.error(
                f"Error syncing in-memory cache with Redis: {str(e)}"
            )
            self.redis_increment_operation_queue = []

    def add_to_in_memory_keys_to_update(self, key: str):
        self.in_memory_keys_to_update.add(key)

    def get_in_memory_keys_to_update(self) -> Set[str]:
        return self.in_memory_keys_to_update

    def reset_in_memory_keys_to_update(self):
        self.in_memory_keys_to_update = set()

    async def _sync_in_memory_spend_with_redis(self):
        """
        Ensures in-memory cache is updated with latest Redis values for all provider spends.

        Why Do we need this?
        - Optimization to hit sub 100ms latency. Performance was impacted when redis was used for read/write per request
        - Use provider budgets in multi-instance environment, we use Redis to sync spend across all instances

        What this does:
        1. Push all provider spend increments to Redis
        2. Fetch all current provider spend from Redis to update in-memory cache
        """

        try:
            # No need to sync if Redis cache is not initialized
            if self.dual_cache.redis_cache is None:
                return

            # 1. Push all provider spend increments to Redis
            await self._push_in_memory_increments_to_redis()

            # 2. Fetch all current provider spend from Redis to update in-memory cache
            cache_keys = self.get_in_memory_keys_to_update()

            cache_keys_list = list(cache_keys)

            # Batch fetch current spend values from Redis
            redis_values = await self.dual_cache.redis_cache.async_batch_get_cache(
                key_list=cache_keys_list
            )

            # Update in-memory cache with Redis values
            if isinstance(redis_values, dict):  # Check if redis_values is a dictionary
                for key, value in redis_values.items():
                    if value is not None:
                        await self.dual_cache.in_memory_cache.async_set_cache(
                            key=key, value=float(value)
                        )
                        verbose_router_logger.debug(
                            f"Updated in-memory cache for {key}: {value}"
                        )

            self.reset_in_memory_keys_to_update()
        except Exception as e:
            verbose_router_logger.exception(
                f"Error syncing in-memory cache with Redis: {str(e)}"
            )

    def _create_sync_thread(self, default_sync_interval):
        """Helper method to create a new thread for periodic sync"""
        thread = threading.Thread(
            target=asyncio.run,
            args=(
                self.periodic_sync_in_memory_spend_with_redis(
                    default_sync_interval=default_sync_interval
                ),
            ),
            daemon=True,
        )
        thread.start()
