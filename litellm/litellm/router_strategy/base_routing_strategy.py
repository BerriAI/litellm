"""
Base class across routing strategies to abstract commmon functions like batch incrementing redis
"""

import asyncio
from abc import ABC
from typing import Dict, List, Optional, Set, Tuple, Union

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
        self._sync_task: Optional[asyncio.Task[None]] = None
        if should_batch_redis_writes:
            self.setup_sync_task(default_sync_interval)

        self.in_memory_keys_to_update: set[
            str
        ] = set()  # Set with max size of 1000 keys

    def setup_sync_task(self, default_sync_interval: Optional[Union[int, float]]):
        """Setup the sync task in a way that's compatible with FastAPI"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self._sync_task = loop.create_task(
            self.periodic_sync_in_memory_spend_with_redis(
                default_sync_interval=default_sync_interval
            )
        )

    async def cleanup(self):
        """Cleanup method to be called when shutting down"""
        if self._sync_task is not None:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

    async def _increment_value_list_in_current_window(
        self, increment_list: List[Tuple[str, int]], ttl: int
    ) -> List[float]:
        """
        Increment a list of values in the current window
        """
        results = []
        for key, value in increment_list:
            result = await self._increment_value_in_current_window(
                key=key, value=value, ttl=ttl
            )
            results.append(result)
        return results

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
        - This function compresses multiple increments for the same key into a single operation
        - Then pushes all increments to Redis in a batched pipeline to optimize performance

        Only runs if Redis is initialized
        """
        try:
            if not self.dual_cache.redis_cache:
                return  # Redis is not initialized

            if len(self.redis_increment_operation_queue) > 0:
                # Compress operations for the same key
                compressed_ops: Dict[str, RedisPipelineIncrementOperation] = {}
                ops_to_remove = []
                for idx, op in enumerate(self.redis_increment_operation_queue):
                    if op["key"] in compressed_ops:
                        # Add to existing increment
                        compressed_ops[op["key"]]["increment_value"] += op[
                            "increment_value"
                        ]
                    else:
                        compressed_ops[op["key"]] = op

                    ops_to_remove.append(idx)

                # Convert back to list
                compressed_queue = list(compressed_ops.values())

                increment_result = (
                    await self.dual_cache.redis_cache.async_increment_pipeline(
                        increment_list=compressed_queue,
                    )
                )

                self.redis_increment_operation_queue = [
                    op
                    for idx, op in enumerate(self.redis_increment_operation_queue)
                    if idx not in ops_to_remove
                ]

                if increment_result is not None:
                    return_result = {
                        key["key"]: op
                        for key, op in zip(compressed_queue, increment_result)
                    }
                else:
                    return_result = {}
                return return_result

        except Exception as e:
            verbose_router_logger.error(
                f"Error syncing in-memory cache with Redis: {str(e)}"
            )
            self.redis_increment_operation_queue = []

    def add_to_in_memory_keys_to_update(self, key: str):
        self.in_memory_keys_to_update.add(key)

    def get_key_pattern_to_sync(self) -> Optional[str]:
        """
        Get the key pattern to sync
        """
        return None

    def get_in_memory_keys_to_update(self) -> Set[str]:
        return self.in_memory_keys_to_update

    def get_and_reset_in_memory_keys_to_update(self) -> Set[str]:
        """Atomic get and reset in-memory keys to update"""
        keys = self.in_memory_keys_to_update
        self.in_memory_keys_to_update = set()
        return keys

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

            # 2. Fetch all current provider spend from Redis to update in-memory cache
            cache_keys = (
                self.get_in_memory_keys_to_update()
            )  # if no pattern OR redis cache does not support scan_iter, use in-memory keys

            cache_keys_list = list(cache_keys)

            # 1. Snapshot in-memory before
            in_memory_before_dict = {}
            in_memory_before = (
                await self.dual_cache.in_memory_cache.async_batch_get_cache(
                    keys=cache_keys_list
                )
            )
            for k, v in zip(cache_keys_list, in_memory_before):
                in_memory_before_dict[k] = float(v or 0)

            # 1. Push all provider spend increments to Redis
            redis_values = await self._push_in_memory_increments_to_redis()
            if redis_values is None:
                return

            # 4. Merge
            for key in cache_keys_list:
                redis_val = float(redis_values.get(key, 0) or 0)
                before = float(in_memory_before_dict.get(key, 0) or 0)
                after = float(
                    await self.dual_cache.in_memory_cache.async_get_cache(key=key) or 0
                )
                delta = after - before
                if after <= redis_val:
                    merged = redis_val + delta
                else:
                    continue
                # elif "rpm" in key:  # redis is behind in-memory cache
                #     # shut down the proxy
                #     print(f"self.redis_increment_operation_queue: {self.redis_increment_operation_queue}")
                #     print(f"Redis_val={redis_val} is behind in-memory cache_val={after} for key: {key}. This should not happen, since we should be updating redis with in-memory cache.")
                #     import os
                #     os._exit(1)
                #     raise Exception(f"Redis is behind in-memory cache for key: {key}. This should not happen, since we should be updating redis with in-memory cache.")
                await self.dual_cache.in_memory_cache.async_set_cache(
                    key=key, value=merged
                )

        except Exception as e:
            verbose_router_logger.exception(
                f"Error syncing in-memory cache with Redis: {str(e)}"
            )
