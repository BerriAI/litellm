"""
Base class across routing strategies to abstract commmon functions like batch incrementing redis
"""

import asyncio
import logging
from abc import ABC
from typing import Final

from litellm._internal_context import with_service_target
from litellm._logging import verbose_router_logger
from litellm.caching.redis_cache import log_redis_failure
from litellm.constants import DEFAULT_REDIS_SYNC_INTERVAL
from litellm.redis.stores.router_usage import ROUTER_USAGE_TARGET, CounterIncrement, RouterUsage, UsageCounter


class BaseRoutingStrategy(ABC):
    def __init__(
        self,
        usage: RouterUsage,
        should_batch_redis_writes: bool,
        default_sync_interval: float | None,
    ):
        self.usage = usage
        self.redis_increment_operation_queue: list[CounterIncrement] = []
        self._sync_task: asyncio.Task[None] | None = None
        self._retired = False
        self._flush_after_retire: asyncio.Task[None] | None = None
        if should_batch_redis_writes:
            self.setup_sync_task(default_sync_interval)

        self.in_memory_keys_to_update: set[UsageCounter] = set()  # Set with max size of 1000 keys

    def setup_sync_task(self, default_sync_interval: float | None):
        """Setup the sync task in a way that's compatible with FastAPI"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        self._sync_task = loop.create_task(
            self.periodic_sync_in_memory_spend_with_redis(default_sync_interval=default_sync_interval)
        )

    def cancel_sync_task(self) -> None:
        if self._sync_task is not None:
            self._sync_task.cancel()

    def retire(self) -> None:
        self.cancel_sync_task()
        self._retired = True
        self._schedule_flush_after_retire()

    def _schedule_flush_after_retire(self) -> None:
        if self._flush_after_retire is not None and not self._flush_after_retire.done():
            return
        if not self.redis_increment_operation_queue or not self.usage.has_redis:
            return
        try:
            loop: Final = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._flush_after_retire = loop.create_task(self._drain_increments_after_retire())

    async def _drain_increments_after_retire(self) -> None:
        if not self.redis_increment_operation_queue:
            return
        await self._push_in_memory_increments_to_redis()
        await self._drain_increments_after_retire()

    async def cleanup(self):
        """Cleanup method to be called when shutting down"""
        if self._sync_task is not None:
            self.cancel_sync_task()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass

    async def _increment_value_in_current_window(self, counter: UsageCounter, value: float, ttl: int):
        """
        Increment spend within existing budget window

        Runs once the budget start time exists in Redis Cache (on the 2nd and subsequent requests to the same provider)

        - Increments the spend in memory cache (so spend instantly updated in memory)
        - Queues the increment operation to Redis Pipeline (using batched pipeline to optimize performance. Using Redis for multi instance environment of LiteLLM)
        """
        increment_op: Final = CounterIncrement(counter=counter, amount=value, ttl=ttl)
        result: Final = self.usage.increment_local(increment_op)

        self.redis_increment_operation_queue.append(increment_op)
        self.add_to_in_memory_keys_to_update(counter)
        if self._retired:
            self._schedule_flush_after_retire()
        return result

    @with_service_target(ROUTER_USAGE_TARGET)
    async def periodic_sync_in_memory_spend_with_redis(self, default_sync_interval: float | None):
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
                verbose_router_logger.error("Error in periodic sync task: %s", e)
                await asyncio.sleep(
                    default_sync_interval
                )  # Still wait DEFAULT_REDIS_SYNC_INTERVAL seconds on error before retrying

    @with_service_target(ROUTER_USAGE_TARGET)
    async def _push_in_memory_increments_to_redis(self):
        """
        How this works:
        - async_log_success_event collects all provider spend increments in `redis_increment_operation_queue`
        - This function compresses multiple increments for the same key into a single operation
        - Then pushes all increments to Redis in a batched pipeline to optimize performance

        Only runs if Redis is initialized
        """
        try:
            if not self.usage.has_redis:
                return  # Redis is not initialized

            if len(self.redis_increment_operation_queue) > 0:
                queued: Final = tuple(self.redis_increment_operation_queue)
                compressed_queue: Final = tuple(
                    CounterIncrement(
                        counter=counter,
                        amount=sum(op.amount for op in queued if op.counter == counter),
                        ttl=next(op.ttl for op in queued if op.counter == counter),
                    )
                    for counter in dict.fromkeys(op.counter for op in queued)
                )

                increment_result: Final = await self.usage.increment_remote(compressed_queue)

                self.redis_increment_operation_queue = self.redis_increment_operation_queue[len(queued) :]

                if increment_result is not None:
                    return_result = {op.counter: total for op, total in zip(compressed_queue, increment_result)}
                else:
                    return_result = {}
                return return_result

        except Exception as e:
            log_redis_failure(verbose_router_logger, logging.ERROR, "Error syncing in-memory cache with Redis", e)
            self.redis_increment_operation_queue = []

    def add_to_in_memory_keys_to_update(self, counter: UsageCounter):
        self.in_memory_keys_to_update.add(counter)

    def get_in_memory_keys_to_update(self) -> set[UsageCounter]:
        return self.in_memory_keys_to_update

    def get_and_reset_in_memory_keys_to_update(self) -> set[UsageCounter]:
        """Atomic get and reset in-memory keys to update"""
        keys: Final = self.in_memory_keys_to_update
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
            if not self.usage.has_redis:
                return

            # 2. Fetch all current provider spend from Redis to update in-memory cache
            cache_keys: Final = (
                self.get_in_memory_keys_to_update()
            )  # if no pattern OR redis cache does not support scan_iter, use in-memory keys

            cache_keys_list: Final = list(cache_keys)

            # 1. Snapshot in-memory before
            in_memory_before_dict: Final = {
                k: float(v or 0) for k, v in zip(cache_keys_list, self.usage.get_local_many(cache_keys_list))
            }

            # 1. Push all provider spend increments to Redis
            redis_values: Final = await self._push_in_memory_increments_to_redis()
            if redis_values is None:
                return

            # 4. Merge
            for key in cache_keys_list:
                redis_val = float(redis_values.get(key, 0) or 0)
                before = float(in_memory_before_dict.get(key, 0) or 0)
                after = float(self.usage.get_local(key) or 0)
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
                self.usage.set_local(key, merged)

        except Exception as e:
            verbose_router_logger.exception("Error syncing in-memory cache with Redis: %s", e)
