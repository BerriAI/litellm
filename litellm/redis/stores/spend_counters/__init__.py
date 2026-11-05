from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from typing import Final

from litellm._internal_context import with_service_target
from litellm.caching.redis_batch import Batch
from litellm.redis.stores._route import post_call_request, route
from litellm.redis.stores.spend_counters.handle import HandleFactory, SpendCountersHandle
from litellm.redis.stores.spend_counters.types import (
    END_USER_COUNTER_PREFIX,
    SPEND_COUNTER_PREFIX,
    SPEND_COUNTERS_TARGET,
    SpendIncrement,
    SpendReading,
    end_user_counter,
    is_unavailable,
    key_counter,
    key_window_counter,
    model_access_group_counter,
    org_counter,
    project_counter,
    shadow_eval_counter,
    tag_counter,
    team_counter,
    team_member_counter,
    team_window_counter,
    user_counter,
)
from litellm.rust_bridge._native import NativeRedisClient, NativeSpendCounters

__all__ = [
    "END_USER_COUNTER_PREFIX",
    "SPEND_COUNTERS_TARGET",
    "SPEND_COUNTER_PREFIX",
    "HandleFactory",
    "SpendCounterStore",
    "SpendCountersHandle",
    "SpendIncrement",
    "SpendReading",
    "end_user_counter",
    "is_unavailable",
    "key_counter",
    "key_window_counter",
    "model_access_group_counter",
    "org_counter",
    "project_counter",
    "shadow_eval_counter",
    "tag_counter",
    "team_counter",
    "team_member_counter",
    "team_window_counter",
    "user_counter",
]


async def _by_counter(
    counters: Sequence[str], values: Awaitable[Sequence[float | None] | None] | None
) -> Mapping[str, float | None] | None:
    read: Final = None if values is None else await values
    return None if read is None else dict(zip(counters, read, strict=True))


class SpendCounterStore:
    """The ``spend:*`` budget counters: Redis shared by every worker, and this worker's memory copy that answers
    only when Redis cannot. Counters are named by the builders above."""

    def __init__(
        self, client: NativeRedisClient | None, *, redis_ttl: int = 60, handle: HandleFactory = NativeSpendCounters
    ) -> None:
        self._handle: Final = handle(client, redis_ttl)
        self._redis_ttl = None if client is None else redis_ttl

    def attach_redis(self, client: NativeRedisClient | None, *, redis_ttl: int) -> None:
        """Adds the Redis tier once, keeping the memory copies, as ``DualCache.attach_redis_cache`` does."""
        if client is None or not self._handle.attach(client, redis_ttl):
            return
        self._redis_ttl = redis_ttl

    @property
    def redis_ttl(self) -> int | None:
        """The TTL every Redis write gives a counter, None when there is no Redis."""
        return self._redis_ttl

    @property
    def has_redis(self) -> bool:
        return self._redis_ttl is not None

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def read(self, counter: str) -> SpendReading:
        source, value = await self._handle.read(counter)
        return value, source == "redis"

    @with_service_target(SPEND_COUNTERS_TARGET)
    def read_many(self, counters: Sequence[str], *, batch: Batch) -> Awaitable[Mapping[str, float | None] | None]:
        """One MGET declared now on the batch ``batch`` picks. None when there is no Redis or the read failed, so
        each caller reads on its own."""
        target: Final = route(batch)
        pending: Final = self._handle.read_many(counters, target.send)
        return _by_counter(counters, None if pending is None else target.result(pending))

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def increment(self, counter: str, amount: float) -> float:
        """Raises when the increment may not have landed, after dropping the counter everywhere."""
        return await self._handle.increment(counter, amount)

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def increment_many(self, increments: Sequence[SpendIncrement]) -> tuple[float, ...]:
        return tuple(await self._handle.increment_many(increments))

    @with_service_target(SPEND_COUNTERS_TARGET)
    def increment_many_post_call(self, increments: Sequence[SpendIncrement]) -> bool:
        """Declares the increments on the request's post-call batch, False when there is no Redis or request.
        Each total lands in memory once the batch goes out; a counter whose increment failed is dropped everywhere."""
        request: Final = post_call_request() if self.has_redis and increments else None
        if request is None:
            return False
        return self._handle.increment_many_post_call(request, increments)

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def set_max(self, counter: str, spend: float) -> float | None:
        """Raise memory and Redis to ``spend``, never lowering either; what Redis holds after, None without Redis."""
        return await self._handle.set_max(counter, spend)

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def seed(self, counter: str, spend: float, *, increment_if_lost: bool) -> float:
        """Seed a cold counter with SET NX so one worker wins, answering with what the counter then holds."""
        return await self._handle.seed(counter, spend, increment_if_lost=increment_if_lost)

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def set(self, counter: str, spend: float, ttl: float | None = None) -> None:
        """Memory, then Redis; raises when Redis failed, with memory already written."""
        await self._handle.set(counter, spend, ttl)

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def delete(self, counters: Sequence[str]) -> None:
        """Memory, then Redis in one round trip; raises when Redis failed, with memory already dropped."""
        await self._handle.delete(counters)

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def refresh_ttl(self, counter: str) -> bool:
        return await self._handle.refresh_ttl(counter)

    def get_local(self, counter: str) -> float | None:
        return self._handle.get_local(counter)

    def set_local(self, counter: str, spend: float, ttl: float | None = None) -> None:
        self._handle.set_local(counter, spend, ttl)

    def delete_local(self, counter: str) -> None:
        self._handle.delete_local(counter)

    def increment_local(self, counter: str, amount: float) -> float:
        return self._handle.increment_local(counter, amount)

    def flush_local(self) -> None:
        self._handle.flush_local()

    def snapshot(self) -> Mapping[str, float]:
        return self._handle.snapshot()

    def apply_broadcast(self, cache_key: str, new_value: float | None, ttl: float | None) -> None:
        """A cross-worker auth cache invalidation: a SET writes this worker's copy, a delete drops it. Keys outside
        the spend counter keyspace are not this store's."""
        if not cache_key.startswith(SPEND_COUNTER_PREFIX):
            return
        try:
            if new_value is None:
                self.delete_local(cache_key)
            else:
                self.set_local(cache_key, new_value, ttl)
        except ValueError:
            return
