from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final, Literal, Protocol, TypeAlias

from litellm._internal_context import service_target, with_service_target
from litellm.caching.redis_batch import (
    NativeBatchResult,
    NativePending,
    RequestRedisBatches,
    active_request_redis_batches,
)
from litellm.rust_bridge._native import (
    NativeRedisClient,
    NativeRequestRedis,
    NativeSpendCounterError,
    NativeSpendCounters,
)

SPEND_COUNTERS_TARGET: Final = "spend_counters"
SPEND_COUNTER_PREFIX: Final = "spend:"
END_USER_COUNTER_PREFIX: Final = "spend:end_user:"

SpendIncrement: TypeAlias = tuple[str, float]
SpendReading: TypeAlias = tuple[float | None, bool]
"""(value, authoritative): authoritative when Redis answered, a clean miss included."""


def key_counter(token: str) -> str:
    return f"spend:key:{token}"


def key_window_counter(token: str, budget_duration: str) -> str:
    return f"spend:key:{token}:window:{budget_duration}"


def team_counter(team_id: str) -> str:
    return f"spend:team:{team_id}"


def team_window_counter(team_id: str, budget_duration: str) -> str:
    return f"spend:team:{team_id}:window:{budget_duration}"


def team_member_counter(user_id: str, team_id: str) -> str:
    return f"spend:team_member:{user_id}:{team_id}"


def user_counter(user_id: str) -> str:
    return f"spend:user:{user_id}"


def end_user_counter(end_user_id: str) -> str:
    return f"{END_USER_COUNTER_PREFIX}{end_user_id}"


def org_counter(org_id: str) -> str:
    return f"spend:org:{org_id}"


def tag_counter(tag: str) -> str:
    return f"spend:tag:{tag}"


def project_counter(project_id: str) -> str:
    return f"spend:project:{project_id}"


def model_access_group_counter(access_group: str) -> str:
    return f"spend:model_access_group:{access_group}"


def shadow_eval_counter(job_id: str) -> str:
    return f"spend:shadow_eval:{job_id}"


def is_unavailable(error: BaseException) -> bool:
    """Redis timed out or its breaker is open, the failures a post-call increment drops."""
    return isinstance(error, NativeSpendCounterError) and error.args[:1] in (("timeout",), ("breaker_open",))


class SpendCountersHandle(Protocol):
    def read(self, counter: str) -> Awaitable[tuple[Literal["redis", "memory"], float | None]]: ...
    def read_many(self, counters: Sequence[str]) -> Awaitable[Sequence[float | None] | None]: ...
    def declare_read_many(
        self, request: NativeRequestRedis, counters: Sequence[str]
    ) -> NativePending[Sequence[float | None] | None] | None: ...
    def increment(self, counter: str, amount: float) -> Awaitable[float]: ...
    def increment_many(self, increments: Sequence[SpendIncrement]) -> Awaitable[Sequence[float]]: ...
    def increment_many_post_call(self, request: NativeRequestRedis, increments: Sequence[SpendIncrement]) -> bool: ...
    def set_max(self, counter: str, spend: float) -> Awaitable[float | None]: ...
    def seed(self, counter: str, spend: float, *, increment_if_lost: bool) -> Awaitable[float]: ...
    def set(self, counter: str, spend: float, ttl_seconds: float | None = None) -> Awaitable[None]: ...
    def delete(self, counters: Sequence[str]) -> Awaitable[None]: ...
    def refresh_ttl(self, counter: str) -> Awaitable[bool]: ...
    def get_local(self, counter: str) -> float | None: ...
    def set_local(self, counter: str, spend: float, ttl_seconds: float | None = None) -> None: ...
    def delete_local(self, counter: str) -> None: ...
    def increment_local(self, counter: str, amount: float) -> float: ...
    def flush_local(self) -> None: ...
    def snapshot(self) -> Mapping[str, float]: ...


HandleFactory: TypeAlias = Callable[[NativeRedisClient | None, int], SpendCountersHandle]


class SpendCounterStore:
    """The ``spend:*`` budget counters: Redis shared by every worker, and this worker's memory copy that answers
    only when Redis cannot. Counters are named by the builders above."""

    def __init__(
        self, client: NativeRedisClient | None, *, redis_ttl: int = 60, handle: HandleFactory = NativeSpendCounters
    ) -> None:
        self._build: Final = handle
        self._handle = handle(client, redis_ttl)
        self._redis_ttl = None if client is None else redis_ttl

    def attach_redis(self, client: NativeRedisClient | None, *, redis_ttl: int) -> None:
        if client is None or self._redis_ttl is not None:
            return
        self._handle = self._build(client, redis_ttl)
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
    async def read_many(self, counters: Sequence[str]) -> Mapping[str, float | None] | None:
        """None when there is no Redis or the read failed, so each caller reads on its own."""
        values: Final = await self._handle.read_many(counters)
        return None if values is None else dict(zip(counters, values, strict=True))

    def declare_read_many(
        self, request: RequestRedisBatches, counters: Sequence[str]
    ) -> NativeBatchResult[Sequence[float | None] | None] | None:
        """``read_many`` declared on the request's native pre-call batch; None when there is no Redis."""
        if not self.has_redis:
            return None
        with service_target(SPEND_COUNTERS_TARGET):
            pending: Final = self._handle.declare_read_many(request.native(), counters)
        return None if pending is None else request.native_result(pending)

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def increment(self, counter: str, amount: float) -> float:
        """Raises when the increment may not have landed, after dropping the counter everywhere."""
        return await self._handle.increment(counter, amount)

    @with_service_target(SPEND_COUNTERS_TARGET)
    async def increment_many(self, increments: Sequence[SpendIncrement]) -> tuple[float, ...]:
        return tuple(await self._handle.increment_many(increments))

    def increment_many_post_call(self, increments: Sequence[SpendIncrement]) -> bool:
        """Declares the increments on the request's post-call batch, False when there is no Redis or request.
        Each total lands in memory once the batch goes out; a counter whose increment failed is dropped everywhere."""
        request: Final = active_request_redis_batches()
        if request is None or not self.has_redis or not increments:
            return False
        with service_target(SPEND_COUNTERS_TARGET):
            return self._handle.increment_many_post_call(request.native_post_call(), increments)

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
