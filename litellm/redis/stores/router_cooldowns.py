from __future__ import annotations

import functools
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

if TYPE_CHECKING:
    from opentelemetry.trace import Span
from typing_extensions import ReadOnly, TypedDict

from litellm._internal_context import service_target
from litellm._logging import verbose_router_logger
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS, DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE

from . import _routing_prefetch
from ._routing_prefetch import TierRead


class CooldownCacheValue(TypedDict):
    exception_received: ReadOnly[str]
    status_code: ReadOnly[str]
    timestamp: ReadOnly[float]
    cooldown_time: ReadOnly[float]


@dataclass(frozen=True, slots=True)
class CooldownPolicy:
    redis_read_interval: float = DEFAULT_COOLDOWN_REDIS_READ_INTERVAL_SECONDS
    backfill_ttl: int = 600
    max_memory_entries: int = 200
    max_batch_entries: int = DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE
    max_corrected_ttl: float = 60.0


POLICY: Final = CooldownPolicy()
ROUTER_COOLDOWNS_TARGET: Final = "router_cooldowns"
ActiveCooldowns: TypeAlias = Sequence[tuple[str, CooldownCacheValue]]


@dataclass(frozen=True, slots=True)
class RoutingRead:
    cooldowns: ActiveCooldowns
    selector_values: Sequence[object | None] | None


@functools.lru_cache(maxsize=1024)
def _cooldown_key(model_id: str) -> str:
    return "deployment:" + model_id + ":cooldown"


class RouterCooldowns(Protocol):
    def attach_redis(self, connection: RedisCache | None) -> None: ...
    def flush_local_counters(self) -> None: ...
    def put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None: ...
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int: ...
    def arm_routing_read(self, model_ids: Sequence[str], selector: TierRead | None) -> None: ...
    async def routing_read(
        self, model_ids: Sequence[str], selector: TierRead | None, parent_otel_span: Span | None
    ) -> RoutingRead: ...
    async def active(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns: ...
    def active_sync(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns: ...
    def minimum(self, model_ids: Sequence[str], default: float, parent_otel_span: Span | None) -> float: ...
    def evict_local(self, model_id: str) -> None: ...


class RouterCooldownStore:
    def __init__(self, connection: RedisCache | None, *, clock: Callable[[], float] | None = None) -> None:
        self._clock: Final = clock if clock is not None else lambda: time.time()
        self._memory: Final = InMemoryCache(
            max_size_in_memory=POLICY.max_memory_entries, default_ttl=POLICY.backfill_ttl, clock=self._clock
        )
        self._cache: Final = DualCache(
            redis_cache=connection,
            in_memory_cache=self._memory,
            default_redis_batch_cache_expiry=POLICY.redis_read_interval,
            default_max_redis_batch_cache_size=POLICY.max_batch_entries,
            clock=self._clock,
        )
        self._fails: Final = DualCache(redis_cache=connection, in_memory_cache=InMemoryCache(clock=self._clock))

    def attach_redis(self, connection: RedisCache | None) -> None:
        self._cache.attach_redis_cache(connection)
        self._fails.attach_redis_cache(connection)

    def flush_local_counters(self) -> None:
        self._fails.in_memory_cache.flush_cache()

    def put(self, model_id: str, value: CooldownCacheValue, ttl: float) -> None:
        with service_target(ROUTER_COOLDOWNS_TARGET):
            self._cache.set_cache(key=_cooldown_key(model_id), value=value, ttl=ttl)

    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl: float) -> int:
        base: Final = f"deployment:{model_id}:allowed_fails"
        key: Final = f"{base}:{suffix}" if suffix else base
        with service_target(ROUTER_COOLDOWNS_TARGET):
            try:
                return self._fails.increment_cache(key=key, value=1, ttl=ttl)
            except Exception as e:  # noqa: BLE001  # preserve worker-local counting during a Redis outage
                verbose_router_logger.warning("allowed_fails counter fell back to this worker's in-memory count: %s", e)
                local: Final[object] = self._fails.get_cache(key=key, local_only=True)
                return local if isinstance(local, int) else 0

    def _tier_read(self, model_ids: Sequence[str]) -> TierRead:
        return TierRead(self._cache, tuple(_cooldown_key(model_id) for model_id in model_ids))

    def arm_routing_read(self, model_ids: Sequence[str], selector: TierRead | None) -> None:
        _routing_prefetch.arm((self._tier_read(model_ids), *(() if selector is None else (selector,))))

    async def routing_read(
        self, model_ids: Sequence[str], selector: TierRead | None, parent_otel_span: Span | None
    ) -> RoutingRead:
        reads: Final = (self._tier_read(model_ids), *(() if selector is None else (selector,)))
        results: Final = await _routing_prefetch.read(reads, parent_otel_span)
        return RoutingRead(
            cooldowns=self._active_from_results(model_ids, results[0]),
            selector_values=None if selector is None else results[1],
        )

    def _corrected_active_cooldown(
        self, model_id: str, result: Mapping[str, object], current_time: float
    ) -> CooldownCacheValue | None:
        value: Final = CooldownCacheValue(
            **result  # pyright: ignore[reportArgumentType]  # preserve main's unvalidated dict codec
        )
        remaining: Final = value["timestamp"] + value["cooldown_time"] - current_time
        key: Final = _cooldown_key(model_id)
        if remaining <= 0:
            self._memory.delete_cache(key)
            return None
        expiry: Final = self._memory.ttl_dict.get(key)
        if expiry is not None and expiry > current_time + remaining + 5:
            self._memory.delete_cache(key)
            self._memory.set_cache(key, result, ttl=min(remaining, POLICY.max_corrected_ttl))
        return value

    def _active_from_results(self, model_ids: Sequence[str], results: Sequence[object] | None) -> ActiveCooldowns:
        if results is None:
            return ()
        now: Final = self._clock()
        return tuple(
            (model_id, active)
            for model_id, result in zip(model_ids, results)
            if isinstance(result, dict)
            and result
            and (active := self._corrected_active_cooldown(model_id, result, now)) is not None
        )

    async def active(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns:
        with service_target(ROUTER_COOLDOWNS_TARGET):
            results: Final = await self._cache.async_batch_get_cache(
                keys=[_cooldown_key(model_id) for model_id in model_ids], parent_otel_span=parent_otel_span
            )
        return self._active_from_results(model_ids, results)

    def minimum(self, model_ids: Sequence[str], default: float, parent_otel_span: Span | None) -> float:
        with service_target(ROUTER_COOLDOWNS_TARGET):
            results: Final = (
                self._cache.batch_get_cache(
                    keys=[_cooldown_key(model_id) for model_id in model_ids], parent_otel_span=parent_otel_span
                )
                or ()
            )
        durations: Final = tuple(
            CooldownCacheValue(**result)["cooldown_time"]  # pyright: ignore[reportArgumentType]  # preserve main's dict codec
            for result in results
            if isinstance(result, dict) and result
        )
        return (min(durations) if durations else None) or default

    def active_sync(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> ActiveCooldowns:
        with service_target(ROUTER_COOLDOWNS_TARGET):
            results: Final = self._cache.batch_get_cache(
                keys=[_cooldown_key(model_id) for model_id in model_ids], parent_otel_span=parent_otel_span
            )
        return self._active_from_results(model_ids, results)

    def evict_local(self, model_id: str) -> None:
        self._memory.delete_cache(_cooldown_key(model_id))
