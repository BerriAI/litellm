from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias

if TYPE_CHECKING:
    from opentelemetry.trace import Span

from litellm._internal_context import with_service_target
from litellm.caching.dual_cache import DualCache
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCache
from litellm.constants import DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE
from litellm.types.caching import RedisPipelineIncrementOperation

from ._keys import ROUTER_RPM_KEY, ROUTER_TPM_KEY


@dataclass(frozen=True, slots=True)
class UsagePolicy:
    redis_read_interval: float | None = None
    memory_ttl: float | None = None
    redis_ttl: float | None = None
    backfill_ttl: int = 600
    max_memory_entries: int = 200
    max_batch_entries: int = DEFAULT_MAX_REDIS_BATCH_CACHE_SIZE
    counter_ttl: int = 60


POLICY: Final = UsagePolicy()
ROUTER_USAGE_TARGET: Final = "router_usage"
CounterKind: TypeAlias = Literal["tpm", "rpm"]


@dataclass(frozen=True, slots=True)
class UsageCounter:
    deployment_id: str
    model: str | None
    minute: str
    kind: CounterKind


def _key(counter: UsageCounter) -> str:
    template: Final = ROUTER_TPM_KEY if counter.kind == "tpm" else ROUTER_RPM_KEY
    return template.format(id=counter.deployment_id, model=counter.model, current_minute=counter.minute)


class RouterUsage(Protocol):
    def attach_redis(self, connection: RedisCache | None) -> None: ...
    def flush_local_counters(self) -> None: ...
    async def async_increment(
        self, counter: UsageCounter, value: float, ttl: int, parent_otel_span: Span | None = None
    ) -> float | None: ...
    async def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
        parent_otel_span: Span | None,
    ) -> None: ...
    async def async_batch_get(self, counters: Sequence[UsageCounter]) -> Sequence[object] | None: ...


class RouterUsageStore:
    def __init__(self, connection: RedisCache | None) -> None:
        self._cache: Final = DualCache(
            redis_cache=connection,
            in_memory_cache=InMemoryCache(
                max_size_in_memory=POLICY.max_memory_entries, default_ttl=POLICY.backfill_ttl
            ),
            default_in_memory_ttl=POLICY.memory_ttl,
            default_redis_ttl=POLICY.redis_ttl,
            default_redis_batch_cache_expiry=POLICY.redis_read_interval,
            default_max_redis_batch_cache_size=POLICY.max_batch_entries,
        )

    def attach_redis(self, connection: RedisCache | None) -> None:
        self._cache.attach_redis_cache(connection)

    def flush_local_counters(self) -> None:
        self._cache.in_memory_cache.flush_cache()

    @with_service_target(ROUTER_USAGE_TARGET)
    async def async_increment(
        self, counter: UsageCounter, value: float, ttl: int, parent_otel_span: Span | None = None
    ) -> float | None:
        return await self._cache.async_increment_cache(
            key=_key(counter), value=value, ttl=ttl, parent_otel_span=parent_otel_span
        )

    @with_service_target(ROUTER_USAGE_TARGET)
    async def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
        parent_otel_span: Span | None,
    ) -> None:
        tpm: Final = UsageCounter(deployment_id, model, minute, "tpm")
        rpm: Final = UsageCounter(deployment_id, model, minute, "rpm")
        operations: Final = [
            RedisPipelineIncrementOperation(key=_key(counter), increment_value=value, ttl=POLICY.counter_ttl)
            for counter, value in ((tpm, tokens), (rpm, requests))
        ]
        values: Final = await self._cache.async_increment_cache_pipeline(
            increment_list=operations, parent_otel_span=parent_otel_span
        )
        if values is not None and self._cache.redis_cache is not None:
            for operation, value in zip(operations, values):
                await self._cache.async_set_cache(operation["key"], int(value), local_only=True, ttl=POLICY.counter_ttl)

    async def async_batch_get(self, counters: Sequence[UsageCounter]) -> Sequence[object] | None:
        return await self._cache.async_batch_get_cache(keys=[_key(counter) for counter in counters])
