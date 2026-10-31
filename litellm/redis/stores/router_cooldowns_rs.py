import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, TypeAlias

from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.redis.stores.router_cooldowns import CooldownCacheValue, Span, cooldown_key, decode
from litellm.redis.failure import RedisFailure, from_native
from litellm.rust_bridge._native import NativeRedisFailure, NativeRouterCooldowns

_ENTRY: Final = TypeAdapter(CooldownCacheValue)

Entries: TypeAlias = tuple[CooldownCacheValue | None, ...]


def _entry(item: dict[str, str | float] | NativeRedisFailure | None) -> CooldownCacheValue | None:
    if item is None or isinstance(item, NativeRedisFailure):
        return None
    return _ENTRY.validate_python(item)


def _entries(
    reply: list[dict[str, str | float] | NativeRedisFailure | None] | NativeRedisFailure,
) -> Entries | RedisFailure:
    return from_native(reply) if isinstance(reply, NativeRedisFailure) else tuple(map(_entry, reply))


@dataclass(frozen=True, slots=True)
class NativeCooldownRedis:
    handle: NativeRouterCooldowns

    def put(self, model_id: str, entry: CooldownCacheValue, *, ttl: float) -> None:
        failure: Final = self.handle.put(model_id, entry, ttl_seconds=int(ttl))
        if failure is not None:
            verbose_logger.error("router cooldown write failed for %s: %s", model_id, failure.message)

    def get_many(self, model_ids: Sequence[str]) -> Entries | RedisFailure:
        return _entries(self.handle.get_many(model_ids))

    async def async_get_many(self, model_ids: Sequence[str]) -> Entries | RedisFailure:
        return _entries(await self.handle.aget_many(model_ids))


@dataclass(frozen=True, slots=True)
class NativeRouterCooldownStore:
    memory: InMemoryCache
    redis: NativeCooldownRedis
    refresh_after: float
    reads: InMemoryCache = field(default_factory=InMemoryCache)

    def put(self, model_id: str, entry: CooldownCacheValue, *, ttl: float) -> None:
        self.memory.set_cache(cooldown_key(model_id), json.dumps(entry), ttl=ttl)
        self.redis.put(model_id, entry, ttl=ttl)

    def get_many(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> Entries:
        local: Final = self._local(model_ids)
        due: Final = self._reserve(model_ids, local)
        return local if not due else self._merge(model_ids, local, due, self.redis.get_many(due))

    async def async_get_many(self, model_ids: Sequence[str], parent_otel_span: Span | None) -> Entries:
        local: Final = self._local(model_ids)
        due: Final = self._reserve(model_ids, local)
        return local if not due else self._merge(model_ids, local, due, await self.redis.async_get_many(due))

    def reset_local_ttl(self, model_id: str, entry: CooldownCacheValue, *, ttl: float) -> None:
        self.memory.delete_cache(cooldown_key(model_id))
        self.memory.set_cache(cooldown_key(model_id), json.dumps(entry), ttl=ttl)

    def drop_local(self, model_id: str) -> None:
        self.memory.delete_cache(cooldown_key(model_id))

    def _local(self, model_ids: Sequence[str]) -> Entries:
        return tuple(decode(self.memory.get_cache(cooldown_key(model_id))) for model_id in model_ids)

    def _reserve(self, model_ids: Sequence[str], local: Entries) -> tuple[str, ...]:
        due: Final = tuple(
            model_id
            for model_id, entry in zip(model_ids, local)
            if entry is None and self.reads.get_cache(model_id) is None
        )
        for model_id in due:
            self.reads.set_cache(model_id, True, ttl=self.refresh_after)
        return due

    def _merge(
        self, model_ids: Sequence[str], local: Entries, due: tuple[str, ...], fetched: Entries | RedisFailure
    ) -> Entries:
        if isinstance(fetched, RedisFailure):
            for model_id in due:
                self.reads.delete_cache(model_id)
            if fetched.kind == "breaker_open":
                return local
            verbose_logger.error("router cooldown read failed: %s", fetched.message)
            return tuple(None for _ in model_ids)
        found: Final = dict(zip(due, fetched))
        for model_id, entry in found.items():
            if entry is not None:
                self.memory.set_cache(cooldown_key(model_id), json.dumps(entry))
        return tuple(found.get(model_id) if entry is None else entry for model_id, entry in zip(model_ids, local))
