from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from typing import Final, Protocol

from mcp.types import ListToolsResult
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.constants import MCP_TOOL_LISTING_TIMEOUT, MCP_TOOLS_LIST_CACHE_GENERATION_TTL

_RESULT: Final[TypeAdapter[ListToolsResult]] = TypeAdapter(ListToolsResult)
_STRING: Final[TypeAdapter[str]] = TypeAdapter(str)
_INT: Final[TypeAdapter[int]] = TypeAdapter(int)
_FLOAT: Final[TypeAdapter[float]] = TypeAdapter(float)
_CACHE_PREFIX: Final = "litellm:mcp:tools_list:v1"
TOOLS_LIST_CACHE_SETTINGS_KEY: Final = "mcp_tools_list_cache_ttl"


class _RedisCacheAdapter(Protocol):
    async def async_get_cache(self, key: str) -> object: ...

    async def async_set_cache(self, key: str, value: object, *, ttl: int | None = None, nx: bool = False) -> object: ...

    async def async_increment(self, key: str, value: float, ttl: int | None = None) -> object: ...

    async def async_delete_cache(self, key: str) -> object: ...


class ToolsListStore(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ttl_seconds: float) -> None: ...

    async def set_nx(self, key: str, value: str, ttl_seconds: float) -> bool: ...

    async def incr(self, key: str) -> int: ...

    async def delete(self, key: str) -> None: ...


class CacheEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fetched_at: float
    fresh_until: float
    stale_until: float
    payload: str


class RedisToolsListStore:
    def __init__(self, redis_cache: _RedisCacheAdapter) -> None:
        self._redis_cache = redis_cache

    async def get(self, key: str) -> str | None:
        value: Final = await self._redis_cache.async_get_cache(key=key)
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return str(value)
        return _STRING.validate_python(value)

    async def set(self, key: str, value: str, ttl_seconds: float) -> None:
        await self._redis_cache.async_set_cache(key=key, value=value, ttl=math.ceil(ttl_seconds))

    async def set_nx(self, key: str, value: str, ttl_seconds: float) -> bool:
        result: Final = await self._redis_cache.async_set_cache(
            key=key,
            value=value,
            ttl=math.ceil(ttl_seconds),
            nx=True,
        )
        return result in {"OK", b"OK"} or result is True

    async def incr(self, key: str) -> int:
        value: Final = await self._redis_cache.async_increment(
            key=key,
            value=1,
            ttl=MCP_TOOLS_LIST_CACHE_GENERATION_TTL,
        )
        return _INT.validate_python(value)

    async def delete(self, key: str) -> None:
        await self._redis_cache.async_delete_cache(key=key)


class InProcessToolsListStore:
    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._values: dict[str, tuple[str, float | None]] = {}  # mutable-ok: store state
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> str | None:
        async with self._lock:
            current: Final = self._values.get(key)
            if current is None:
                return None
            value, expires_at = current
            if expires_at is not None and expires_at <= self._clock():
                self._values.pop(key, None)
                return None
            return value

    async def set(self, key: str, value: str, ttl_seconds: float) -> None:
        async with self._lock:
            self._values[key] = (value, self._clock() + ttl_seconds)

    async def set_nx(self, key: str, value: str, ttl_seconds: float) -> bool:
        async with self._lock:
            existing: Final = self._values.get(key)
            if existing is not None and (existing[1] is None or existing[1] > self._clock()):
                return False
            self._values[key] = (value, self._clock() + ttl_seconds)
            return True

    async def incr(self, key: str) -> int:
        async with self._lock:
            existing: Final = self._values.get(key)
            incremented: Final = (0 if existing is None else _INT.validate_python(existing[0])) + 1
            self._values[key] = (str(incremented), None)
            return incremented

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._values.pop(key, None)


def tools_list_cache_ttl(ui_value: object | None, env_value: str | None) -> float:
    raw: Final = ui_value if ui_value is not None else env_value
    if raw is None:
        return 0.0
    if isinstance(raw, bool):
        verbose_logger.warning("Invalid MCP tools/list cache TTL; disabling the cache")
        return 0.0
    try:
        value: Final = _FLOAT.validate_python(raw)
    except (OverflowError, TypeError, ValueError, ValidationError):
        verbose_logger.warning("Invalid MCP tools/list cache TTL; disabling the cache")
        return 0.0
    if not math.isfinite(value) or value < 0:
        verbose_logger.warning("Invalid MCP tools/list cache TTL; disabling the cache")
        return 0.0
    return value


class SharedToolsListCache:
    def __init__(
        self,
        store_provider: Callable[[], ToolsListStore],
        ttl_seconds: Callable[[], float],
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._store_provider = store_provider
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._generations: dict[str, int] = {}  # mutable-ok: invalidation state
        self._fetches: dict[str, asyncio.Task[ListToolsResult]] = {}  # mutable-ok: single-flight tasks
        self._refreshes: dict[str, asyncio.Task[None]] = {}  # mutable-ok: background task references
        self._tasks: set[asyncio.Task[None]] = set()  # mutable-ok: background task references

    async def get(
        self,
        server_id: str,
        config_digest: str,
        cursor_slot: str,
        fetch: Callable[[], Awaitable[ListToolsResult]],
    ) -> ListToolsResult:
        configured_ttl: Final = self._ttl_seconds()
        if not math.isfinite(configured_ttl) or configured_ttl <= 0:
            return await fetch()
        try:
            store: Final = self._store_provider()
            remote_generation: Final = await store.get(f"{_CACHE_PREFIX}:{server_id}:gen")
            generation: Final = max(
                self._generations.get(server_id, 0),
                0 if remote_generation is None else _INT.validate_python(remote_generation),
            )
            key: Final = f"{_CACHE_PREFIX}:{server_id}:{generation}:{config_digest}:{cursor_slot}"
            entry_value: Final = await store.get(key)
        except Exception as exception:  # noqa: BLE001  # cache read failures must not block tool listing
            verbose_logger.debug("MCP tools/list cache read failed: %s", exception)
            return await fetch()
        if entry_value is not None:
            try:
                entry: Final = CacheEntry.model_validate_json(entry_value)
                cached_result: Final = _RESULT.validate_json(entry.payload)
            except Exception as exception:  # noqa: BLE001  # invalid cache data must not block tool listing
                verbose_logger.debug("MCP tools/list cache entry was invalid: %s", exception)
                return await self._singleflight(store, key, configured_ttl, fetch)
            now: Final = self._clock()
            if now < entry.fresh_until - (entry.fresh_until - entry.fetched_at) * 0.2:
                return cached_result
            if now < entry.stale_until:
                self._schedule_refresh(store, key, entry, configured_ttl, fetch)
                return cached_result
        return await self._singleflight(store, key, configured_ttl, fetch)

    def invalidate(self, server_id: str) -> None:
        self._generations[server_id] = self._generations.get(server_id, 0) + 1
        try:
            task: Final = asyncio.get_running_loop().create_task(self._increment_generation(server_id))
        except RuntimeError:
            return
        self._retain_task(task)

    async def _increment_generation(self, server_id: str) -> None:
        try:
            await self._store_provider().incr(f"{_CACHE_PREFIX}:{server_id}:gen")
        except Exception as exception:  # noqa: BLE001  # cache invalidation failures must not block tool listing
            verbose_logger.debug("MCP tools/list cache invalidation failed: %s", exception)

    async def _singleflight(
        self,
        store: ToolsListStore,
        key: str,
        configured_ttl: float,
        fetch: Callable[[], Awaitable[ListToolsResult]],
    ) -> ListToolsResult:
        task: Final = self._fetches.get(key)
        if task is not None:
            return await asyncio.shield(task)
        created: Final = asyncio.create_task(self._fetch_and_store(store, key, configured_ttl, fetch))
        self._fetches[key] = created
        created.add_done_callback(lambda completed: self._fetches.pop(key, None))
        return await asyncio.shield(created)

    async def _fetch_and_store(
        self,
        store: ToolsListStore,
        key: str,
        configured_ttl: float,
        fetch: Callable[[], Awaitable[ListToolsResult]],
    ) -> ListToolsResult:
        result: Final = await fetch()
        effective_ttl: Final = self._effective_ttl(result, configured_ttl)
        if effective_ttl <= 0:
            try:
                await store.delete(key)
            except Exception as exception:  # noqa: BLE001  # cache delete failures must not block tool listing
                verbose_logger.debug("MCP tools/list cache delete failed: %s", exception)
            return result
        now: Final = self._clock()
        fresh_until: Final = now + effective_ttl
        entry: Final = CacheEntry(
            fetched_at=now,
            fresh_until=fresh_until,
            stale_until=fresh_until + effective_ttl,
            payload=result.model_dump_json(),
        )
        try:
            await store.set(key, entry.model_dump_json(), entry.stale_until - now)
        except Exception as exception:  # noqa: BLE001  # cache write failures must not block tool listing
            verbose_logger.debug("MCP tools/list cache write failed: %s", exception)
        return result

    def _schedule_refresh(
        self,
        store: ToolsListStore,
        key: str,
        entry: CacheEntry,
        configured_ttl: float,
        fetch: Callable[[], Awaitable[ListToolsResult]],
    ) -> None:
        if key in self._refreshes:
            return
        task: Final = asyncio.create_task(self._refresh(store, key, entry, configured_ttl, fetch))
        self._refreshes[key] = task
        task.add_done_callback(lambda completed: self._refreshes.pop(key, None))
        self._retain_task(task)

    async def _refresh(
        self,
        store: ToolsListStore,
        key: str,
        entry: CacheEntry,
        configured_ttl: float,
        fetch: Callable[[], Awaitable[ListToolsResult]],
    ) -> None:
        lock_key: Final = f"{key}:refresh"
        try:
            acquired: Final = await store.set_nx(
                lock_key,
                "1",
                max(math.ceil(MCP_TOOL_LISTING_TIMEOUT), 1),
            )
            if not acquired:
                return
            try:
                await self._fetch_and_store(store, key, configured_ttl, fetch)
            finally:
                await store.delete(lock_key)
        except Exception as exception:  # noqa: BLE001  # background refresh failures must preserve stale tools
            verbose_logger.warning("MCP tools/list background refresh failed: %s", exception)

    def _effective_ttl(self, result: ListToolsResult, configured_ttl: float) -> float:
        if "ttl_ms" not in result.model_fields_set:
            return configured_ttl
        if result.ttl_ms == 0:
            return 0.0
        return min(configured_ttl, result.ttl_ms / 1000)

    def _retain_task(self, task: asyncio.Task[None]) -> None:
        self._tasks.add(task)

        def observe(completed: asyncio.Task[None]) -> None:
            self._tasks.discard(completed)
            if not completed.cancelled():
                completed.exception()

        task.add_done_callback(observe)
