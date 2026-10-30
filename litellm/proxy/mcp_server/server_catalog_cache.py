from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import suppress
from datetime import timedelta
from typing import TYPE_CHECKING, Final, TypeVar

from pydantic import TypeAdapter

from litellm.caching.redis_batch import active_request_redis_batch
from litellm.redis.stores.mcp_server_catalog import (
    CACHED_TOOLSET,
    SUBMITTED_SERVER_IDS,
    TOOLSET_NOT_FOUND,
    TOOLSET_PERMISSIONS,
    CachedToolset,
    McpServerCatalogStore,
    TransportMcpServerCatalogStore,
    submitted_servers_key,
    toolset_name_key,
    toolset_permissions_key,
)
from litellm.redis.transport.contract import RedisFailure
from litellm.types.mcp_server.mcp_toolset import MCPToolset

if TYPE_CHECKING:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache

_T = TypeVar("_T")
_LOCAL_VALUE: Final = TypeAdapter(object)


async def get_submitted_server_ids(cache: UserApiKeyCache, user_id: str) -> tuple[str, ...] | None:
    return await _read_through(
        cache,
        submitted_servers_key(user_id),
        SUBMITTED_SERVER_IDS,
        lambda store: store.get_submitted_server_ids(user_id),
        list,
    )


async def put_submitted_server_ids(
    cache: UserApiKeyCache, user_id: str, server_ids: Sequence[str], *, ttl: float
) -> None:
    await _write_through(
        cache,
        submitted_servers_key(user_id),
        list(server_ids),
        ttl,
        lambda store, expiry: store.put_submitted_server_ids(user_id, server_ids, ttl=expiry),
    )


async def invalidate_submitted_server_ids(cache: UserApiKeyCache, user_id: str) -> None:
    cache.in_memory_cache.delete_cache(submitted_servers_key(user_id))
    if cache.redis_cache is None:
        return
    with suppress(Exception):
        _ = await _store(cache).invalidate_submitted_server_ids(user_id)


async def get_toolset_permissions(
    cache: UserApiKeyCache, toolset_ids: Sequence[str]
) -> dict[str, tuple[str, ...]] | None:
    return await _read_through(
        cache,
        toolset_permissions_key(toolset_ids),
        TOOLSET_PERMISSIONS,
        lambda store: store.get_toolset_permissions(toolset_ids),
        _permissions_value,
    )


async def put_toolset_permissions(
    cache: UserApiKeyCache, toolset_ids: Sequence[str], permissions: Mapping[str, Sequence[str]], *, ttl: float
) -> None:
    await _write_through(
        cache,
        toolset_permissions_key(toolset_ids),
        _permissions_value(permissions),
        ttl,
        lambda store, expiry: store.put_toolset_permissions(toolset_ids, permissions, ttl=expiry),
    )


async def get_toolset_by_name(cache: UserApiKeyCache, toolset_name: str) -> CachedToolset | None:
    return await _read_through(
        cache,
        toolset_name_key(toolset_name),
        CACHED_TOOLSET,
        lambda store: store.get_toolset_by_name(toolset_name),
        _toolset_value,
    )


async def put_toolset_by_name(
    cache: UserApiKeyCache, toolset_name: str, toolset: MCPToolset | None, *, ttl: float
) -> None:
    await _write_through(
        cache,
        toolset_name_key(toolset_name),
        _toolset_value(toolset if toolset is not None else TOOLSET_NOT_FOUND),
        ttl,
        lambda store, expiry: store.put_toolset_by_name(toolset_name, toolset, ttl=expiry),
    )


def _permissions_value(permissions: Mapping[str, Sequence[str]]) -> dict[str, list[str]]:
    return {server_id: list(tools) for server_id, tools in permissions.items()}


def _toolset_value(toolset: CachedToolset) -> object:
    return toolset if isinstance(toolset, str) else toolset.model_dump(mode="json")


def _store(cache: UserApiKeyCache) -> McpServerCatalogStore:
    assert cache.redis_cache is not None
    return TransportMcpServerCatalogStore(cache.redis_cache.transport())


async def _read_through(
    cache: UserApiKeyCache,
    key: str,
    adapter: TypeAdapter[_T],
    fetch: Callable[[McpServerCatalogStore], Awaitable[_T | None | RedisFailure]],
    to_local: Callable[[_T], object],
) -> _T | None:
    local: Final = _LOCAL_VALUE.validate_python(await cache.async_get_cache(key=key, local_only=True))
    if local is not None:
        with suppress(ValueError):
            return adapter.validate_python(local)
    if cache.redis_cache is None:
        return None
    batch: Final = active_request_redis_batch(cache.redis_cache)
    if batch is not None and batch.read_as_missing(key):
        return None
    with suppress(Exception):
        result: Final = await fetch(_store(cache))
        if result is None or isinstance(result, RedisFailure):
            return None
        await cache.async_set_cache(key=key, value=to_local(result), local_only=True)
        return result
    return None


async def _write_through(
    cache: UserApiKeyCache,
    key: str,
    local_value: object,
    ttl: float,
    put: Callable[[McpServerCatalogStore, timedelta], Awaitable[bool | None | RedisFailure]],
) -> None:
    await cache.async_set_cache(key=key, value=local_value, ttl=ttl, local_only=True)
    if cache.redis_cache is None:
        return
    with suppress(Exception):
        _ = await put(_store(cache), timedelta(seconds=int(ttl)))
