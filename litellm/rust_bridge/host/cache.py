from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.caching.caching import Cache
    from litellm.caching.dual_cache import DualCache


@dataclass(frozen=True, slots=True)
class CacheAccess:
    cache: Cache
    read: bool
    write: bool
    memory_tier: DualCache | None
    tiered_writes: bool


def access(kwargs: Mapping[str, object], call_type: str) -> CacheAccess | None:
    import litellm
    from litellm.caching.caching_handler import cache_access, response_memory_tier

    cache: Final = litellm.cache
    if cache is None:
        return None
    model: Final = kwargs.get("model")
    read, write = cache_access(model if isinstance(model, str) else None, kwargs, call_type)
    if not (read or write):
        return None
    return CacheAccess(
        cache=cache,
        read=read,
        write=write,
        memory_tier=response_memory_tier(cache),
        tiered_writes=call_type != "anthropic_messages",
    )


def lookup(access: CacheAccess, kwargs: Mapping[str, object]) -> object:
    from litellm.caching.caching_handler import is_response_without_output

    result: Final = access.cache.get_cache(dynamic_cache_object=None, **kwargs)  # pyright: ignore[reportUnknownMemberType]  # Cache.get_cache kwargs are untyped
    return None if is_response_without_output(result) else result


async def alookup(access: CacheAccess, kwargs: Mapping[str, object]) -> object:
    from litellm.caching.caching_handler import is_response_without_output

    result: Final = await access.cache.async_get_cache(dynamic_cache_object=access.memory_tier, **kwargs)  # pyright: ignore[reportUnknownMemberType]  # Cache.async_get_cache kwargs are untyped
    if not is_response_without_output(result):
        return result
    if access.memory_tier is not None:
        access.memory_tier.in_memory_cache.delete_cache(access.cache.get_cache_key(**kwargs))
    return None


def store(access: CacheAccess, value: object, kwargs: Mapping[str, object]) -> None:
    from litellm.caching.caching_handler import is_response_without_output

    if not is_response_without_output(value):
        access.cache.add_cache(value, **kwargs)


async def astore(access: CacheAccess, value: object, kwargs: Mapping[str, object]) -> None:
    from litellm.caching.caching_handler import create_cache_write_task, is_response_without_output

    if is_response_without_output(value):
        return
    tier: Final = access.memory_tier if access.tiered_writes else None
    create_cache_write_task(lambda: access.cache.async_add_cache(value, dynamic_cache_object=tier, **kwargs))
