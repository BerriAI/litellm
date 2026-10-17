from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.caching.caching import Cache


@dataclass(frozen=True, slots=True)
class CacheAccess:
    cache: Cache
    read: bool
    write: bool


def access(kwargs: Mapping[str, object], call_type: str) -> CacheAccess | None:
    import litellm
    from litellm.caching.caching_handler import may_read_cache, may_store_cache

    cache: Final = litellm.cache
    if cache is None or cache.should_use_cache(**kwargs) is not True:
        return None
    model: Final = kwargs.get("model")
    model_name: Final = model if isinstance(model, str) else None
    read: Final = may_read_cache(model_name, kwargs, call_type)
    write: Final = may_store_cache(model_name, kwargs, call_type)
    return CacheAccess(cache=cache, read=read, write=write) if read or write else None


def lookup(cache: Cache, kwargs: Mapping[str, object]) -> object:
    return cache.get_cache(dynamic_cache_object=None, **kwargs)  # pyright: ignore[reportUnknownMemberType]  # Cache.get_cache kwargs are untyped


async def alookup(cache: Cache, kwargs: Mapping[str, object]) -> object:
    return await cache.async_get_cache(dynamic_cache_object=None, **kwargs)  # pyright: ignore[reportUnknownMemberType]  # Cache.async_get_cache kwargs are untyped


def store(cache: Cache, value: object, kwargs: Mapping[str, object]) -> None:
    cache.add_cache(value, **kwargs)


async def astore(cache: Cache, value: object, kwargs: Mapping[str, object]) -> None:
    await cache.async_add_cache(value, dynamic_cache_object=None, **kwargs)
