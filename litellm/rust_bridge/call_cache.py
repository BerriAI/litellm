"""The response-cache policy ``@client`` applies around a Python call, for a native call.

Reads and writes go through ``litellm.cache``'s own methods so a native call and a Python
call honor the same cache selection at use time.
"""

from __future__ import annotations

import datetime
from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final, cast

import litellm
from litellm._logging import verbose_logger
from litellm.caching.caching import Cache
from litellm.caching.caching_handler import LLMCachingHandler, create_cache_write_task
from litellm.litellm_core_utils.core_helpers import (
    _get_parent_otel_span_from_kwargs,  # pyright: ignore[reportPrivateUsage,reportUnknownVariableType]  # the same untyped span helper the Python path adds to cache kwargs
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def _original_function(call_type: str) -> Callable[..., None]:
    def wrapped(*args: object) -> None:
        return None

    wrapped.__name__ = call_type  # pyright: ignore[reportFunctionMemberAccess]  # setting the wrapped name like functools.wraps
    return wrapped


def _control(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    control: Final = kwargs.get("cache")
    return (
        cast(Mapping[str, object], control)  # cast-ok: isinstance narrows the cache control value to a Mapping
        if isinstance(control, Mapping)
        else MappingProxyType({})
    )


def _lookup_enabled(cache: Cache, call_type: str, kwargs: Mapping[str, object]) -> bool:
    if cache.supported_call_types is None or call_type not in cache.supported_call_types:
        return False
    return (kwargs.get("caching") is None or kwargs.get("caching") is True) and (
        _control(kwargs).get("no-cache") is not True
    )


def _request_kwargs(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(
        {
            name: value
            for name, value in kwargs.items()
            if name != "litellm_logging_obj" and (name != "metadata" or value is not None)
        }
    )


def _cache_key(cache: Cache, kwargs: Mapping[str, object], request_kwargs: Mapping[str, object]) -> str | None:
    preset: Final = kwargs.get("cache_key")
    if isinstance(preset, str):
        return preset
    return cache.get_cache_key(**request_kwargs)  # pyright: ignore[reportUnknownMemberType]  # get_cache_key is untyped on Cache


def _logging_obj(kwargs: Mapping[str, object]) -> LiteLLMLoggingObj | None:
    logging_obj: Final = kwargs.get("litellm_logging_obj")
    return logging_obj if isinstance(logging_obj, LiteLLMLoggingObj) else None


def _mark_hit(
    kwargs: Mapping[str, object],
    call_type: str,
    request_kwargs: Mapping[str, object],
    cache_key: str | None,
    cached: object,
    is_async: bool,
    duration_ms: float,
) -> None:
    logging_obj: Final = _logging_obj(kwargs)
    if logging_obj is None:
        return
    provider: Final = kwargs.get("custom_llm_provider")
    handler: Final = LLMCachingHandler(
        original_function=_original_function(call_type),
        request_kwargs=dict(request_kwargs),  # mutable-ok: the caching handler keeps an owned kwargs copy
        start_time=datetime.datetime.now(),
    )
    handler.preset_cache_key = cache_key
    handler._update_litellm_logging_obj_environment(  # pyright: ignore[reportPrivateUsage]  # one shared hit-marking path with the Python wrapper
        logging_obj=logging_obj,
        model=str(kwargs.get("model") or ""),
        kwargs=dict(kwargs),  # mutable-ok: Logging mutates call details into this kwargs copy
        cached_result=cached,
        is_async=is_async,
        custom_llm_provider=provider if isinstance(provider, str) else None,
        cache_duration_ms=duration_ms,
    )
    logging_obj.model_call_details["cache_hit"] = True


def mark_hit(
    call_type: str,
    kwargs: Mapping[str, object],
    cached: Mapping[str, object],
    is_async: bool,
    duration_ms: float,
) -> None:
    if _logging_obj(kwargs) is None:
        return
    cache: Final = litellm.cache
    if cache is None:
        return
    request: Final = _request_kwargs(kwargs)
    key: Final = _cache_key(cache, kwargs, request)
    request_kwargs: Final = MappingProxyType({name: value for name, value in request.items() if name != "cache_key"})
    _mark_hit(kwargs, call_type, request_kwargs, key, cached, is_async, duration_ms)


async def lookup(call_type: str, kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    cache: Final = litellm.cache
    if cache is None or not _lookup_enabled(cache, call_type, kwargs):
        return None
    request: Final = _request_kwargs(kwargs)
    key: Final = _cache_key(cache, kwargs, request)
    request_kwargs: Final = MappingProxyType({name: value for name, value in request.items() if name != "cache_key"})
    supports_async: Final = cache._supports_async() is True  # pyright: ignore[reportPrivateUsage,reportAny]  # same support check the caching handler runs
    result: Final = (  # pyright: ignore[reportUnknownVariableType]  # untyped Cache entry point
        await cache.async_get_cache(cache_key=key, **request_kwargs)  # pyright: ignore[reportUnknownMemberType,reportArgumentType]  # untyped Cache entry point takes arbitrary call kwargs
        if supports_async
        else cache.get_cache(cache_key=key, **request_kwargs)  # pyright: ignore[reportUnknownMemberType,reportArgumentType]  # untyped Cache entry point takes arbitrary call kwargs
    )
    if not isinstance(result, Mapping):
        return None
    found: Final = cast(Mapping[str, object], result)  # cast-ok: isinstance narrows the hit to a Mapping
    return found


def lookup_sync(call_type: str, kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    cache: Final = litellm.cache
    if cache is None or not _lookup_enabled(cache, call_type, kwargs):
        return None
    request: Final = _request_kwargs(kwargs)
    key: Final = _cache_key(cache, kwargs, request)
    request_kwargs: Final = MappingProxyType({name: value for name, value in request.items() if name != "cache_key"})
    result: Final = cache.get_cache(cache_key=key, **request_kwargs)  # pyright: ignore[reportUnknownMemberType,reportArgumentType,reportUnknownVariableType]  # untyped Cache entry point takes arbitrary call kwargs
    if not isinstance(result, Mapping):
        return None
    found: Final = cast(Mapping[str, object], result)  # cast-ok: isinstance narrows the hit to a Mapping
    return found


def _store_enabled(cache: Cache, call_type: str, kwargs: Mapping[str, object]) -> bool:
    return (
        cache.supported_call_types is not None
        and call_type in cache.supported_call_types
        and _control(kwargs).get("no-store") is not True
    )


def _store_kwargs(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    request: Final = _request_kwargs(kwargs)
    return MappingProxyType(
        {
            **request,
            "parent_otel_span": _get_parent_otel_span_from_kwargs(
                dict(request)  # mutable-ok: the otel span helper takes a plain dict
            ),
        }
    )


def store(call_type: str, kwargs: Mapping[str, object], response: Mapping[str, object]) -> None:
    cache: Final = litellm.cache
    if cache is None or not _store_enabled(cache, call_type, kwargs):
        return
    new_kwargs: Final = _store_kwargs(kwargs)
    try:
        create_cache_write_task(lambda: cache.async_add_cache(response, **new_kwargs))  # pyright: ignore[reportUnknownMemberType,reportArgumentType]  # untyped Cache entry point takes arbitrary call kwargs
    except Exception as error:  # noqa: BLE001  # a cache write failure never fails the call
        verbose_logger.error("LiteLLM Cache: exception in add_cache: %s", error)


def store_sync(call_type: str, kwargs: Mapping[str, object], response: Mapping[str, object]) -> None:
    cache: Final = litellm.cache
    if cache is None or not _store_enabled(cache, call_type, kwargs):
        return
    new_kwargs: Final = _store_kwargs(kwargs)
    try:
        cache.add_cache(response, **new_kwargs)  # pyright: ignore[reportUnknownMemberType]  # untyped Cache entry point takes arbitrary call kwargs
    except Exception as error:  # noqa: BLE001  # a cache write failure never fails the call
        verbose_logger.error("LiteLLM Cache: exception in add_cache: %s", error)
