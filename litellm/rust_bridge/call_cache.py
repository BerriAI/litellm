"""The response-cache policy ``@client`` applies around a Python call, for a native call.

Reads and writes go through ``litellm.cache``'s own methods so a native call and a Python
call honor the same cache selection at use time.
"""

from __future__ import annotations

import datetime
import time
from collections.abc import Awaitable, Callable, Mapping
from types import MappingProxyType
from typing import Final, cast

import litellm
from litellm._logging import verbose_logger
from litellm.caching.caching_handler import (
    LLMCachingHandler,
    _drop_logging_obj_from_kwargs,  # pyright: ignore[reportPrivateUsage]  # the shared cycle-breaking helper
    create_cache_write_task,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def _original_function(call_type: str) -> Callable[..., None]:
    function: Final[Callable[..., None]] = lambda *args: None  # noqa: E731  # stand-in for the wrapped call; only __name__ is read
    function.__name__ = call_type  # pyright: ignore[reportFunctionMemberAccess]  # setting the wrapped name like functools.wraps
    return function


def _control(kwargs: Mapping[str, object]) -> Mapping[str, object]:
    control: Final = kwargs.get("cache")
    return cast(Mapping[str, object], control) if isinstance(control, Mapping) else MappingProxyType({})


def _lookup_enabled(call_type: str, kwargs: Mapping[str, object]) -> bool:
    cache: Final = litellm.cache
    if cache is None or cache.supported_call_types is None or call_type not in cache.supported_call_types:
        return False
    return (kwargs.get("caching") is None or kwargs.get("caching") is True) and (
        _control(kwargs).get("no-cache") is not True
    )


def _request_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    request: Final = _drop_logging_obj_from_kwargs(dict(kwargs))
    if request.get("metadata") is None:
        request.pop("metadata", None)
    return request


def _cache_key(kwargs: Mapping[str, object], request_kwargs: Mapping[str, object]) -> str | None:
    cache: Final = litellm.cache
    if cache is None:
        return None
    preset: Final = kwargs.get("cache_key")
    if isinstance(preset, str):
        return preset
    get_cache_key: Final = cast(Callable[..., str | None], getattr(cache, "get_cache_key"))
    return get_cache_key(**request_kwargs)


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
        request_kwargs=_drop_logging_obj_from_kwargs(dict(request_kwargs)),
        start_time=datetime.datetime.now(),
    )
    handler.preset_cache_key = cache_key
    handler._update_litellm_logging_obj_environment(  # pyright: ignore[reportPrivateUsage]  # one shared hit-marking path with the Python wrapper
        logging_obj=logging_obj,
        model=str(kwargs.get("model") or ""),
        kwargs=dict(kwargs),
        cached_result=cached,
        is_async=is_async,
        custom_llm_provider=provider if isinstance(provider, str) else None,
        cache_duration_ms=duration_ms,
    )
    logging_obj.model_call_details["cache_hit"] = True


def _get_sync(cache: object, key: str | None, request_kwargs: Mapping[str, object]) -> object:
    get_cache: Final = cast(Callable[..., object], getattr(cache, "get_cache"))
    return get_cache(cache_key=key, **dict(request_kwargs))


async def _get(cache: object, key: str | None, request_kwargs: Mapping[str, object]) -> object:
    async_get_cache: Final = cast(
        Callable[..., Awaitable[object]],
        getattr(cache, "async_get_cache"),
    )
    supports_async: Final = cast(
        bool, getattr(cache, "_supports_async")()
    )  # same support check the caching handler runs
    return (
        await async_get_cache(cache_key=key, **dict(request_kwargs))
        if supports_async
        else _get_sync(cache, key, request_kwargs)
    )


async def lookup(call_type: str, kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    cache: Final = litellm.cache
    if cache is None or not _lookup_enabled(call_type, kwargs):
        return None
    started: Final = time.perf_counter()
    request_kwargs: Final = _request_kwargs(kwargs)
    key: Final = _cache_key(kwargs, request_kwargs)
    request_kwargs.pop("cache_key", None)
    result: Final = await _get(cache, key, request_kwargs)
    if not isinstance(result, Mapping):
        return None
    found: Final = cast(Mapping[str, object], result)
    _mark_hit(kwargs, call_type, request_kwargs, key, found, True, (time.perf_counter() - started) * 1000)
    return found


def lookup_sync(call_type: str, kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    cache: Final = litellm.cache
    if cache is None or not _lookup_enabled(call_type, kwargs):
        return None
    started: Final = time.perf_counter()
    request_kwargs: Final = _request_kwargs(kwargs)
    key: Final = _cache_key(kwargs, request_kwargs)
    request_kwargs.pop("cache_key", None)
    result: Final = _get_sync(cache, key, request_kwargs)
    if not isinstance(result, Mapping):
        return None
    found: Final = cast(Mapping[str, object], result)
    _mark_hit(kwargs, call_type, request_kwargs, key, found, False, (time.perf_counter() - started) * 1000)
    return found


def _store_enabled(call_type: str, kwargs: Mapping[str, object]) -> bool:
    cache: Final = litellm.cache
    return (
        cache is not None
        and cache.supported_call_types is not None
        and call_type in cache.supported_call_types
        and _control(kwargs).get("no-store") is not True
    )


def _parent_span(request_kwargs: Mapping[str, object]) -> object:
    from litellm.litellm_core_utils.core_helpers import (
        _get_parent_otel_span_from_kwargs,  # pyright: ignore[reportPrivateUsage,reportUnknownVariableType]  # the same untyped span helper the Python path adds to cache kwargs
    )

    span: Final = cast(
        Callable[..., object],
        _get_parent_otel_span_from_kwargs,
    )
    return span(dict(request_kwargs))


def store(call_type: str, kwargs: Mapping[str, object], response: Mapping[str, object]) -> None:
    if not _store_enabled(call_type, kwargs):
        return
    cache: Final = litellm.cache
    assert cache is not None
    new_kwargs: Final = _request_kwargs(kwargs)
    new_kwargs["parent_otel_span"] = _parent_span(new_kwargs)
    async_add_cache: Final = cast(Callable[..., Awaitable[None]], getattr(cache, "async_add_cache"))
    try:
        create_cache_write_task(lambda: async_add_cache(dict(response), **new_kwargs))
    except Exception as error:  # noqa: BLE001  # a cache write failure never fails the call
        verbose_logger.error("LiteLLM Cache: exception in add_cache: %s", error)


def store_sync(call_type: str, kwargs: Mapping[str, object], response: Mapping[str, object]) -> None:
    if not _store_enabled(call_type, kwargs):
        return
    cache: Final = litellm.cache
    assert cache is not None
    new_kwargs: Final = _request_kwargs(kwargs)
    new_kwargs["parent_otel_span"] = _parent_span(new_kwargs)
    add_cache: Final = cast(Callable[..., None], getattr(cache, "add_cache"))
    try:
        add_cache(dict(response), **new_kwargs)
    except Exception as error:  # noqa: BLE001  # a cache write failure never fails the call
        verbose_logger.error("LiteLLM Cache: exception in add_cache: %s", error)
