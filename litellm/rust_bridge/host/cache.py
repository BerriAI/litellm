from __future__ import annotations

import weakref
from collections.abc import AsyncIterator, Callable, Mapping
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # bounded cast of the served response

from pydantic import TypeAdapter

from litellm.litellm_core_utils.add_retry_fallback_headers import get_hidden_params_dict

if TYPE_CHECKING:
    from litellm.caching.caching import Cache
    from litellm.litellm_core_utils.litellm_logging import Logging

_RUST_RESPONSE_HEADER: Final = "x-litellm-rust"
_HEADERS: Final = TypeAdapter(dict[str, object])
_EVENTS: Final = TypeAdapter(list[str])
_HITS: Final[weakref.WeakKeyDictionary[Logging, Mapping[str, object]]] = weakref.WeakKeyDictionary()
_UNCLAMPED: Final[weakref.WeakKeyDictionary[Logging, object]] = weakref.WeakKeyDictionary()


def configured_cache() -> Cache | None:
    import litellm

    return litellm.cache


def _python_entry(response: Mapping[str, object]) -> Mapping[str, object]:
    hidden: Final = get_hidden_params_dict(response)
    headers: Final = hidden.get("additional_headers")
    if not isinstance(headers, Mapping) or _RUST_RESPONSE_HEADER not in headers:
        return response
    kept: Final = {
        name: value for name, value in _HEADERS.validate_python(headers).items() if name != _RUST_RESPONSE_HEADER
    }
    stored: Final = {name: value for name, value in hidden.items() if name != "additional_headers" or kept} | (
        {"additional_headers": kept} if kept else {}
    )
    entry: Final = {name: value for name, value in response.items() if name != "_hidden_params"}
    return {**entry, "_hidden_params": stored} if stored else entry


def store_response(
    cache: Cache,
    response: Mapping[str, object],
    /,
    **kwargs: object,  # kwargs-ok: the cache call's own arguments
) -> None:
    from litellm.caching.caching_handler import create_cache_write_task, is_response_without_output

    if is_response_without_output(response):
        return
    create_cache_write_task(
        lambda: cache.async_add_cache(  # pyright: ignore[reportUnknownMemberType]  # Cache.async_add_cache leaves its result untyped
            _python_entry(response),
            **kwargs,  # pyright: ignore[reportArgumentType]  # the cache call's own keyword arguments, as Python passes them
        )
    )


def store_verbatim(
    cache: Cache,
    response: Mapping[str, object],
    /,
    **kwargs: object,  # kwargs-ok: the cache call's own arguments
) -> None:
    from litellm.caching.caching_handler import create_cache_write_task, is_response_without_output

    if is_response_without_output(response):
        return
    create_cache_write_task(
        lambda: cache.async_add_cache(  # pyright: ignore[reportUnknownMemberType]  # Cache.async_add_cache leaves its result untyped
            response,
            **kwargs,  # pyright: ignore[reportArgumentType]  # the cache call's own keyword arguments, as Python passes them
        )
    )


async def store_stream(
    cache: Cache,
    entry: Mapping[str, object],
    /,
    **kwargs: object,  # kwargs-ok: the cache call's own arguments
) -> None:
    from litellm._logging import verbose_logger
    from litellm.caching.caching_handler import LLMCachingHandler, create_cache_write_task
    from litellm.types.caching import CACHED_STREAM_EVENTS_KEY

    events: Final = entry.get(CACHED_STREAM_EVENTS_KEY)
    handler: Final = getattr(kwargs.get("litellm_logging_obj"), "llm_caching_handler", None)
    preset_cache_key: Final = handler.preset_cache_key if isinstance(handler, LLMCachingHandler) else None
    cache_key_override: Final[Mapping[str, object]] = (
        {"cache_key": preset_cache_key} if preset_cache_key is not None else {}
    )

    async def write() -> None:
        try:
            await cache.async_add_cache(  # pyright: ignore[reportUnknownMemberType]  # Cache.async_add_cache leaves its result untyped
                {**entry, CACHED_STREAM_EVENTS_KEY: tuple(_EVENTS.validate_python(events))}
                if isinstance(events, list)
                else entry,
                dynamic_cache_object=handler.dual_cache if isinstance(handler, LLMCachingHandler) else None,
                **{**kwargs, **cache_key_override},  # pyright: ignore[reportArgumentType]  # the cache call's own keyword arguments, as Python passes them
            )
        except Exception as e:  # noqa: BLE001  # a cache write must never surface as a client-visible stream error
            verbose_logger.exception("Anthropic Messages stream cache write failed: %s", e)

    create_cache_write_task(write)


def caching_handler(
    **kwargs: object,  # kwargs-ok: the cache call's own arguments
) -> object | None:
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging) or not isinstance(logging_obj.llm_caching_handler, LLMCachingHandler):
        return None
    return logging_obj.llm_caching_handler


def wrap_stream(
    _cache: Cache,
    original_function: Callable[..., object],
    stream: object,
    read_handler: object | None,
    /,
    **kwargs: object,  # kwargs-ok: the cache call's own arguments
) -> object:
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return stream
    handler: Final = (
        read_handler
        if isinstance(read_handler, LLMCachingHandler)
        else LLMCachingHandler(
            original_function=original_function, request_kwargs=dict(kwargs), start_time=logging_obj.start_time
        )
    )
    return handler.wrap_streaming_result_for_cache(result=stream, call_type=original_function.__name__)


def remember_unclamped(logging_obj: object, max_tokens: object) -> None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    if isinstance(logging_obj, Logging):
        _UNCLAMPED[logging_obj] = max_tokens


async def lookup(
    _cache: Cache,
    original_function: Callable[..., object],
    /,
    **kwargs: object,  # kwargs-ok: the cache call's own arguments
) -> object | None:
    from litellm.litellm_core_utils.litellm_logging import Logging

    logging_obj: Final = kwargs.get("litellm_logging_obj")
    if not isinstance(logging_obj, Logging):
        return None
    return await _read(
        original_function,
        logging_obj,
        {**kwargs, "max_tokens": _UNCLAMPED.pop(logging_obj)} if logging_obj in _UNCLAMPED else kwargs,
    )


async def _read(
    original_function: Callable[..., object],
    logging_obj: Logging,
    kwargs: dict[str, object],  # mutable-ok: the read call's own keyword arguments, handed on as the handler takes them
) -> object | None:
    from litellm.caching.caching_handler import LLMCachingHandler

    handler: Final = LLMCachingHandler(
        original_function=original_function, request_kwargs=dict(kwargs), start_time=logging_obj.start_time
    )
    logging_obj.llm_caching_handler = handler
    model: Final = kwargs.get("model")
    model_name: Final = model if isinstance(model, str) else ""
    if not handler.reads_cache(model=model_name, kwargs=kwargs):
        return None
    read: Final = await handler.async_read_cache(
        model=model_name,
        original_function=original_function,
        logging_obj=logging_obj,
        call_type=original_function.__name__,
        kwargs=kwargs,
        args=(),
    )
    if not read.hit:
        return None
    _remember(logging_obj, read.cached_result)
    return read.cached_result


def _remember(logging_obj: Logging, cached: object) -> None:
    if isinstance(cached, Mapping):
        _HITS[logging_obj] = _HEADERS.validate_python(cached)


def _replace(response: object, cached: Mapping[str, object] | None) -> None:
    if isinstance(response, dict) and cached is not None:
        served: Final = cast(  # cast-ok: a Messages response is the str-keyed dict the route serves
            "dict[str, object]", response
        )
        served.clear()
        served.update(cached)


def stream_hit(logger: Logging) -> object | None:
    """The cached stream Python's caching handler returns for the hit this call read, if it read one."""
    from litellm.caching.caching_handler import (
        _set_cached_hidden_param,  # pyright: ignore[reportPrivateUsage]  # the @client cache hit's own hidden params
    )
    from litellm.llms.anthropic.messages.native.response_cache import (
        CachedAnthropicMessagesStreamIterator,
        convert_cached_anthropic_messages_result,
    )

    cached: Final = _HITS.pop(logger, None)
    handler: Final = logger.llm_caching_handler
    if cached is None or handler is None:
        return None
    stream: Final = convert_cached_anthropic_messages_result(
        cached_result=cached, logging_obj=logger, kwargs=handler.request_kwargs
    )
    if not isinstance(stream, CachedAnthropicMessagesStreamIterator):
        return None
    _set_cached_hidden_param(stream, "cache_key", hit_cache_key(logger))
    return stream


def stream_writer(logger: Logging, stream: object) -> object:
    """The stream as Python's caching handler hands a cache miss back. Rust stores the stream itself, so the
    writer never persists it again."""
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.llms.anthropic.messages.native.response_cache import AnthropicMessagesStreamCacheWriter

    handler: Final = logger.llm_caching_handler
    if not isinstance(handler, LLMCachingHandler):
        return stream
    wrapped: Final = handler.wrap_streaming_result_for_cache(
        result=cast("AsyncIterator[bytes]", stream),  # cast-ok: the public native stream yields SSE bytes
        call_type=handler.original_function.__name__,
    )
    if isinstance(wrapped, AnthropicMessagesStreamCacheWriter):
        wrapped.persisted = True
    return wrapped


def restore_hit(response: object, logger: Logging) -> None:
    from litellm.caching.caching_handler import (
        _set_cached_hidden_param,  # pyright: ignore[reportPrivateUsage]  # the @client cache hit's own hidden params
    )

    _replace(response, _HITS.pop(logger, None))
    _set_cached_hidden_param(response, "cache_hit", True)


def hit_cache_key(logger: Logging) -> object:
    from litellm.caching.caching_handler import LLMCachingHandler

    handler: Final = logger.llm_caching_handler
    return handler.hit_cache_key() if isinstance(handler, LLMCachingHandler) else None
