import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

from pydantic import TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.caching.caching import CacheMode
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.common_utils import AnthropicModelInfo
from litellm.llms.anthropic.experimental_pass_through.messages import handler as main
from litellm.rust_bridge.catalog import Delivery, Route, RouteContext
from litellm.rust_bridge.dispatch import PublicDispatch, call_hook
from litellm.rust_bridge.messages.entrypoints import (
    NATIVE_AMESSAGES,
    NATIVE_MESSAGES,
    LiteLLMMessagesRequest,
    NativeAmessages,
)
from litellm.rust_bridge.public_call import (
    bind,
    optional_bool,
    optional_mapping,
    optional_sequence,
    optional_str,
    signature,
)
from litellm.rust_bridge.response_cache import NativeResponseCacheRuntime, ResponseCacheRuntime
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse

__all__ = ("anthropic_messages", "anthropic_messages_handler")

MessagesResult: TypeAlias = AnthropicMessagesResponse | Iterator[bytes] | AsyncIterator[object]
PythonMessages: TypeAlias = Callable[..., MessagesResult | Coroutine[object, object, MessagesResult]]
PythonAmessages: TypeAlias = Callable[..., Awaitable[MessagesResult]]


def _python_messages() -> PythonMessages:
    return cast(  # cast-ok: forward the original call shape through the legacy handler
        PythonMessages,
        main.anthropic_messages_handler,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


def _python_amessages() -> PythonAmessages:
    return cast(  # cast-ok: forward the original call shape through the Python @client decorator
        PythonAmessages,
        main.anthropic_messages,  # noqa: TID251  # dispatch boundary owns this Python fallback
    )


_PYTHON_MESSAGES: Final = _python_messages()
_MESSAGES: Final = signature(_PYTHON_MESSAGES)
_PYTHON_AMESSAGES: Final = _python_amessages()
_AMESSAGES: Final = signature(_PYTHON_AMESSAGES)
_CALLBACKS_ADAPTER: Final = TypeAdapter(tuple[object, ...])


def _resolved_model_provider(model: str, provider: str | None) -> tuple[str, str] | None:
    try:
        resolved_model, resolved_provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=provider)
        return resolved_model, resolved_provider
    except Exception:
        return None


def _message_cache_point(value: object) -> bool:
    if not isinstance(value, Mapping):
        return False
    point: Final = cast(Mapping[object, object], value)  # cast-ok: runtime Mapping check narrows legacy input
    return point.get("location") == "message"


def _public_request(
    legacy: inspect.Signature, args: tuple[object, ...], kwargs: Mapping[str, object]
) -> LiteLLMMessagesRequest | None:
    fields: Final = bind(legacy, args, kwargs)
    if fields is None:
        return None
    model: Final = fields.get("model")
    messages: Final = optional_sequence(fields.get("messages"))
    max_tokens: Final = fields.get("max_tokens")
    if not isinstance(model, str) or messages is None or not isinstance(max_tokens, int):
        return None
    extras: Final = optional_mapping(fields.get("kwargs")) or MappingProxyType({})
    points: Final = extras.get("cache_control_injection_points")
    supported_points: Final = isinstance(points, list) and all(
        _message_cache_point(point)
        for point in cast(list[object], points)  # cast-ok: list elements are validated above
    )
    provider: Final = optional_str(fields.get("custom_llm_provider"))
    resolved: Final = _resolved_model_provider(model, provider)
    speed: Final = extras.get("speed")
    if speed is not None and resolved is not None and resolved[1] == "anthropic":
        if not AnthropicConfig._model_supports_speed_param(  # pyright: ignore[reportPrivateUsage]  # Python validates unsupported speed before send
            resolved[0], resolved[1]
        ):
            return None
    auto_prompt_cache: Final = litellm.enable_anthropic_prompt_caching or extras.get("enable_prompt_caching") is True
    proxy_request: Final = extras.get("proxy_server_request")
    if auto_prompt_cache and isinstance(proxy_request, Mapping):
        proxy_fields: Final = cast(  # cast-ok: runtime Mapping check validates this legacy request
            Mapping[object, object], proxy_request
        )
        proxy_headers: Final = proxy_fields.get("headers")
        if isinstance(proxy_headers, Mapping):
            headers: Final = cast(  # cast-ok: runtime Mapping check validates these legacy headers
                Mapping[object, object], proxy_headers
            )
            user_agent: Final = next(
                (value for name, value in headers.items() if isinstance(name, str) and name.lower() == "user-agent"),
                None,
            )
            if isinstance(user_agent, str) and user_agent.startswith(("claude-cli/", "claude-code/")):
                return None
    sampling: Final = (
        (fields.get("temperature") is not None and fields.get("temperature") != 1)
        or fields.get("top_p") is not None
        or fields.get("top_k") is not None
    )
    if resolved is not None and resolved[1] == "anthropic" and sampling:
        if not AnthropicModelInfo._supports_sampling_params(  # pyright: ignore[reportPrivateUsage]  # share the Python Messages model gate
            resolved[0]
        ):
            return None
    if (
        fields.get("client") is not None
        or (points is not None and not supported_points)
        or extras.get("additional_drop_params") is not None
        or extras.get("mock_response") is not None
    ):
        return None
    return LiteLLMMessagesRequest(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        stream=optional_bool(fields.get("stream")),
        api_key=optional_str(fields.get("api_key")),
        api_base=optional_str(fields.get("api_base")),
        custom_llm_provider=resolved[1] if resolved is not None else None,
        kwargs=extras,
    )


def _context(request: LiteLLMMessagesRequest) -> RouteContext:
    return RouteContext(
        Route.MESSAGES,
        provider=request.custom_llm_provider,
        model=request.model,
        delivery=Delivery.STREAMING if request.stream else Delivery.COMPLETED,
    )


_DISPATCH: Final = PublicDispatch(
    route=Route.MESSAGES,
    request=lambda args, kwargs: _public_request(_MESSAGES, args, kwargs),
    context=_context,
    bypass=lambda request: request.kwargs.get("is_async") is True,
)

_ADISPATCH: Final = PublicDispatch(
    route=Route.MESSAGES,
    request=lambda args, kwargs: _public_request(_AMESSAGES, args, kwargs),
    context=_context,
)


def _cache_call_kwargs(args: tuple[object, ...], kwargs: Mapping[str, object]) -> Mapping[str, object]:
    fields: Final = bind(_AMESSAGES, args, kwargs)
    assert fields is not None
    extras: Final = optional_mapping(fields.get("kwargs")) or MappingProxyType({})
    return MappingProxyType({**{name: value for name, value in fields.items() if name != "kwargs"}, **extras})


def _legacy_cache_now() -> datetime:
    return datetime.now(timezone.utc).astimezone().replace(tzinfo=None)


def _has_custom_deployment_hook(kwargs: Mapping[str, object]) -> bool:
    from litellm.integrations.custom_logger import CustomLogger

    def overridden(callback: object) -> bool:
        if not isinstance(callback, CustomLogger):
            return False
        actual: Final = cast(  # cast-ok: compare hook identity without its untyped return
            object, type(callback).async_pre_call_deployment_hook
        )
        default: Final = cast(  # cast-ok: compare hook identity without its untyped return
            object, CustomLogger.async_pre_call_deployment_hook
        )
        return actual is not default

    call_callbacks: Final = kwargs.get("callbacks")
    dynamic: Final = (
        _CALLBACKS_ADAPTER.validate_python(call_callbacks) if isinstance(call_callbacks, list | tuple) else ()
    )
    registered: Final = _CALLBACKS_ADAPTER.validate_python(litellm.callbacks)  # pyright: ignore[reportUnknownMemberType]  # validate the legacy callback registry
    return any(overridden(callback) for callback in (*registered, *dynamic))


def _native_cache_hit(
    cached: object,
    cache_key: str,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> MessagesResult:
    from litellm._uuid import uuid
    from litellm.caching.caching_handler import LLMCachingHandler
    from litellm.llms.anthropic.experimental_pass_through.messages.response_cache import (
        CachedAnthropicMessagesStreamIterator,
    )
    from litellm.utils import Rules, function_setup

    started: Final = _legacy_cache_now()
    call_kwargs: Final = _cache_call_kwargs(args, kwargs)
    call_id: Final = call_kwargs.get("litellm_call_id") or str(uuid.uuid4())
    logging_obj, prepared_kwargs = function_setup(
        "anthropic_messages",
        Rules(),
        started,
        **{**call_kwargs, "litellm_call_id": call_id},  # pyright: ignore[reportArgumentType]  # dynamic options do not set is_async_call
    )
    handler: Final = LLMCachingHandler(
        original_function=_PYTHON_AMESSAGES,
        request_kwargs=prepared_kwargs,
        start_time=started,
    )
    handler.preset_cache_key = cache_key
    logging_obj._llm_caching_handler = handler  # pyright: ignore[reportPrivateUsage]  # cache-hit callbacks read this handler
    model_value: Final = prepared_kwargs.get("model")
    model: Final = model_value if isinstance(model_value, str) else ""
    resolved_model, provider, _, _ = litellm.get_llm_provider(
        model=model,
        custom_llm_provider=prepared_kwargs.get("custom_llm_provider"),
        api_base=prepared_kwargs.get("api_base"),
        api_key=prepared_kwargs.get("api_key"),
    )
    handler._update_litellm_logging_obj_environment(  # pyright: ignore[reportPrivateUsage]  # preserve cache-hit callback context
        logging_obj=logging_obj,
        model=resolved_model,
        kwargs=prepared_kwargs,
        cached_result=cached,
        is_async=True,
        custom_llm_provider=provider,
    )
    response: Final = handler._convert_cached_result_to_model_response(  # pyright: ignore[reportPrivateUsage]  # reuse the public cache-hit response conversion
        cached_result=cached,
        call_type="anthropic_messages",
        kwargs=prepared_kwargs,
        logging_obj=logging_obj,
        model=resolved_model,
        args=args,
        custom_llm_provider=provider,
    )
    if not isinstance(response, CachedAnthropicMessagesStreamIterator):
        handler._async_log_cache_hit_on_callbacks(  # pyright: ignore[reportPrivateUsage]  # preserve success callback timing
            logging_obj=logging_obj,
            cached_result=response,
            start_time=started,
            end_time=_legacy_cache_now(),
            cache_hit=True,
        )
    if isinstance(response, CachedAnthropicMessagesStreamIterator):
        response._hidden_params["cache_key"] = cache_key  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # cached stream exposes this metadata
    return cast(MessagesResult, response)  # cast-ok: converter returns the public Messages result


async def _native_amessages_with_native_cache(
    hook: NativeAmessages,
    request: LiteLLMMessagesRequest,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
    runtime: ResponseCacheRuntime,
    selected: NativeResponseCacheRuntime,
) -> MessagesResult:
    from litellm.caching.caching_handler import create_cache_write_task
    from litellm.llms.anthropic.experimental_pass_through.messages.response_cache import (
        AnthropicMessagesStreamCacheWriter,
    )

    cache: Final = litellm.cache
    assert cache is not None
    merged: Final = _cache_call_kwargs(args, kwargs)
    call_kwargs: Final = MappingProxyType(
        {name: value for name, value in merged.items() if name != "metadata" or value is not None}
    )
    cache_request: Final = runtime.request(cache, call_kwargs)
    if cache_request is None:
        return await hook(request, args, kwargs)
    control_value: Final = call_kwargs.get("cache")
    controls: Final[Mapping[object, object]] = (
        cast(Mapping[object, object], control_value)  # cast-ok: runtime Mapping check narrows caller controls
        if isinstance(control_value, Mapping)
        else MappingProxyType({})
    )
    enabled: Final = cache.mode == CacheMode.default_on or controls.get("use-cache") is True
    read: Final = enabled and call_kwargs.get("caching") is not False and controls.get("no-cache") is not True
    write: Final = enabled and controls.get("no-store") is not True
    if read:
        try:
            cached: Final = await selected.async_lookup(cache_request)
        except Exception as error:  # noqa: BLE001  # cache read failure must not suppress inference
            verbose_logger.exception("Anthropic Messages cache lookup failed: %s", error)
        else:
            if cached is not None:
                return _native_cache_hit(cached, cache_request["key"]["preset"], args, kwargs)

    result: Final = await hook(request, args, kwargs)
    if not write:
        return result
    if isinstance(result, AsyncIterator):

        async def store_stream(payload: Mapping[str, object]) -> None:
            await selected.async_store(cache_request, payload)

        return AnthropicMessagesStreamCacheWriter(
            stream=cast(AsyncIterator[bytes | str], result),  # cast-ok: Messages stream yields provider SSE bytes
            native_store=store_stream,
        )

    async def store_result() -> None:
        try:
            await selected.async_store(cache_request, result)
        except Exception as error:  # noqa: BLE001  # cache write failures do not change the provider response
            verbose_logger.exception("Anthropic Messages cache write failed: %s", error)

    create_cache_write_task(store_result)
    return result


async def _native_amessages_with_cache(
    hook: NativeAmessages,
    request: LiteLLMMessagesRequest,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> MessagesResult:
    cache: Final = litellm.cache
    cache_control: Final = kwargs.get("cache")
    controls: Final[Mapping[object, object]] = (
        cast(Mapping[object, object], cache_control)  # cast-ok: runtime Mapping check narrows legacy cache controls
        if isinstance(cache_control, Mapping)
        else MappingProxyType({})
    )
    if cache is None or cache.supported_call_types is None or "anthropic_messages" not in cache.supported_call_types:
        return await hook(request, args, kwargs)

    runtime: Final = cache._native_cache  # pyright: ignore[reportPrivateUsage]  # use the backend selected when litellm.cache was configured
    if runtime is not None and kwargs.get("litellm_logging_obj") is None and not _has_custom_deployment_hook(kwargs):
        from litellm.rust_bridge import _native

        selected: Final = _native._CacheResolver(litellm).resolve()  # pyright: ignore[reportPrivateUsage]  # resolve the configured backend in Rust
        if selected.kind == "native":
            return await _native_amessages_with_native_cache(hook, request, args, kwargs, runtime, selected)

    from litellm.caching.caching_handler import LLMCachingHandler

    call_kwargs: Final = dict(kwargs)  # mutable-ok: legacy caching handler edits owned request kwargs
    caching_handler: Final = LLMCachingHandler(
        original_function=_PYTHON_AMESSAGES,
        request_kwargs=call_kwargs,
        start_time=_legacy_cache_now(),
    )
    cached: Final = (
        await caching_handler._retrieve_from_cache(  # pyright: ignore[reportPrivateUsage]  # reuse the public decorator's key and cache policy
            call_type="anthropic_messages",
            kwargs=call_kwargs,
            args=args,
        )
        if kwargs.get("caching") is None and controls.get("no-cache") is not True
        else None
    )
    if cached is not None:
        return await _PYTHON_AMESSAGES(*args, **kwargs)

    result: Final = await hook(request, args, kwargs)
    if isinstance(result, AsyncIterator):
        return caching_handler.wrap_streaming_result_for_cache(result, "anthropic_messages")
    await caching_handler.async_set_cache(  # pyright: ignore[reportUnknownMemberType]  # legacy cache handler has untyped callback return
        result, original_function=_PYTHON_AMESSAGES, kwargs=call_kwargs, args=args
    )
    return result


def anthropic_messages_handler(
    *args: object,
    **kwargs: object,  # kwargs-ok: preserve the public Anthropic Messages call shape
) -> MessagesResult | Coroutine[object, object, MessagesResult]:
    python: Final = _PYTHON_MESSAGES
    return _DISPATCH.run(
        args,
        kwargs,
        python=python,
        binding=NATIVE_MESSAGES,
        native=call_hook,
    )


async def anthropic_messages(*args: object, **kwargs: object) -> MessagesResult:  # kwargs-ok: public call shape
    python: Final = _PYTHON_AMESSAGES
    return await _ADISPATCH.arun(
        args,
        kwargs,
        python=python,
        binding=NATIVE_AMESSAGES,
        native=_native_amessages_with_cache,
    )


anthropic_messages_handler.__doc__ = _PYTHON_MESSAGES.__doc__
anthropic_messages_handler.__wrapped__ = _PYTHON_MESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
anthropic_messages.__doc__ = _PYTHON_AMESSAGES.__doc__
anthropic_messages.__wrapped__ = _PYTHON_AMESSAGES  # pyright: ignore[reportFunctionMemberAccess]  # inspect.signature follows __wrapped__ to the legacy signature
