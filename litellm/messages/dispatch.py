import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Coroutine, Iterator, Mapping
from datetime import datetime
from types import MappingProxyType
from typing import Final, TypeAlias, cast  # noqa: TID251  # native binding selects a sync result or an async awaitable

import litellm
from litellm.integrations.custom_logger import CustomLogger
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


def _resolved_model_provider(model: str, provider: str | None) -> tuple[str, str] | None:
    try:
        resolved_model, resolved_provider, _, _ = litellm.get_llm_provider(model=model, custom_llm_provider=provider)
        return resolved_model, resolved_provider
    except Exception:
        return None


def _has_python_request_hook() -> bool:
    return any(
        isinstance(callback, CustomLogger)
        and (
            type(callback).async_pre_request_hook is not CustomLogger.async_pre_request_hook
            or type(callback).async_pre_call_deployment_hook is not CustomLogger.async_pre_call_deployment_hook
        )
        for callback in litellm.callbacks
    )


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
    output_config: Final = extras.get("output_config")
    thinking: Final = fields.get("thinking")
    provider: Final = optional_str(fields.get("custom_llm_provider"))
    resolved: Final = _resolved_model_provider(model, provider)
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
        _has_python_request_hook()
        or litellm.enable_anthropic_prompt_caching
        or fields.get("client") is not None
        or extras.get("reasoning_effort") is not None
        or extras.get("speed") is not None
        or extras.get("cache_control_injection_points") is not None
        or extras.get("additional_drop_params") is not None
        or extras.get("enable_prompt_caching") is True
        or extras.get("mock_response") is not None
        or isinstance(output_config, Mapping)
        and output_config.get("effort") is not None
        or thinking is not None
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


async def _native_amessages_with_cache(
    hook: NativeAmessages,
    request: LiteLLMMessagesRequest,
    args: tuple[object, ...],
    kwargs: Mapping[str, object],
) -> MessagesResult:
    cache: Final = litellm.cache
    cache_control: Final = kwargs.get("cache")
    controls: Final = cache_control if isinstance(cache_control, Mapping) else MappingProxyType({})
    if cache is None or cache.supported_call_types is None or "anthropic_messages" not in cache.supported_call_types:
        return await hook(request, args, kwargs)

    from litellm.caching.caching_handler import LLMCachingHandler

    call_kwargs: Final = dict(kwargs)
    caching_handler: Final = LLMCachingHandler(
        original_function=_PYTHON_AMESSAGES,
        request_kwargs=call_kwargs,
        start_time=datetime.now(),
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
    if request.stream:
        return caching_handler.wrap_streaming_result_for_cache(result, "anthropic_messages")
    await caching_handler.async_set_cache(result, original_function=_PYTHON_AMESSAGES, kwargs=call_kwargs, args=args)
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
