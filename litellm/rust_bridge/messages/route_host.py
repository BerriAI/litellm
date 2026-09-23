from __future__ import annotations

import inspect
from collections.abc import Awaitable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, Protocol, cast  # noqa: TID251  # narrows legacy Python values at the native boundary

from pydantic import TypeAdapter

import litellm
from litellm.integrations.anthropic_cache_control_hook import AnthropicCacheControlHook
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.llms.anthropic.common_utils import (
    AnthropicModelInfo,
    flatten_unencrypted_web_search_results_in_anthropic_messages,
    sanitize_tool_use_ids_in_anthropic_messages,
    strip_empty_content_blocks_from_anthropic_messages,
    supports_anthropic_cache_control,
)
from litellm.llms.anthropic.experimental_pass_through.messages import handler as main
from litellm.llms.anthropic.experimental_pass_through.utils import is_reasoning_auto_summary_enabled
from litellm.rust_bridge import failures
from litellm.rust_bridge.messages.entrypoints import LiteLLMMessagesRequest
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse


class _HookRunner(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[Mapping[str, object]],
        tools: Sequence[Mapping[str, object]] | None,
        stream: bool | None,
        custom_llm_provider: str | None,
        **kwargs: object,  # kwargs-ok: caller hooks use the dynamic Messages option set
    ) -> Awaitable[Mapping[str, object]]: ...


class _CacheInjector(Protocol):
    def __call__(
        self,
        messages: Sequence[Mapping[str, object]],
        system: str | Sequence[Mapping[str, object]] | None,
        kwargs: Mapping[str, object],
        model: str | None,
        custom_llm_provider: str | None,
        tools: Sequence[Mapping[str, object]] | None,
        api_base: str | None,
    ) -> tuple[Sequence[Mapping[str, object]], str | Sequence[Mapping[str, object]] | None]: ...


class _DirectHandler(Protocol):
    def __call__(self, **kwargs: object) -> object: ...  # kwargs-ok: legacy handler has dynamic Messages options


_MESSAGES_ADAPTER: Final = TypeAdapter(Sequence[Mapping[str, object]])
_SYSTEM_ADAPTER: Final[TypeAdapter[str | Sequence[Mapping[str, object]] | None]] = TypeAdapter(
    str | Sequence[Mapping[str, object]] | None
)
_OPTIONAL_MESSAGES_ADAPTER: Final[TypeAdapter[Sequence[Mapping[str, object]] | None]] = TypeAdapter(
    Sequence[Mapping[str, object]] | None
)
_MAPPING_ADAPTER: Final = TypeAdapter(Mapping[str, object])


def response(value: Mapping[str, object]) -> AnthropicMessagesResponse:
    return cast(  # cast-ok: AnthropicMessagesResponse is a TypedDict over the normalized native payload
        AnthropicMessagesResponse,
        dict(value),  # mutable-ok: the public Messages response is a TypedDict the caller may annotate in place
    )


def arguments(request: LiteLLMMessagesRequest) -> Mapping[str, object]:
    return request.kwargs


def messages_features(model: str, provider: str) -> Mapping[str, object]:
    return MappingProxyType(
        {
            "adaptive": AnthropicModelInfo._is_adaptive_thinking_model(model, provider),  # pyright: ignore[reportPrivateUsage]  # use the resolved model map
            "always_on": AnthropicModelInfo._is_always_on_thinking_model(model, provider),  # pyright: ignore[reportPrivateUsage]  # use the resolved model map
            "legacy": AnthropicModelInfo._supports_legacy_thinking(model, provider),  # pyright: ignore[reportPrivateUsage]  # use the resolved model map
            "reasoning": AnthropicModelInfo._supports_model_capability(model, "supports_reasoning", provider),  # pyright: ignore[reportPrivateUsage]  # use the resolved model map
            "effort": AnthropicConfig._model_supports_effort_param(model, provider),  # pyright: ignore[reportPrivateUsage]  # use the resolved model map
            "xhigh": AnthropicConfig._supports_effort_level(model, "xhigh", provider),  # pyright: ignore[reportPrivateUsage]  # use the resolved model map
            "max": AnthropicConfig._supports_effort_level(model, "max", provider),  # pyright: ignore[reportPrivateUsage]  # use the resolved model map
            "speed": AnthropicConfig._model_supports_speed_param(model, provider),  # pyright: ignore[reportPrivateUsage]  # use the resolved model map
            "auto_summary": is_reasoning_auto_summary_enabled(),
            "prompt_cache_supported": supports_anthropic_cache_control(model, provider),
            "prompt_cache_enabled": litellm.enable_anthropic_prompt_caching is True,
            "prompt_cache_ttl": litellm.anthropic_prompt_caching_ttl,
        }
    )


async def run_pre_request_hooks(
    arguments: Mapping[str, object], request: LiteLLMMessagesRequest
) -> tuple[bool, object]:
    registered: Final = cast(list[object], litellm.callbacks)  # cast-ok: callback registry has legacy loose types
    callbacks: Final = tuple(registered)
    if not any(
        isinstance(callback, CustomLogger)
        and type(callback).async_pre_request_hook is not CustomLogger.async_pre_request_hook  # pyright: ignore[reportUnknownMemberType]  # legacy callback signature
        for callback in callbacks
    ):
        return False, arguments

    source_messages: Final = _MESSAGES_ADAPTER.validate_python(request.messages)
    raw_messages: Final[object] = flatten_unencrypted_web_search_results_in_anthropic_messages(
        sanitize_tool_use_ids_in_anthropic_messages(
            strip_empty_content_blocks_from_anthropic_messages(source_messages)  # pyright: ignore[reportArgumentType]  # sanitizer only reads the input sequence
        )
    )
    messages: Final = _MESSAGES_ADAPTER.validate_python(raw_messages)
    named: Final = frozenset(
        (
            "model",
            "messages",
            "max_tokens",
            "tools",
            "stream",
            "custom_llm_provider",
            "tool_choice",
            "system",
            "metadata",
            "stop_sequences",
            "temperature",
            "thinking",
            "top_k",
            "top_p",
            "container",
            "api_key",
            "api_base",
            "client",
        )
    )
    extras: Final = {  # mutable-ok: legacy cache injection and hooks edit owned keyword arguments
        name: value for name, value in arguments.items() if name not in named
    }  # mutable-ok: legacy hook edits owned kwargs
    tools: Final = _OPTIONAL_MESSAGES_ADAPTER.validate_python(arguments.get("tools"))
    system: Final = _SYSTEM_ADAPTER.validate_python(arguments.get("system"))
    injector: Final = cast(  # cast-ok: legacy injector accepts validated Messages fields
        _CacheInjector, AnthropicCacheControlHook.maybe_inject_cache_control
    )
    cleaned_messages, cleaned_system = injector(
        messages=messages,
        system=system,
        kwargs=extras,
        model=request.model,
        custom_llm_provider=request.custom_llm_provider,
        tools=tools,
        api_base=request.api_base,
    )
    runner: Final = cast(  # cast-ok: legacy hook has untyped arguments
        _HookRunner,
        main._execute_pre_request_hooks,  # pyright: ignore[reportPrivateUsage]  # existing private hook is the callback contract
    )
    modified: Final = _MAPPING_ADAPTER.validate_python(
        await runner(
            model=request.model,
            messages=cleaned_messages,
            tools=tools,
            stream=request.stream,
            custom_llm_provider=request.custom_llm_provider,
            tool_choice=arguments.get("tool_choice"),
            **extras,
        )
    )
    filtered: Final = MappingProxyType({name: value for name, value in modified.items() if name != "litellm_params"})
    prepared: Final = MappingProxyType(
        {
            **arguments,
            "messages": cleaned_messages,
            "system": cleaned_system,
            **filtered,
        }
    )
    parameters: Final = _MAPPING_ADAPTER.validate_python(modified.get("litellm_params") or MappingProxyType({}))
    selected: Final = modified.get("custom_llm_provider") or parameters.get("custom_llm_provider")
    if isinstance(selected, str) and selected != "anthropic":
        redirected: Final = MappingProxyType(
            {
                "model": request.model,
                "max_tokens": request.max_tokens,
                "api_key": request.api_key,
                "api_base": request.api_base,
                **prepared,
                "custom_llm_provider": selected,
                "is_async": True,
            }
        )
        handler: Final = cast(  # cast-ok: legacy direct handler accepts validated dynamic options
            _DirectHandler, main.anthropic_messages_handler
        )
        result: Final = handler(**redirected)
        return True, await result if inspect.isawaitable(result) else result
    return False, dict(prepared)  # mutable-ok: route driver requires a PyDict keyword view


def map_failure(error: Exception, request: LiteLLMMessagesRequest, request_provider: str) -> Exception:
    from litellm.rust_bridge._native import RustRequestError

    if isinstance(error, RustRequestError):
        return litellm.BadRequestError(message=str(error), model=request.model, llm_provider=request_provider)
    return failures.map_native_failure(error, request.model, request_provider, arguments(request), request.api_base)
