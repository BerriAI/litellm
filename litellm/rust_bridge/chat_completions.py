"""Thin Python wrapper for the native Rust chat completions bridge.

The Rust core owns the conversation translation, the provider call, and the
response normalization for the subset of `/chat/completions` requests it
accepts. This module only marshals inputs and hands the normalized result to
LiteLLM's existing `ModelResponse` builder.

``None`` means the provider was never called, so the caller is free to serve the
request on the Python path. A failure after the call was issued raises instead:
retrying it there would bill the customer for the same work twice.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

import httpx
from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.bedrock.request_metadata import get_bedrock_request_metadata_fields
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.protocols import (
    RustAchatCompletions,
    RustChatCompletions,
)
from litellm.rust_bridge.request import (
    NativeAnthropicOptions,
    NativeBedrockOptions,
    NativeChatCompletionsRequest,
    NativePreCallDetails,
    NativeRequestCapabilities,
    NativeRequestContext,
    NativeRequestOptions,
    PreparedNativeCall,
    anthropic_options,
    bedrock_options,
    call_native,
    request_context,
    with_capabilities,
)
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointDispatch,
    async_none,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.secret_managers.main import get_secret_str
from litellm.types.completion import (
    _CompletionDispatchContext,  # pyright: ignore[reportPrivateUsage]  # shared internal SDK dispatch context
    _CompletionDispatchResult,  # pyright: ignore[reportPrivateUsage]  # shared internal SDK dispatch result
)
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

RUST_RESPONSE_HEADER: Final = "x-litellm-rust"


class ResponseObserver(Protocol):
    """Invoked with the payload the core returned, on success only.

    Lets the caller emit its own `post_call` on whichever path served the
    request. Both entry points call it, so the synchronous and asynchronous
    paths cannot drift apart the way the pre_call suppression once did.
    """

    def __call__(self, rust_response: Mapping[str, object], /) -> None:
        raise NotImplementedError


def response_logger(
    *,
    logging_obj: LiteLLMLoggingObj,
    messages: Sequence[object],
    api_key: str,
    additional_args: Mapping[str, object],
) -> ResponseObserver:
    """A `ResponseObserver` that emits the caller's `post_call` for a Rust-served
    request.

    The core owns the provider call, so the Python transform that normally
    raises this event never runs; without it every `post_call` callback goes
    silent on a Rust-served request and `original_response` stays unset. The
    payload is the core's normalized response rather than the provider's wire
    body, which is the closest thing that crosses the bridge.
    """

    def log(rust_response: Mapping[str, object], /) -> None:
        logging_obj.post_call(
            input=messages,
            api_key=api_key,
            original_response=json.dumps(rust_response),
            additional_args=additional_args,
        )

    return log


_CHAT: Final[EndpointDispatch[RustChatCompletions, RustAchatCompletions]] = EndpointDispatch.native(
    route="chat_completions",
    sync=lambda native: native.chat_completions,
    asynchronous=lambda native: native.achat_completions,
    enabled=rust_enabled,
)


def set_rust_chat_completions(
    *,
    chat_completions: RustChatCompletions | None | Unchanged = UNCHANGED,
    achat_completions: RustAchatCompletions | None | Unchanged = UNCHANGED,
) -> None:
    """Inject the native callables, so tests can supply a double instead of
    patching module attributes."""
    if not isinstance(chat_completions, Unchanged):
        if chat_completions is None:
            _CHAT.sync.reset()
        else:
            _CHAT.sync.override(chat_completions)
    if not isinstance(achat_completions, Unchanged):
        if achat_completions is None:
            _CHAT.asynchronous.reset()
        else:
            _CHAT.asynchronous.override(achat_completions)


def _provider_eligibility_options(
    provider: str | None,
    litellm_params: Mapping[str, object] | None,
    optional_params: Mapping[str, object],
) -> NativeRequestOptions:
    bedrock: Final = (
        replace(
            bedrock_options(optional_params),
            request_metadata_fields=get_bedrock_request_metadata_fields(),
        )
        if provider == "bedrock"
        else None
    )
    anthropic: Final = anthropic_options(litellm_params) if provider == "anthropic" else None
    return NativeRequestOptions(custom_llm_provider=provider, bedrock=bedrock, anthropic=anthropic)


def _eligibility_context(
    *,
    execution_mode: str | None = None,
    stream: bool,
    has_custom_client: bool = False,
    has_agentic_hook: bool = False,
) -> NativeRequestContext:
    return NativeRequestContext(
        capabilities=NativeRequestCapabilities(
            execution_mode=execution_mode,
            stream=stream,
            has_custom_client=has_custom_client,
            has_agentic_hook=has_agentic_hook,
        )
    )


def _build_model_response(
    rust_response: Mapping[str, object],
    model_response: ModelResponse,
) -> ModelResponse:
    built: Final = convert_to_model_response_object(
        response_object=dict(rust_response),  # mutable-ok: the converter takes a real dict and rewrites it
        model_response_object=model_response,
        hidden_params={"additional_headers": {RUST_RESPONSE_HEADER: "true"}},  # mutable-ok: rewritten by the converter
    )
    if not isinstance(built, ModelResponse):
        raise TypeError(f"expected a ModelResponse from the rust path, got {type(built).__name__}")
    return built


def chat_completions(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    model_response: ModelResponse,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    on_response: ResponseObserver,
    bedrock: NativeBedrockOptions | None = None,
    anthropic: NativeAnthropicOptions | None = None,
    context: NativeRequestContext | None = None,
) -> ModelResponse | None:
    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    return _CHAT.invoke(
        prepare=lambda: PreparedNativeCall(
            NativeChatCompletionsRequest(
                model=model,
                messages=messages,
                optional_params=optional_params,
            ),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
                bedrock=bedrock,
                anthropic=anthropic,
            ),
            context=_execution_context(context, "sync"),
        ),
        call=call_native,
        fallback=lambda: None,
        adapt=adapt,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )


async def achat_completions(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    model_response: ModelResponse,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    on_response: ResponseObserver,
    bedrock: NativeBedrockOptions | None = None,
    anthropic: NativeAnthropicOptions | None = None,
    context: NativeRequestContext | None = None,
) -> ModelResponse | None:
    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    return await _CHAT.ainvoke(
        prepare=lambda: PreparedNativeCall(
            NativeChatCompletionsRequest(
                model=model,
                messages=messages,
                optional_params=optional_params,
            ),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
                bedrock=bedrock,
                anthropic=anthropic,
            ),
            context=_execution_context(context, "async"),
        ),
        call=call_native,
        fallback=async_none,
        adapt=adapt,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )


async def achat_completions_or_fallback(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    model_response: ModelResponse,
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None,
    on_response: ResponseObserver,
    python_fallback: Callable[[], Awaitable[object]],
    bedrock: NativeBedrockOptions | None = None,
    anthropic: NativeAnthropicOptions | None = None,
    context: NativeRequestContext | None = None,
) -> object:
    """Await the Rust path, falling back to the caller's own Python path when
    the bridge is unavailable or the call fails.

    The caller supplies the fallback, so the bridge stays free of provider
    dispatch. This exists because a caller that dispatches asynchronously has
    already returned a coroutine by the time a Rust failure surfaces, and so
    cannot fall back on its own.
    """

    def adapt(rust_response: Mapping[str, object]) -> object:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    return await _CHAT.ainvoke(
        prepare=lambda: PreparedNativeCall(
            NativeChatCompletionsRequest(
                model=model,
                messages=messages,
                optional_params=optional_params,
            ),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
                bedrock=bedrock,
                anthropic=anthropic,
            ),
            context=_execution_context(context, "async"),
        ),
        call=call_native,
        fallback=python_fallback,
        adapt=adapt,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )


_PARAMS_ADAPTER: Final = TypeAdapter(dict[str, object])
_STR_ADAPTER: Final = TypeAdapter(str | None)


@dataclass
class _ChatOperation:
    context: _CompletionDispatchContext
    python: Callable[[], _CompletionDispatchResult]
    pre_call_logged: bool = False

    def prepare(self) -> PreparedNativeCall[NativeChatCompletionsRequest]:
        ctx: Final = self.context
        config: Final = ctx.provider_config
        defaults: Final = (
            _PARAMS_ADAPTER.validate_python(config.get_config_for_model(ctx.model))
            if config is not None
            else MappingProxyType({})
        )
        params: Final = _PARAMS_ADAPTER.validate_python(MappingProxyType({**defaults, **ctx.optional_params}))
        key: Final = (
            ctx.api_key
            or _STR_ADAPTER.validate_python(getattr(litellm, f"{ctx.custom_llm_provider}_key", None))
            or litellm.api_key
            or get_secret_str(f"{ctx.custom_llm_provider.upper()}_API_KEY")
        )
        base: Final = (
            ctx.api_base
            or litellm.api_base
            or get_secret_str(f"{ctx.custom_llm_provider.upper()}_API_BASE")
            or get_secret_str(f"{ctx.custom_llm_provider.upper()}_BASE_URL")
        )
        initial_headers: Final = _PARAMS_ADAPTER.validate_python(
            MappingProxyType({**(ctx.headers or MappingProxyType({})), **(ctx.extra_headers or MappingProxyType({}))})
        )
        headers: Final = (
            _PARAMS_ADAPTER.validate_python(
                config.validate_environment(
                    api_key=key,
                    api_base=base,
                    headers=initial_headers,
                    model=ctx.model,
                    messages=ctx.messages,
                    optional_params=params,
                    litellm_params=ctx.litellm_params,
                )
            )
            if config is not None
            else initial_headers
        )
        log_details: Final[NativePreCallDetails] = {
            "complete_input_dict": {"model": ctx.model, "messages": ctx.messages, **params},
            "api_base": base or "",
            "headers": headers,
        }
        ctx.logging.pre_call(input=ctx.messages, api_key=key, additional_args=log_details)
        self.pre_call_logged = True
        provider_options: Final = _provider_eligibility_options(ctx.custom_llm_provider, ctx.litellm_params, params)
        return PreparedNativeCall(
            NativeChatCompletionsRequest(
                model=ctx.model,
                messages=ctx.messages,
                optional_params=params,
            ),
            options=replace(
                provider_options,
                api_key=key,
                api_base=base,
                extra_headers=headers,
                timeout_seconds=timeout_to_seconds(float(ctx.timeout) if isinstance(ctx.timeout, str) else ctx.timeout),
            ),
            context=request_context(
                logging_obj=ctx.logging,
                request_model=ctx.logging.model,
                litellm_params=ctx.litellm_params,
                capabilities=NativeRequestCapabilities(
                    execution_mode="async" if ctx.acompletion else "sync",
                    stream=bool(ctx.stream),
                    has_custom_client=ctx.client is not None or ctx.shared_session is not None,
                    has_agentic_hook=BaseLLMHTTPHandler.has_agentic_completion_hook(ctx.logging),
                ),
            ),
        )

    def fallback(self) -> _CompletionDispatchResult:
        with self.context.logging.suppress_next_pre_call() if self.pre_call_logged else nullcontext():
            return self.python()

    async def afallback(self) -> ModelResponse | litellm.CustomStreamWrapper:
        with self.context.logging.suppress_next_pre_call() if self.pre_call_logged else nullcontext():
            result: Final = self.python()
            return await result if isinstance(result, Coroutine) else result

    def adapt(self, response: Mapping[str, object]) -> ModelResponse:
        self.context.logging.post_call(
            input=self.context.messages,
            api_key=self.context.api_key,
            original_response=json.dumps(response),
        )
        return _build_model_response(response, self.context.model_response)


def dispatch_completion(
    context: _CompletionDispatchContext,
    fallback: Callable[[], _CompletionDispatchResult],
) -> _CompletionDispatchResult:
    operation: Final = _ChatOperation(context, fallback)
    error_context: Final = BridgeErrorContext(provider=context.custom_llm_provider, model=context.model)
    if context.acompletion:
        return _CHAT.ainvoke(
            prepare=operation.prepare,
            call=call_native,
            fallback=operation.afallback,
            adapt=operation.adapt,
            error_context=error_context,
        )
    return _CHAT.invoke(
        prepare=operation.prepare,
        call=call_native,
        fallback=operation.fallback,
        adapt=operation.adapt,
        error_context=error_context,
    )
