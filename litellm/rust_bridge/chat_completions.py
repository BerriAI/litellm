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
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Final, Protocol

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.bedrock.request_metadata import get_bedrock_request_metadata_fields
from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.protocols import (
    RustAchatCompletions,
    RustChatCompletions,
    RustChatCompletionsDecline,
)
from litellm.rust_bridge.request import (
    NativeChatCompletionsRequest,
    NativeRequestContext,
    NativeRequestOptions,
    PreparedNativeCall,
    call_native,
    provider_connection_params,
    provider_request_params,
)
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointBinding,
    EndpointDispatch,
    async_none,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

_LITELLM_METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])

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
_CHAT_PREFLIGHT: Final[EndpointBinding[RustChatCompletionsDecline]] = EndpointBinding.native(
    route="chat_completions",
    select=lambda native: native.chat_completions_decline,
    enabled=rust_enabled,
)


def set_rust_chat_completions(
    *,
    chat_completions: RustChatCompletions | None | Unchanged = UNCHANGED,
    achat_completions: RustAchatCompletions | None | Unchanged = UNCHANGED,
    decline: RustChatCompletionsDecline | None | Unchanged = UNCHANGED,
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
    if not isinstance(decline, Unchanged):
        if decline is None:
            _CHAT_PREFLIGHT.reset()
        else:
            _CHAT_PREFLIGHT.override(decline)


def _preflight_context(litellm_params: Mapping[str, object] | None) -> NativeRequestContext:
    metadata: Final = litellm_params.get("metadata") if litellm_params is not None else None
    try:
        entries: Final = _LITELLM_METADATA_ADAPTER.validate_python(metadata)
    except ValidationError:
        return NativeRequestContext(request_metadata_fields=get_bedrock_request_metadata_fields())
    return NativeRequestContext(
        metadata=entries,
        request_metadata_fields=get_bedrock_request_metadata_fields(),
    )


def rust_chat_completions_accepts(
    *,
    model: str,
    messages: Sequence[object],
    optional_params: Mapping[str, object],
    custom_llm_provider: str | None,
    litellm_params: Mapping[str, object] | None,
    stream: object,
) -> bool:
    """Whether the Rust path will serve this request.

    Asked before the caller commits to either path, so pre-call logging is
    emitted exactly once, on whichever path actually runs. The core's own
    capability gate answers the second half; it resolves no credentials and
    performs no I/O.
    """
    return _CHAT_PREFLIGHT.accepts(
        check=lambda decline: decline(
            model=model,
            messages=messages,
            optional_params=optional_params,
            custom_llm_provider=custom_llm_provider,
            context=_preflight_context(litellm_params),
            stream=bool(stream),
        ),
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
) -> ModelResponse | None:
    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    return _CHAT.invoke(
        prepare=lambda: PreparedNativeCall(
            NativeChatCompletionsRequest(
                model=model,
                messages=messages,
                optional_params=provider_request_params(optional_params),
                options=NativeRequestOptions(
                    api_key=api_key,
                    api_base=api_base,
                    custom_llm_provider=custom_llm_provider,
                    extra_headers=extra_headers,
                    timeout_seconds=timeout_to_seconds(timeout),
                    provider_connection=provider_connection_params(optional_params),
                ),
            ),
            context=NativeRequestContext(),
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
) -> ModelResponse | None:
    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    return await _CHAT.ainvoke(
        prepare=lambda: PreparedNativeCall(
            NativeChatCompletionsRequest(
                model=model,
                messages=messages,
                optional_params=provider_request_params(optional_params),
                options=NativeRequestOptions(
                    api_key=api_key,
                    api_base=api_base,
                    custom_llm_provider=custom_llm_provider,
                    extra_headers=extra_headers,
                    timeout_seconds=timeout_to_seconds(timeout),
                    provider_connection=provider_connection_params(optional_params),
                ),
            ),
            context=NativeRequestContext(),
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
                optional_params=provider_request_params(optional_params),
                options=NativeRequestOptions(
                    api_key=api_key,
                    api_base=api_base,
                    custom_llm_provider=custom_llm_provider,
                    extra_headers=extra_headers,
                    timeout_seconds=timeout_to_seconds(timeout),
                    provider_connection=provider_connection_params(optional_params),
                ),
            ),
            context=NativeRequestContext(),
        ),
        call=call_native,
        fallback=python_fallback,
        adapt=adapt,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )
