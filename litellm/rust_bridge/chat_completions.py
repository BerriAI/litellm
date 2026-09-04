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
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # runtime typing constructs

import httpx
from pydantic import TypeAdapter, ValidationError

from .._logging import verbose_logger
from ..litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from ..litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,  # pyright: ignore[reportUnknownVariableType]  # legacy converter is untyped
)
from ..llms.bedrock.request_metadata import bedrock_request_metadata_is_owned
from ..types.utils import ModelResponse
from .bindings import UNCHANGED, Unchanged
from .callbacks import OneShotCallbackHandle
from .configuration import rust_enabled
from .runtime import (
    BridgeErrorContext,
    EndpointDispatch,
    async_none,
)
from .timeouts import timeout_to_seconds

# Providers whose `/chat/completions` deployments the Rust core can serve. A
# provider outside this set never reaches the bridge.
RUST_CHAT_COMPLETIONS_PROVIDERS: Final = frozenset({"anthropic", "bedrock"})

# `litellm_params` values are `object`, so validate the one this module reads
# rather than narrowing an unparameterized `Mapping` and typing the result Any.
_LITELLM_METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])

RUST_RESPONSE_HEADER: Final = "x-litellm-rust"


class RustChatCompletions(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        callback_adapter: OneShotCallbackHandle | None,
    ) -> Mapping[str, object]:
        raise NotImplementedError


class RustAchatCompletions(Protocol):
    def __call__(
        self,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object] | None,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        callback_adapter: OneShotCallbackHandle | None,
    ) -> Awaitable[Mapping[str, object]]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ChatCompletionsCallbackHandle:
    logging_obj: LiteLLMLoggingObj
    messages: Sequence[object]
    api_key: str

    def pre_call(self, payload: object, /) -> None:
        event: Final = _LITELLM_METADATA_ADAPTER.validate_python(payload)
        request: Final = event.get("request")
        headers: Final = event.get("headers")
        api_base: Final = event.get("api_base", event.get("url", ""))
        self.logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # legacy logger is untyped
            input=self.messages,
            api_key=self.api_key,
            additional_args={  # mutable-ok: legacy logger accepts a mutable payload
                "complete_input_dict": request if isinstance(request, Mapping) else event,
                "api_base": api_base if isinstance(api_base, str) else str(api_base),
                "headers": headers if isinstance(headers, Mapping) else {},  # mutable-ok: empty logging headers
            },
        )

    def post_call(self, payload: object, /) -> None:
        event: Final = _LITELLM_METADATA_ADAPTER.validate_python(payload)
        response: Final = event.get("response", event)
        self.logging_obj.post_call(  # pyright: ignore[reportUnknownMemberType]  # legacy logger is untyped
            input=self.messages,
            api_key=self.api_key,
            original_response=response if isinstance(response, str) else json.dumps(response),
        )

    def error(self, payload: object, /) -> None:
        event: Final = _LITELLM_METADATA_ADAPTER.validate_python(payload)
        self.logging_obj.model_call_details["provider_error"] = dict(  # mutable-ok: logger stores mutable details
            event
        )


_CHAT: Final = cast(  # cast-ok: generic classmethod loses the route Protocol parameters
    EndpointDispatch[RustChatCompletions, RustAchatCompletions],
    EndpointDispatch.native(
        route="chat_completions",
        sync="chat_completions",
        asynchronous="achat_completions",
        enabled=rust_enabled,
    ),
)  # cast-ok: generic classmethod cannot preserve the route Protocol parameters


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


def _anthropic_user_id_reaches_the_body(litellm_params: Mapping[str, object] | None) -> bool:
    metadata: Final = litellm_params.get("metadata") if litellm_params is not None else None
    try:
        entries: Final = _LITELLM_METADATA_ADAPTER.validate_python(metadata)
    except ValidationError:
        return False
    return entries.get("user_id") is not None


def _litellm_metadata_reaches_the_provider(
    custom_llm_provider: str | None, litellm_params: Mapping[str, object] | None
) -> bool:
    """Whether the Python transform would promote proxy-owned attribution into the
    provider request, below this gate and inside the function the Rust route replaces.

    `AnthropicConfig.transform_request` promotes a valid `metadata["user_id"]`
    into the Messages body, so the core never sees the key and would send the
    request to Anthropic with the abuse-detection attribution missing.

    `AmazonConverseConfig` resolves proxy-owned `requestMetadata` onto the
    Converse body whenever the operator armed `bedrock_request_metadata_fields`.
    Owning that field also means evicting a caller-supplied one, which the core
    cannot do either, so ownership alone is the condition rather than whether
    anything resolved.

    Deliberately a superset of Python's condition in both cases: declining a
    request Python would not have attributed anyway costs only the Rust path,
    while missing one loses the attribution silently.
    """
    match custom_llm_provider:
        case "anthropic":
            return _anthropic_user_id_reaches_the_body(litellm_params)
        case "bedrock":
            return bedrock_request_metadata_is_owned()
        case _:
            return False


def rust_request_override(litellm_params: Mapping[str, object] | None) -> bool | None:
    raw_request_override: Final = litellm_params.get("rust") if litellm_params is not None else None
    return raw_request_override if isinstance(raw_request_override, bool) else None


def rust_chat_completions_accepts(
    *,
    custom_llm_provider: str | None,
    litellm_params: Mapping[str, object] | None,
    stream: object,
    asynchronous: bool = False,
) -> bool:
    """Whether a ready native binding can attempt this request."""
    if custom_llm_provider not in RUST_CHAT_COMPLETIONS_PROVIDERS:
        return False
    if stream:
        return False
    if _litellm_metadata_reaches_the_provider(custom_llm_provider, litellm_params):
        verbose_logger.debug("Rust chat completions declined (litellm metadata user_id); using the Python path")
        return False
    binding: Final = _CHAT.asynchronous if asynchronous else _CHAT.sync
    return binding.can_attempt(
        request_override=rust_request_override(litellm_params),
    )


def _request_is_eligible(
    *,
    custom_llm_provider: str | None,
    litellm_params: Mapping[str, object] | None,
    stream: object,
    has_custom_client: bool,
) -> bool:
    return (
        custom_llm_provider in RUST_CHAT_COMPLETIONS_PROVIDERS
        and not stream
        and not has_custom_client
        and not _litellm_metadata_reaches_the_provider(custom_llm_provider, litellm_params)
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
    request_override: bool | None,
    callback_adapter: OneShotCallbackHandle,
) -> ModelResponse | None:
    return _CHAT.invoke(
        call=lambda rust_chat_completions: rust_chat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        fallback=lambda: None,
        adapt=lambda rust_response: _build_model_response(rust_response, model_response),
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=request_override,
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
    request_override: bool | None,
    callback_adapter: OneShotCallbackHandle,
) -> ModelResponse | None:
    return await _CHAT.ainvoke(
        call=lambda rust_achat_completions: rust_achat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        fallback=async_none,
        adapt=lambda rust_response: _build_model_response(rust_response, model_response),
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=request_override,
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
    python_fallback: Callable[[], Awaitable[object]],
    request_override: bool | None,
    callback_adapter: OneShotCallbackHandle,
) -> object:
    """Await the Rust path, falling back to the caller's own Python path when
    the bridge is unavailable or the call fails.

    The caller supplies the fallback, so the bridge stays free of provider
    dispatch. This exists because a caller that dispatches asynchronously has
    already returned a coroutine by the time a Rust failure surfaces, and so
    cannot fall back on its own.
    """

    return await _CHAT.ainvoke(
        call=lambda rust_achat_completions: rust_achat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        fallback=python_fallback,
        adapt=lambda rust_response: _build_model_response(rust_response, model_response),
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=request_override,
    )


def dispatch_chat_completions(
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
    litellm_params: Mapping[str, object] | None,
    has_custom_client: bool,
    callback_adapter: OneShotCallbackHandle,
    python_fallback: Callable[[], object],
) -> object:
    """Select one complete sync provider implementation at the SDK boundary."""
    return _CHAT.invoke(
        call=lambda native: native(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        fallback=python_fallback,
        adapt=lambda response: _build_model_response(response, model_response),
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=rust_request_override(litellm_params),
        eligible=_request_is_eligible(
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            stream=optional_params.get("stream", False),
            has_custom_client=has_custom_client,
        ),
    )


async def adispatch_chat_completions(
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
    litellm_params: Mapping[str, object] | None,
    has_custom_client: bool,
    callback_adapter: OneShotCallbackHandle,
    python_fallback: Callable[[], Awaitable[object]],
) -> object:
    """Select one complete async provider implementation at the SDK boundary."""
    return await _CHAT.ainvoke(
        call=lambda native: native(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        fallback=python_fallback,
        adapt=lambda response: _build_model_response(response, model_response),
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=rust_request_override(litellm_params),
        eligible=_request_is_eligible(
            custom_llm_provider=custom_llm_provider,
            litellm_params=litellm_params,
            stream=optional_params.get("stream", False),
            has_custom_client=has_custom_client,
        ),
    )
