"""Python runtime boundary for native chat completions."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # runtime typing constructs

import httpx

from ..litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,  # pyright: ignore[reportUnknownVariableType]  # legacy converter is untyped
)
from ..types.utils import ModelResponse
from .bindings import UNCHANGED, Unchanged
from .callbacks import OneShotCallbackHandle
from .configuration import rust_enabled
from .runtime import BridgeErrorContext, EndpointDispatch
from .timeouts import timeout_to_seconds

RUST_RESPONSE_HEADER: Final = "x-litellm-rust"


@dataclass(frozen=True, slots=True)
class NativeChatContext:
    metadata: Mapping[str, object] | None
    litellm_metadata: Mapping[str, object] | None
    request_metadata_fields: tuple[str, ...]


class RustChatCompletions(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        request_context: NativeChatContext,
        callback_adapter: OneShotCallbackHandle,
    ) -> Mapping[str, object]: ...


class RustAchatCompletions(Protocol):
    def __call__(
        self,
        *,
        model: str,
        messages: Sequence[object],
        optional_params: Mapping[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: Mapping[str, object] | None,
        timeout_seconds: float | None,
        request_context: NativeChatContext,
        callback_adapter: OneShotCallbackHandle,
    ) -> Awaitable[Mapping[str, object]]: ...


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
    sync: RustChatCompletions | None | Unchanged = UNCHANGED,
    asynchronous: RustAchatCompletions | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(sync, Unchanged):
        if sync is None:
            _CHAT.sync.reset()
        else:
            _CHAT.sync.override(sync)
    if not isinstance(asynchronous, Unchanged):
        if asynchronous is None:
            _CHAT.asynchronous.reset()
        else:
            _CHAT.asynchronous.override(asynchronous)


def _build_model_response(
    native_response: Mapping[str, object],
    model_response: ModelResponse,
) -> ModelResponse:
    result: Final = convert_to_model_response_object(
        response_object=dict(native_response),  # mutable-ok: the converter takes a real dict and rewrites it
        model_response_object=model_response,
        hidden_params={  # mutable-ok: rewritten by the converter
            "additional_headers": {RUST_RESPONSE_HEADER: "true"}
        },
    )
    if not isinstance(result, ModelResponse):
        raise TypeError(f"native chat completions returned an invalid response type: {type(result).__name__}")
    return result


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
    request_context: NativeChatContext,
    request_override: bool | None,
    has_custom_client: bool,
    callback_adapter: OneShotCallbackHandle,
    python_fallback: Callable[[], object],
) -> object:
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
            request_context=request_context,
            callback_adapter=callback_adapter,
        ),
        fallback=python_fallback,
        adapt=lambda response: _build_model_response(response, model_response),
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=request_override,
        eligible=not has_custom_client,
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
    request_context: NativeChatContext,
    request_override: bool | None,
    has_custom_client: bool,
    callback_adapter: OneShotCallbackHandle,
    python_fallback: Callable[[], Awaitable[object]],
) -> object:
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
            request_context=request_context,
            callback_adapter=callback_adapter,
        ),
        fallback=python_fallback,
        adapt=lambda response: _build_model_response(response, model_response),
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=request_override,
        eligible=not has_custom_client,
    )
