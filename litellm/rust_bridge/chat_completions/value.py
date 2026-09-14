from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from types import MappingProxyType
from typing import Final, cast  # noqa: TID251  # native callables are validated at load time

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.llms.bedrock.request_metadata import bedrock_request_metadata_is_owned
from litellm.rust_bridge.bindings import BINDING_UNSET, BindingUnset
from litellm.rust_bridge.chat_completions.definition import COMPONENT
from litellm.rust_bridge.chat_completions.types import ResponseObserver, RustAchatCompletions, RustChatCompletions
from litellm.rust_bridge.configuration import CapabilityContext, DeliveryMode
from litellm.rust_bridge.runtime import BridgeErrorContext, ainvoke, invoke
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.utils import ModelResponse

_LITELLM_METADATA_ADAPTER: Final = TypeAdapter(Mapping[str, object])
RUST_RESPONSE_HEADER: Final = "x-litellm-rust"


def _as_chat(value: object) -> RustChatCompletions | None:
    return cast(RustChatCompletions, value) if callable(value) else None


def _as_achat(value: object) -> RustAchatCompletions | None:
    return cast(RustAchatCompletions, value) if callable(value) else None


_CHAT: Final = COMPONENT.bind("chat_completions", validate=_as_chat)
_ACHAT: Final = COMPONENT.bind("achat_completions", validate=_as_achat)


def set_rust_chat_completions(
    *,
    chat_completions: RustChatCompletions | None | BindingUnset = BINDING_UNSET,
    achat_completions: RustAchatCompletions | None | BindingUnset = BINDING_UNSET,
) -> None:
    _CHAT.configure(chat_completions)
    _ACHAT.configure(achat_completions)


def load_rust_chat_completions() -> RustChatCompletions | None:
    return COMPONENT.resolve().select(_CHAT)


def load_rust_achat_completions() -> RustAchatCompletions | None:
    return COMPONENT.resolve().select(_ACHAT)


def _anthropic_user_id_reaches_the_body(litellm_params: Mapping[str, object] | None) -> bool:
    metadata: Final = litellm_params.get("metadata") if litellm_params is not None else None
    try:
        entries: Final = _LITELLM_METADATA_ADAPTER.validate_python(metadata)
    except ValidationError:
        return False
    return entries.get("user_id") is not None


def _host_facts(stream: object, litellm_params: Mapping[str, object] | None) -> Mapping[str, bool]:
    return MappingProxyType(
        {
            "stream": bool(stream),
            "anthropic_user_id": _anthropic_user_id_reaches_the_body(litellm_params),
            "bedrock_metadata_owned": bedrock_request_metadata_is_owned(),
        }
    )


def _build_model_response(rust_response: Mapping[str, object], model_response: ModelResponse) -> ModelResponse:
    built: Final = convert_to_model_response_object(
        response_object=dict(rust_response),  # mutable-ok: the converter takes a real dict and rewrites it
        model_response_object=model_response,
        hidden_params={"additional_headers": {RUST_RESPONSE_HEADER: "true"}},  # mutable-ok: converter rewrites it
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
    python_fallback: Callable[[], object],
    stream: object = False,
    litellm_params: Mapping[str, object] | None = None,
    on_request: Callable[[], None] = lambda: None,
    on_response: ResponseObserver = lambda _response: None,
) -> object:
    execution: Final = COMPONENT.resolve(
        CapabilityContext(
            provider=custom_llm_provider or "",
            model=model,
            delivery=DeliveryMode.STREAMING if bool(stream) else DeliveryMode.COMPLETED,
        )
    )
    rust_chat_completions: Final = execution.select(_CHAT)

    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    native_call: Final[Callable[[], Mapping[str, object]] | None] = (
        lambda: rust_chat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            host_facts=_host_facts(stream, litellm_params),
            on_request=on_request,
        )
        if rust_chat_completions is not None
        else None
    )
    return invoke(
        execution=execution,
        native_call=native_call,
        python_fallback=python_fallback,
        adapt=adapt,
        context=BridgeErrorContext(route=COMPONENT.name.value, provider=custom_llm_provider or "", model=model),
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
    python_fallback: Callable[[], Awaitable[object]],
    stream: object = False,
    litellm_params: Mapping[str, object] | None = None,
    on_request: Callable[[], None] = lambda: None,
    on_response: ResponseObserver = lambda _response: None,
) -> object:
    execution: Final = COMPONENT.resolve(
        CapabilityContext(
            provider=custom_llm_provider or "",
            model=model,
            delivery=DeliveryMode.STREAMING if bool(stream) else DeliveryMode.COMPLETED,
        )
    )
    rust_achat_completions: Final = execution.select(_ACHAT)

    def adapt(rust_response: Mapping[str, object]) -> ModelResponse:
        on_response(rust_response)
        return _build_model_response(rust_response, model_response)

    native_call: Final[Callable[[], Awaitable[Mapping[str, object]]] | None] = (
        lambda: rust_achat_completions(
            model=model,
            messages=messages,
            optional_params=optional_params,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            host_facts=_host_facts(stream, litellm_params),
            on_request=on_request,
        )
        if rust_achat_completions is not None
        else None
    )
    return await ainvoke(
        execution=execution,
        native_call=native_call,
        python_fallback=python_fallback,
        adapt=adapt,
        context=BridgeErrorContext(route=COMPONENT.name.value, provider=custom_llm_provider or "", model=model),
    )
