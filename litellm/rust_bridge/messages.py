"""Thin Python wrapper for the native Rust Anthropic Messages bridge."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from functools import reduce
from types import MappingProxyType
from typing import Final, Literal

import httpx
from pydantic import BaseModel, TypeAdapter
from typing_extensions import ReadOnly

from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta
from litellm.litellm_core_utils.dot_notation_indexing import delete_nested_value
from litellm.litellm_core_utils.get_provider_specific_headers import ProviderSpecificHeaderUtils
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.protocols import RustAmessages, RustMessages
from litellm.rust_bridge.request import (
    NativeMessagesRequest,
    NativePreCallDetails,
    NativeRequestCapabilities,
    NativeRequestContext,
    NativeRequestOptions,
    PreparedNativeCall,
    call_native,
    request_context,
    with_capabilities,
)
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointDispatch,
    async_none,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.llms.anthropic_messages.anthropic_response import AnthropicMessagesResponse
from litellm.types.llms.openai import ChatCompletionUserMessage
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders, ProviderSpecificHeader
from litellm.utils import ProviderConfigManager

_MESSAGES: Final[EndpointDispatch[RustMessages, RustAmessages]] = EndpointDispatch.native(
    route="messages",
    sync=lambda native: native.messages,
    asynchronous=lambda native: native.amessages,
    enabled=rust_enabled,
)


def load_rust_messages() -> RustMessages | None:
    return _MESSAGES.sync.load()


def load_rust_amessages() -> RustAmessages | None:
    return _MESSAGES.asynchronous.load()


def messages(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: float | httpx.Timeout | None,
    stream: bool = False,
    has_custom_client: bool = False,
    has_agentic_hook: bool = False,
    context: NativeRequestContext | None = None,
) -> dict[str, object] | None:
    return _MESSAGES.invoke(
        prepare=lambda: PreparedNativeCall(
            NativeMessagesRequest(
                model=model,
                body=body,
            ),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
            ),
            context=with_capabilities(
                context or NativeRequestContext(),
                NativeRequestCapabilities(
                    execution_mode="sync",
                    stream=stream,
                    has_custom_client=has_custom_client,
                    has_agentic_hook=has_agentic_hook,
                ),
            ),
        ),
        call=call_native,
        fallback=lambda: None,
        adapt=identity,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )


async def amessages(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: float | httpx.Timeout | None,
    stream: bool = False,
    has_custom_client: bool = False,
    has_agentic_hook: bool = False,
    context: NativeRequestContext | None = None,
) -> dict[str, object] | None:
    return await _MESSAGES.ainvoke(
        prepare=lambda: PreparedNativeCall(
            NativeMessagesRequest(
                model=model,
                body=body,
            ),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
            ),
            context=with_capabilities(
                context or NativeRequestContext(),
                NativeRequestCapabilities(
                    execution_mode="async",
                    stream=stream,
                    has_custom_client=has_custom_client,
                    has_agentic_hook=has_agentic_hook,
                ),
            ),
        ),
        call=call_native,
        fallback=async_none,
        adapt=identity,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )


MessagesResponse = AnthropicMessagesResponse | Iterator[bytes] | AsyncIterator[object]
MessagesResult = MessagesResponse | Coroutine[object, None, MessagesResponse]
_BODY_ADAPTER: Final = TypeAdapter(dict[str, object])


class _NativeMessagesResponse(BaseModel):
    id: str
    type: Literal["message"]
    role: Literal["assistant"]
    model: str
    content: list[dict[str, object]]
    usage: dict[str, object]


class _BridgedMessagesResponse(AnthropicMessagesResponse):
    _hidden_params: ReadOnly[dict[str, dict[str, str]]]


_RESPONSE_ADAPTER: Final = TypeAdapter(AnthropicMessagesResponse)


@dataclass
class _MessagesOperation:
    model: str
    provider: str
    messages: list[dict[str, object]]
    body: Callable[[], dict[str, object]]
    params: GenericLiteLLMParams
    logging: Logging | None
    api_key: str | None
    api_base: str | None
    python: Callable[[], MessagesResult]
    stream: bool = False
    asynchronous: bool = False
    has_custom_client: bool = False
    logged: bool = False

    def prepare(self) -> PreparedNativeCall[NativeMessagesRequest]:
        requested: Final = self.body()
        body: Final = _BODY_ADAPTER.validate_python(
            reduce(
                delete_nested_value,
                TypeAdapter(tuple[str, ...]).validate_python(self.params.get("additional_drop_params") or ()),
                requested,
            )
        )
        provider_headers: Final = ProviderSpecificHeaderUtils.get_provider_specific_headers(
            TypeAdapter(ProviderSpecificHeader | Sequence[ProviderSpecificHeader] | None).validate_python(
                self.params.get("provider_specific_header")
            ),
            self.provider,
        )
        config: Final = ProviderConfigManager.get_provider_anthropic_messages_config(
            model=self.model, provider=LlmProviders(self.provider)
        )
        initial_headers: Final = _BODY_ADAPTER.validate_python(
            MappingProxyType(
                {
                    **(self.params.get("headers") or MappingProxyType({})),
                    **(self.params.get("extra_headers") or MappingProxyType({})),
                    **provider_headers,
                }
            )
        )
        validated_headers, base = (
            config.validate_anthropic_messages_environment(
                headers=initial_headers,
                model=self.model,
                messages=self.messages,
                optional_params=body,
                litellm_params=self.params.model_dump(),
                api_key=self.api_key,
                api_base=self.api_base,
            )
            if config is not None
            else (initial_headers, self.api_base)
        )
        headers: Final = (
            update_headers_with_filtered_beta(headers=validated_headers, provider=self.provider)
            if config is not None and config.should_filter_anthropic_beta_headers()
            else validated_headers
        )
        request_body: Final = _BODY_ADAPTER.validate_python(
            MappingProxyType({**body, "model": self.model, "messages": self.messages})
        )
        if self.logging is not None:
            self.logging.update_from_kwargs(
                kwargs=self.params.model_dump(),
                model=self.model,
                optional_params=body,
                litellm_params=self.params.model_dump(),
                custom_llm_provider=self.provider,
            )
            self.logging.model_call_details.update(request_body)
            log_details: Final[NativePreCallDetails] = {
                "complete_input_dict": request_body,
                "api_base": base or "",
                "headers": headers,
            }
            log_input: Final[ChatCompletionUserMessage] = {"role": "user", "content": json.dumps(request_body)}
            self.logging.pre_call(
                input=[log_input],  # mutable-ok: logging callbacks expect a concrete message list
                api_key=self.api_key,
                additional_args=log_details,
            )
            self.logged = True
        return PreparedNativeCall(
            NativeMessagesRequest(
                model=self.model,
                body=request_body,
            ),
            options=NativeRequestOptions(
                api_key=self.api_key,
                api_base=base,
                custom_llm_provider=self.provider,
                extra_headers=headers,
                timeout_seconds=timeout_to_seconds(
                    BaseLLMHTTPHandler.resolve_anthropic_messages_timeout(self.params, False, self.provider)
                ),
            ),
            context=request_context(
                logging_obj=self.logging,
                request_model=self.logging.model if self.logging is not None else self.model,
                litellm_params=self.params.model_dump(),
                capabilities=NativeRequestCapabilities(
                    execution_mode="async" if self.asynchronous else "sync",
                    stream=self.stream,
                    has_custom_client=self.has_custom_client,
                    has_agentic_hook=BaseLLMHTTPHandler.has_agentic_completion_hook(self.logging),
                ),
            ),
        )

    def fallback(self) -> MessagesResult:
        with self.logging.suppress_next_pre_call() if self.logged and self.logging is not None else nullcontext():
            return self.python()

    async def afallback(self) -> MessagesResponse:
        with self.logging.suppress_next_pre_call() if self.logged and self.logging is not None else nullcontext():
            response: Final = self.python()
            return await response if isinstance(response, Coroutine) else response

    def adapt(self, response: dict[str, object]) -> AnthropicMessagesResponse:
        _NativeMessagesResponse.model_validate(response)
        parsed: Final[_BridgedMessagesResponse] = {
            **_RESPONSE_ADAPTER.validate_python(response),
            "_hidden_params": {"additional_headers": {"x-litellm-rust": "true"}},
        }
        if self.logging is not None:
            self.logging.post_call(input=self.messages, api_key=self.api_key, original_response=json.dumps(response))
        return parsed


def dispatch_messages(
    *,
    model: str,
    provider: str,
    messages: list[dict[str, object]],
    body: Callable[[], dict[str, object]],
    params: GenericLiteLLMParams,
    logging: Logging | None,
    api_key: str | None,
    api_base: str | None,
    stream: bool,
    asynchronous: bool,
    has_custom_client: bool,
    fallback: Callable[[], MessagesResult],
) -> MessagesResult:
    operation: Final = _MessagesOperation(
        model,
        provider,
        messages,
        body,
        params,
        logging,
        api_key,
        api_base,
        fallback,
        stream,
        asynchronous,
        has_custom_client,
    )

    error_context: Final = BridgeErrorContext(provider=provider, model=model)
    if asynchronous:
        return _MESSAGES.ainvoke(
            prepare=operation.prepare,
            call=call_native,
            adapt=operation.adapt,
            fallback=operation.afallback,
            error_context=error_context,
        )
    return _MESSAGES.invoke(
        prepare=operation.prepare,
        call=call_native,
        adapt=operation.adapt,
        fallback=operation.fallback,
        error_context=error_context,
    )
