"""Thin Python wrapper for the native Rust Anthropic Messages bridge."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # runtime typing constructs

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.callbacks import CallbackDecision, OneShotCallbackHandle
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointDispatch,
    async_none,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds

_CALLBACK_EVENT: Final = TypeAdapter(Mapping[str, object])


class RustMessages(Protocol):
    def __call__(
        self,
        model: str,
        body: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        timeout_seconds: float | None,
        callback_adapter: OneShotCallbackHandle | None,
    ) -> dict[str, object]:
        raise NotImplementedError


class RustAmessages(Protocol):
    def __call__(
        self,
        model: str,
        body: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        timeout_seconds: float | None,
        callback_adapter: OneShotCallbackHandle | None,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class MessagesCallbackHandle:
    logging_obj: LiteLLMLoggingObj
    messages: Sequence[object]
    api_key: str

    def pre_call(self, payload: object, /) -> CallbackDecision:
        event: Final = _CALLBACK_EVENT.validate_python(payload)
        request: Final = event.get("request", event)
        self.logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # legacy logger is untyped
            input=self.messages,
            api_key=self.api_key,
            additional_args={  # mutable-ok: legacy logger accepts a mutable payload
                "complete_input_dict": request,
                "api_base": event.get("api_base", event.get("url", "")),
                "headers": event.get("headers", {}),  # mutable-ok: empty logging headers
            },
        )
        return {"action": "unchanged"}

    def post_call(self, payload: object, /) -> CallbackDecision:
        event: Final = _CALLBACK_EVENT.validate_python(payload)
        response: Final = event.get("response", event)
        self.logging_obj.post_call(  # pyright: ignore[reportUnknownMemberType]  # legacy logger is untyped
            input=self.messages,
            api_key=self.api_key,
            original_response=response if isinstance(response, str) else json.dumps(response),
        )
        return {"action": "unchanged"}

    def error(self, payload: object, /) -> None:
        event: Final = _CALLBACK_EVENT.validate_python(payload)
        self.logging_obj.model_call_details["provider_error"] = dict(  # mutable-ok: logger stores mutable details
            event
        )


_MESSAGES: Final = cast(  # cast-ok: generic classmethod loses the route Protocol parameters
    EndpointDispatch[RustMessages, RustAmessages],
    EndpointDispatch.native(
        route="messages",
        sync="messages",
        asynchronous="amessages",
        enabled=rust_enabled,
    ),
)  # cast-ok: generic classmethod cannot preserve the route Protocol parameters


def set_rust_messages(
    *,
    messages: RustMessages | None | Unchanged = UNCHANGED,
    amessages: RustAmessages | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(messages, Unchanged):
        if messages is None:
            _MESSAGES.sync.reset()
        else:
            _MESSAGES.sync.override(messages)
    if not isinstance(amessages, Unchanged):
        if amessages is None:
            _MESSAGES.asynchronous.reset()
        else:
            _MESSAGES.asynchronous.override(amessages)


def _adapt_response(response: dict[str, object]) -> dict[str, object]:
    return {  # mutable-ok: response metadata is attached to a mutable provider response
        **response,
        "_hidden_params": {  # mutable-ok: public response metadata is mutable
            "additional_headers": {"x-litellm-rust": "true"}  # mutable-ok: concrete response headers
        },
    }


def messages(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: float | httpx.Timeout | None,
    request_override: bool | None = None,
    eligible: bool = True,
    callback_adapter: OneShotCallbackHandle | None = None,
) -> dict[str, object] | None:
    return _MESSAGES.invoke(
        call=lambda rust_messages: rust_messages(
            model=model,
            body=body,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        fallback=lambda: None,
        adapt=identity,
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=request_override,
        eligible=eligible,
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
    request_override: bool | None = None,
    eligible: bool = True,
    callback_adapter: OneShotCallbackHandle | None = None,
) -> dict[str, object] | None:
    return await _MESSAGES.ainvoke(
        call=lambda rust_amessages: rust_amessages(
            model=model,
            body=body,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout),
            callback_adapter=callback_adapter,
        ),
        fallback=async_none,
        adapt=identity,
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=request_override,
        eligible=eligible,
    )


def dispatch_messages(
    *,
    asynchronous: bool,
    model: str,
    prepare: Callable[[], dict[str, object]],
    fallback: Callable[[], object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str,
    extra_headers: dict[str, object] | None,
    timeout: Callable[[], float | httpx.Timeout | None],
    request_override: bool | None,
    eligible: bool,
    callback_adapter: OneShotCallbackHandle,
) -> object:
    if asynchronous:

        async def async_fallback() -> object:
            pending: Final = fallback()
            if isinstance(pending, Awaitable):
                return await cast(Awaitable[object], pending)
            return pending

        return _adispatch_messages(
            model=model,
            prepare=prepare,
            fallback=async_fallback,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout=timeout,
            request_override=request_override,
            eligible=eligible,
            callback_adapter=callback_adapter,
        )
    return _MESSAGES.invoke(
        call=lambda native: native(
            model=model,
            body=prepare(),
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout()),
            callback_adapter=callback_adapter,
        ),
        fallback=fallback,
        adapt=_adapt_response,
        context=BridgeErrorContext(provider=custom_llm_provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


async def _adispatch_messages(
    *,
    model: str,
    prepare: Callable[[], dict[str, object]],
    fallback: Callable[[], Awaitable[object]],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str,
    extra_headers: dict[str, object] | None,
    timeout: Callable[[], float | httpx.Timeout | None],
    request_override: bool | None,
    eligible: bool,
    callback_adapter: OneShotCallbackHandle,
) -> object:
    return await _MESSAGES.ainvoke(
        call=lambda native: native(
            model=model,
            body=prepare(),
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_to_seconds(timeout()),
            callback_adapter=callback_adapter,
        ),
        fallback=fallback,
        adapt=_adapt_response,
        context=BridgeErrorContext(provider=custom_llm_provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )
