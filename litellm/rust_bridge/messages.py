"""Thin Python wrapper for the native Rust Anthropic Messages bridge."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Final, Protocol, cast  # noqa: TID251  # runtime typing constructs

import httpx

from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.callbacks import OneShotCallbackHandle
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.runtime import BridgeErrorContext, EndpointDispatch
from litellm.rust_bridge.timeouts import timeout_to_seconds


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
class NativeMessagesRequest:
    model: str
    body: dict[str, object]
    api_key: str | None
    api_base: str | None
    custom_llm_provider: str | None
    extra_headers: dict[str, object] | None
    timeout: float | httpx.Timeout | None
    callback_adapter: OneShotCallbackHandle


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
    sync: RustMessages | None | Unchanged = UNCHANGED,
    asynchronous: RustAmessages | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(sync, Unchanged):
        if sync is None:
            _MESSAGES.sync.reset()
        else:
            _MESSAGES.sync.override(sync)
    if not isinstance(asynchronous, Unchanged):
        if asynchronous is None:
            _MESSAGES.asynchronous.reset()
        else:
            _MESSAGES.asynchronous.override(asynchronous)


def _adapt_response(response: dict[str, object]) -> dict[str, object]:
    return {  # mutable-ok: response metadata is attached to a mutable provider response
        **response,
        "_hidden_params": {  # mutable-ok: public response metadata is mutable
            "additional_headers": {"x-litellm-rust": "true"}  # mutable-ok: concrete response headers
        },
    }


def dispatch_messages(
    *,
    prepare: Callable[[], NativeMessagesRequest],
    model: str,
    fallback: Callable[[], object],
    provider: str,
    request_override: bool | None,
    eligible: bool,
) -> object:
    return _MESSAGES.invoke(
        prepare=prepare,
        call=_call_messages,
        fallback=fallback,
        adapt=_adapt_response,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


async def adispatch_messages(
    *,
    prepare: Callable[[], NativeMessagesRequest],
    model: str,
    fallback: Callable[[], Awaitable[object]],
    provider: str,
    request_override: bool | None,
    eligible: bool,
) -> object:
    return await _MESSAGES.ainvoke(
        prepare=prepare,
        call=_call_amessages,
        fallback=fallback,
        adapt=_adapt_response,
        context=BridgeErrorContext(provider=provider, model=model),
        request_override=request_override,
        eligible=eligible,
    )


def _call_messages(native: RustMessages, request: NativeMessagesRequest) -> dict[str, object]:
    return native(
        model=request.model,
        body=request.body,
        api_key=request.api_key,
        api_base=request.api_base,
        custom_llm_provider=request.custom_llm_provider,
        extra_headers=request.extra_headers,
        timeout_seconds=timeout_to_seconds(request.timeout),
        callback_adapter=request.callback_adapter,
    )


def _call_amessages(native: RustAmessages, request: NativeMessagesRequest) -> Awaitable[dict[str, object]]:
    return native(
        model=request.model,
        body=request.body,
        api_key=request.api_key,
        api_base=request.api_base,
        custom_llm_provider=request.custom_llm_provider,
        extra_headers=request.extra_headers,
        timeout_seconds=timeout_to_seconds(request.timeout),
        callback_adapter=request.callback_adapter,
    )
