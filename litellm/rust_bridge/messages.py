"""Thin Python wrapper for the native Rust Anthropic Messages bridge."""

from __future__ import annotations

from typing import Final

import httpx

from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.protocols import RustAmessages, RustMessages
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    EndpointDispatch,
    NativeErrorPolicy,
    always_enabled,
    async_none,
    identity,
)
from litellm.rust_bridge.timeouts import timeout_to_seconds

_MESSAGES: Final[EndpointDispatch[RustMessages, RustAmessages]] = EndpointDispatch.native(
    route="messages",
    sync=lambda native: native.messages,
    asynchronous=lambda native: native.amessages,
    enabled=always_enabled,
    error_policy=NativeErrorPolicy.PROPAGATE,
)


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
) -> dict[str, object] | None:
    return _MESSAGES.invoke(
        prepare=lambda: timeout_to_seconds(timeout),
        call=lambda rust_messages, timeout_seconds: rust_messages(
            model=model,
            body=body,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_seconds,
        ),
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
) -> dict[str, object] | None:
    return await _MESSAGES.ainvoke(
        prepare=lambda: timeout_to_seconds(timeout),
        call=lambda rust_amessages, timeout_seconds: rust_amessages(
            model=model,
            body=body,
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            timeout_seconds=timeout_seconds,
        ),
        fallback=async_none,
        adapt=identity,
        error_context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
    )
