"""Thin Python wrapper for the native Rust Anthropic Messages bridge."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol

import httpx

from litellm.rust_bridge.bindings import UNCHANGED, Unchanged
from litellm.rust_bridge.configuration import rust_enabled
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    RustEndpoint,
    async_none,
    identity,
)
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
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


_MESSAGES: Final[RustEndpoint[RustMessages, RustAmessages]] = RustEndpoint.native(
    route="messages",
    sync="messages",
    asynchronous="amessages",
    enabled=rust_enabled,
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
        ),
        fallback=lambda: None,
        adapt=identity,
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=request_override,
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
        ),
        fallback=async_none,
        adapt=identity,
        context=BridgeErrorContext(provider=custom_llm_provider or "", model=model),
        request_override=request_override,
    )
