"""Thin Python wrapper for the native Rust Anthropic Messages bridge."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Final, Protocol

import httpx

from litellm.rust_bridge.bindings import UNSET, NativeBinding, Unset
from litellm.rust_bridge.runtime import (
    BridgeErrorContext,
    FallbackMode,
    ainvoke,
    async_none,
    identity,
    invoke,
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


_MESSAGES: Final = NativeBinding[RustMessages]("messages")
_AMESSAGES: Final = NativeBinding[RustAmessages]("amessages")


def set_rust_messages(
    *,
    messages: RustMessages | None | Unset = UNSET,
    amessages: RustAmessages | None | Unset = UNSET,
) -> None:
    _MESSAGES.update(messages)
    _AMESSAGES.update(amessages)


def load_rust_messages() -> RustMessages | None:
    return _MESSAGES.load()


def load_rust_amessages() -> RustAmessages | None:
    return _AMESSAGES.load()


def _context(model: str, custom_llm_provider: str | None) -> BridgeErrorContext:
    return BridgeErrorContext(
        route="messages",
        provider=custom_llm_provider or "anthropic",
        model=model,
    )


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
    native: Final = load_rust_messages()
    return invoke(
        native_call=(
            None
            if native is None
            else lambda: native(
                model=model,
                body=body,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
            )
        ),
        fallback=lambda: None,
        adapt=identity,
        mode=FallbackMode.PYTHON,
        context=_context(model, custom_llm_provider),
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
    native: Final = load_rust_amessages()
    return await ainvoke(
        native_call=(
            None
            if native is None
            else lambda: native(
                model=model,
                body=body,
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
            )
        ),
        fallback=async_none,
        adapt=identity,
        mode=FallbackMode.PYTHON,
        context=_context(model, custom_llm_provider),
    )
