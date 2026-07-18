"""Thin Python wrapper for the native Rust Anthropic Messages bridge."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Awaitable, Final, Protocol, Union, cast

import httpx

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


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustMessagesState:
    enabled: bool = False
    messages: RustMessages | None = None
    amessages: RustAmessages | None = None


def _env_enables_rust_messages() -> bool:
    return os.getenv("LITELLM_USE_RUST_MESSAGES", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_STATE: Final[_RustMessagesState] = _RustMessagesState(enabled=_env_enables_rust_messages())


def set_rust_messages(
    *,
    enabled: bool | _Unset = _UNSET,
    messages: RustMessages | None | _Unset = _UNSET,
    amessages: RustAmessages | None | _Unset = _UNSET,
) -> None:
    if not isinstance(enabled, _Unset):
        _STATE.enabled = enabled
    if not isinstance(messages, _Unset):
        _STATE.messages = messages
    if not isinstance(amessages, _Unset):
        _STATE.amessages = amessages


def rust_messages_enabled() -> bool:
    return _STATE.enabled


def load_rust_messages() -> RustMessages | None:
    if _STATE.messages is not None:
        return _STATE.messages
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return cast(RustMessages, getattr(native_bridge, "messages", None))


def load_rust_amessages() -> RustAmessages | None:
    if _STATE.amessages is not None:
        return _STATE.amessages
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return cast(RustAmessages, getattr(native_bridge, "amessages", None))


def messages(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: Union[float, httpx.Timeout] | None,
) -> dict[str, object] | None:
    rust_messages = load_rust_messages()
    if rust_messages is None:
        return None
    return rust_messages(
        model=model,
        body=body,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout_seconds=timeout_to_seconds(timeout),
    )


async def amessages(
    *,
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: Union[float, httpx.Timeout] | None,
) -> dict[str, object] | None:
    rust_amessages = load_rust_amessages()
    if rust_amessages is None:
        return None
    return await rust_amessages(
        model=model,
        body=body,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        timeout_seconds=timeout_to_seconds(timeout),
    )
