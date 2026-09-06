"""Thin Python wrapper for the native Rust Anthropic Messages bridge."""

from __future__ import annotations

from typing import Final

import httpx

from litellm.rust_bridge.bindings import UNCHANGED, NativeBinding, Unchanged
from litellm.rust_bridge.protocols import RustAmessages, RustMessages
from litellm.rust_bridge.request import (
    NativeMessagesRequest,
    NativeRequestCapabilities,
    NativeRequestContext,
    NativeRequestOptions,
    PreparedNativeCall,
    call_native,
)
from litellm.rust_bridge.runtime import DispatchResult, aattempt, attempt, identity
from litellm.rust_bridge.timeouts import timeout_to_seconds

_MESSAGES: Final[NativeBinding[RustMessages]] = NativeBinding(lambda native: native.messages)
_AMESSAGES: Final[NativeBinding[RustAmessages]] = NativeBinding(lambda native: native.amessages)


def set_rust_messages(
    *,
    messages: RustMessages | None | Unchanged = UNCHANGED,
    amessages: RustAmessages | None | Unchanged = UNCHANGED,
) -> None:
    if not isinstance(messages, Unchanged):
        if messages is None:
            _MESSAGES.reset()
        else:
            _MESSAGES.override(messages)
    if not isinstance(amessages, Unchanged):
        if amessages is None:
            _AMESSAGES.reset()
        else:
            _AMESSAGES.override(amessages)


def load_rust_messages() -> RustMessages | None:
    return _MESSAGES.load()


def load_rust_amessages() -> RustAmessages | None:
    return _AMESSAGES.load()


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
) -> DispatchResult[dict[str, object]]:
    return attempt(
        load=_MESSAGES.load,
        enabled=True,
        eligible=True,
        prepare=lambda: PreparedNativeCall(
            request=NativeMessagesRequest(model=model, body=body),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
            ),
            context=NativeRequestContext(
                capabilities=NativeRequestCapabilities(
                    execution_mode="sync",
                    stream=stream,
                    has_custom_client=has_custom_client,
                    has_agentic_hook=has_agentic_hook,
                )
            ),
        ),
        call=call_native,
        adapt=identity,
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
) -> DispatchResult[dict[str, object]]:
    return await aattempt(
        load=_AMESSAGES.load,
        enabled=True,
        eligible=True,
        prepare=lambda: PreparedNativeCall(
            request=NativeMessagesRequest(model=model, body=body),
            options=NativeRequestOptions(
                api_key=api_key,
                api_base=api_base,
                custom_llm_provider=custom_llm_provider,
                extra_headers=extra_headers,
                timeout_seconds=timeout_to_seconds(timeout),
            ),
            context=NativeRequestContext(
                capabilities=NativeRequestCapabilities(
                    execution_mode="async",
                    stream=stream,
                    has_custom_client=has_custom_client,
                    has_agentic_hook=has_agentic_hook,
                )
            ),
        ),
        call=call_native,
        adapt=identity,
    )
