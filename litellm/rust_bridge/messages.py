from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime
from typing import Final, Protocol, cast

import httpx

from litellm.rust_bridge._lifecycle import (
    initialize_logging as initialize_lifecycle_logging,
)
from litellm.rust_bridge._lifecycle import invoke_terminal
from litellm.rust_bridge.timeouts import timeout_to_seconds


class RustMessages(Protocol):
    def __call__(self, arguments: dict[str, object]) -> dict[str, object]: ...


class RustAmessages(Protocol):
    def __call__(self, arguments: dict[str, object]) -> Awaitable[dict[str, object]]: ...


class _MessagesLogging(Protocol):
    model_call_details: dict[str, object]

    def _handle_anthropic_messages_response_logging(self, result: object) -> object: ...


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustMessagesState:
    messages: RustMessages | None = None
    amessages: RustAmessages | None = None


_STATE: Final = _RustMessagesState()


def set_rust_messages(
    *,
    messages: RustMessages | None | _Unset = _UNSET,
    amessages: RustAmessages | None | _Unset = _UNSET,
) -> None:
    if not isinstance(messages, _Unset):
        _STATE.messages = messages
    if not isinstance(amessages, _Unset):
        _STATE.amessages = amessages


def load_rust_messages() -> RustMessages | None:
    if _STATE.messages is not None:
        return _STATE.messages
    from litellm.rust_bridge import get_native_bridge

    bridge: Final = get_native_bridge()
    return cast(RustMessages, getattr(bridge, "messages", None)) if bridge is not None else None


def load_rust_amessages() -> RustAmessages | None:
    if _STATE.amessages is not None:
        return _STATE.amessages
    from litellm.rust_bridge import get_native_bridge

    bridge: Final = get_native_bridge()
    return cast(RustAmessages, getattr(bridge, "amessages", None)) if bridge is not None else None


def initialize_logging(arguments: dict[str, object], asynchronous: bool) -> object:
    return initialize_lifecycle_logging(arguments, asynchronous, "messages")


class _RetainedMessagesResponse(dict[str, object]):
    def __init__(self, response: dict[str, object], roots: object, logger: object, start_time: datetime) -> None:
        super().__init__(response)
        self._roots = roots
        self._logger = cast(_MessagesLogging, logger)
        self._start_time = start_time
        self._completed = False

    def complete(self) -> None:
        if self._completed:
            return
        self._completed = True
        roots, self._roots = self._roots, None
        try:
            complete_response = self._logger._handle_anthropic_messages_response_logging(  # pyright: ignore[reportPrivateUsage]  # existing Messages logging transform
                self
            )
            self._logger.model_call_details["complete_streaming_response"] = complete_response
            end_time = datetime.now()
            try:
                invoke_terminal(
                    "async_success",
                    roots,
                    self._logger,
                    None,
                    complete_response,
                    self._start_time,
                    end_time,
                )
            finally:
                invoke_terminal(
                    "sync_success_if_needed",
                    roots,
                    self._logger,
                    None,
                    complete_response,
                    self._start_time,
                    end_time,
                )
        finally:
            from litellm import utils

            utils._restore_correlation_context_if_supported(self._logger)  # pyright: ignore[reportPrivateUsage]  # lifecycle cleanup has no public wrapper


def retain_stream_response(
    response: dict[str, object], roots: object, logger: object, start_time: datetime
) -> dict[str, object]:
    return _RetainedMessagesResponse(response, roots, logger, start_time)


def _arguments(
    arguments: dict[str, object],
    model: str,
    body: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    timeout: float | httpx.Timeout | None,
) -> dict[str, object]:
    return {
        **arguments,
        "model": model,
        "body": body,
        "api_key": api_key,
        "api_base": api_base,
        "custom_llm_provider": custom_llm_provider,
        "extra_headers": extra_headers,
        "timeout_seconds": timeout_to_seconds(timeout),
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
    arguments: dict[str, object] | None = None,
) -> dict[str, object] | None:
    implementation: Final = load_rust_messages()
    if implementation is None:
        return None
    return implementation(
        arguments=_arguments(
            arguments or {}, model, body, api_key, api_base, custom_llm_provider, extra_headers, timeout
        )
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
    arguments: dict[str, object] | None = None,
) -> dict[str, object] | None:
    implementation: Final = load_rust_amessages()
    if implementation is None:
        return None
    return await implementation(
        arguments=_arguments(
            arguments or {}, model, body, api_key, api_base, custom_llm_provider, extra_headers, timeout
        )
    )


__all__ = [
    "amessages",
    "initialize_logging",
    "invoke_terminal",
    "load_rust_amessages",
    "load_rust_messages",
    "messages",
    "retain_stream_response",
    "set_rust_messages",
]
