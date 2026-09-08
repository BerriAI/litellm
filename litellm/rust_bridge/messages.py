from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Protocol, cast

import httpx

from litellm.rust_bridge._lifecycle import (
    initialize_logging as initialize_lifecycle_logging,
)
from litellm.rust_bridge._lifecycle import invoke_terminal
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


class RustMessages(Protocol):
    def __call__(self, arguments: dict[str, object]) -> AnthropicMessagesResponse: ...


class RustAmessages(Protocol):
    def __call__(self, arguments: dict[str, object]) -> Awaitable[AnthropicMessagesResponse]: ...


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


def _as_messages(value: object) -> RustMessages | None:
    return cast(RustMessages, value) if callable(value) else None  # cast-ok: callable native binding


def _as_amessages(value: object) -> RustAmessages | None:
    return cast(RustAmessages, value) if callable(value) else None  # cast-ok: callable native binding


_MESSAGES: Final = NativeBinding("messages", validate=_as_messages)
_AMESSAGES: Final = NativeBinding("amessages", validate=_as_amessages)


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
    return _MESSAGES.load()


def load_rust_amessages() -> RustAmessages | None:
    if _STATE.amessages is not None:
        return _STATE.amessages
    return _AMESSAGES.load()


def initialize_logging(arguments: dict[str, object], asynchronous: bool) -> object:
    return initialize_lifecycle_logging(arguments, asynchronous, "messages")


class _RetainedMessagesResponse(dict[str, object]):
    def __init__(
        self, response: AnthropicMessagesResponse, roots: object, logger: _MessagesLogging, start_time: datetime
    ) -> None:
        super().__init__(response)
        self._roots = roots
        self._logger = logger
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
            end_time = datetime.now(tz=self._start_time.tzinfo or timezone.utc)
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
    response: AnthropicMessagesResponse, roots: object, logger: _MessagesLogging, start_time: datetime
) -> AnthropicMessagesResponse:
    return cast(  # cast-ok: dict subclass preserves the Anthropic response mapping contract
        AnthropicMessagesResponse, _RetainedMessagesResponse(response, roots, logger, start_time)
    )


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
    return {  # mutable-ok: the native bridge requires a concrete argument bag
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
) -> AnthropicMessagesResponse | None:
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
) -> AnthropicMessagesResponse | None:
    implementation: Final = load_rust_amessages()
    if implementation is None:
        return None
    return await implementation(
        arguments=_arguments(
            arguments or {}, model, body, api_key, api_base, custom_llm_provider, extra_headers, timeout
        )
    )


__all__ = (
    "amessages",
    "initialize_logging",
    "invoke_terminal",
    "load_rust_amessages",
    "load_rust_messages",
    "messages",
    "retain_stream_response",
    "set_rust_messages",
)
