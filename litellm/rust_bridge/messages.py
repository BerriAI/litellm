from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Final, Protocol, cast

import httpx

from litellm.rust_bridge._lifecycle import (
    LOGGING_OBJECT_KEY,
    NativeLifecycle,
    NativeLifecycleBindings,
    NativeOutcome,
    TerminalAction,
    advance_host,
    deployment_failure,
    deployment_pre,
    deployment_success,
    drive_async,
    drive_sync,
    host_result,
    invoke_terminal,
    restore_correlation_context,
)
from litellm.rust_bridge._lifecycle import (
    initialize_logging as initialize_lifecycle_logging,
)
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


class RustMessages(Protocol):
    def __call__(
        self, arguments: dict[str, object]
    ) -> AnthropicMessagesResponse: ...  # mutable-ok: native bridge retains and updates Python argument objects


class RustAmessages(Protocol):
    def __call__(
        self, arguments: dict[str, object]
    ) -> Awaitable[
        AnthropicMessagesResponse
    ]: ...  # mutable-ok: native bridge retains and updates Python argument objects


class _MessagesLogging(Protocol):
    model_call_details: dict[str, object]  # mutable-ok: native bridge retains and updates Python argument objects

    def _handle_anthropic_messages_response_logging(self, result: object) -> object: ...


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()


@dataclass(slots=True)
class _RustMessagesState:
    messages: RustMessages | None | _Unset = _UNSET
    amessages: RustAmessages | None | _Unset = _UNSET


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
    if not isinstance(_STATE.messages, _Unset):
        return _STATE.messages
    return _MESSAGES.load()


def load_rust_amessages() -> RustAmessages | None:
    if not isinstance(_STATE.amessages, _Unset):
        return _STATE.amessages
    return _AMESSAGES.load()


def initialize_logging(
    arguments: dict[str, object], asynchronous: bool
) -> object:  # mutable-ok: native bridge retains and updates Python argument objects
    return initialize_lifecycle_logging(arguments, asynchronous, "messages")


class _RetainedMessagesResponse(
    dict[str, object]
):  # mutable-ok: native bridge retains and updates Python argument objects
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
    arguments: dict[str, object],  # mutable-ok: native bridge retains and updates Python argument objects
    model: str,
    body: dict[str, object],  # mutable-ok: native bridge retains and updates Python argument objects
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,  # mutable-ok: native bridge retains and updates Python argument objects
    timeout: float | httpx.Timeout | None,
) -> dict[str, object]:  # mutable-ok: native bridge retains and updates Python argument objects
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
    body: dict[str, object],  # mutable-ok: native bridge retains and updates Python argument objects
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,  # mutable-ok: native bridge retains and updates Python argument objects
    timeout: float | httpx.Timeout | None,
    arguments: dict[str, object] | None = None,  # mutable-ok: native bridge retains and updates Python argument objects
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
    body: dict[str, object],  # mutable-ok: native bridge retains and updates Python argument objects
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,  # mutable-ok: native bridge retains and updates Python argument objects
    timeout: float | httpx.Timeout | None,
    arguments: dict[str, object] | None = None,  # mutable-ok: native bridge retains and updates Python argument objects
) -> AnthropicMessagesResponse | None:
    implementation: Final = load_rust_amessages()
    if implementation is None:
        return None
    return await implementation(
        arguments=_arguments(
            arguments or {}, model, body, api_key, api_base, custom_llm_provider, extra_headers, timeout
        )
    )


class _MessagesLifecycle(NativeLifecycle, Protocol):
    def failed_after_provider_response(self) -> bool: ...


class _MessagesBindings(NativeLifecycleBindings, Protocol):
    Lifecycle: Callable[[bool, bool], _MessagesLifecycle]
    prepare: Callable[
        [dict[str, object], object], object
    ]  # mutable-ok: native bridge retains and updates Python argument objects
    send: Callable[[object], Awaitable[AnthropicMessagesResponse]]
    send_sync: Callable[[object], AnthropicMessagesResponse]
    committed_failure: Callable[[], None]


class _MessagesHost:
    def __init__(
        self, arguments: dict[str, object], asynchronous: bool, bindings: _MessagesBindings
    ) -> None:  # mutable-ok: native bridge retains and updates Python argument objects
        from litellm import utils

        self.bindings: _MessagesBindings = bindings
        self.machine: _MessagesLifecycle = bindings.Lifecycle(asynchronous, utils.is_internal_call.get())
        self.arguments: dict[str, object] = (
            arguments  # mutable-ok: native bridge retains and updates Python argument objects
        )
        self.current: dict[str, object] = (
            arguments  # mutable-ok: native bridge retains and updates Python argument objects
        )
        self.asynchronous: bool = asynchronous
        self.logger: object | None = arguments.get(LOGGING_OBJECT_KEY)
        self.lifecycle_owned: bool = self.logger is None
        self.state: object | None = None
        self.response: object = None
        self.error: BaseException | None = None
        self.start: datetime = datetime.now()  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract
        self.end: datetime | None = None
        self.streaming: bool = False

    def invoke(self) -> tuple[bool, object]:
        return self.bindings.invoke(self.machine, self)

    def setup(self) -> None:
        self.logger = initialize_logging(self.arguments, self.asynchronous)
        self.arguments[LOGGING_OBJECT_KEY] = self.logger
        self.streaming = getattr(self.logger, "stream", False) is True

    async def deployment_pre(self) -> None:
        if not self.lifecycle_owned:
            return
        self.current = await deployment_pre(self.current, "anthropic_messages")
        self.current[LOGGING_OBJECT_KEY] = self.logger

    def prepare(self) -> None:
        if self.logger is None:
            raise RuntimeError("messages logging was not initialized")
        self.state = self.bindings.prepare(self.current, self.logger)

    def send_sync(self) -> None:
        self.response = self.bindings.send_sync(self.state)
        self.end = datetime.now()  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract

    async def send(self) -> None:
        self.response = await self.bindings.send(self.state)
        self.end = datetime.now()  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract

    async def deployment_success(self) -> None:
        from litellm.types.utils import CallTypes

        if self.lifecycle_owned:
            self.response = await deployment_success(self.current, self.response, CallTypes.aanthropic_messages)
        if not self.streaming:
            return
        if self.logger is None:
            raise RuntimeError("messages logging was not initialized")
        self.response = retain_stream_response(
            cast(AnthropicMessagesResponse, self.response),
            (self.arguments, self.current, self.state),
            cast(_MessagesLogging, self.logger),
            self.start,
        )

    async def deployment_failure(self) -> None:
        if not self.lifecycle_owned:
            return
        await deployment_failure(self.current, self.error, "anthropic_messages")

    def terminal(self, action: TerminalAction, value: object) -> object:
        if self.streaming or not self.lifecycle_owned:
            return None
        if self.logger is None or self.end is None:
            raise RuntimeError("messages terminal state was not initialized")
        return invoke_terminal(
            action,
            (self.arguments, self.current, self.state),
            self.logger,
            None,
            value,
            self.start,
            self.end,
        )

    def sync_success(self) -> object:
        return self.terminal("sync_success", self.response)

    def async_success(self) -> object:
        return self.terminal("async_success", self.response)

    def sync_success_if_needed(self) -> object:
        return self.terminal("sync_success_if_needed", self.response)

    def sync_failure(self) -> object:
        return self.terminal("sync_failure", self.error)

    def async_failure(self) -> object:
        return self.terminal("async_failure", self.error)

    def restore(self) -> None:
        if not self.streaming and self.lifecycle_owned:
            restore_correlation_context(self.logger)

    def advance(self, outcome: NativeOutcome, error: BaseException | None = None) -> None:
        advance_host(self, outcome, error)

    def result(self) -> object:
        if self.machine.complete():
            return self.response
        if self.machine.failed_after_provider_response():
            self.bindings.committed_failure()
        return host_result(self)


def _drive_sync(  # pyright: ignore[reportUnusedFunction]  # called by the native extension
    arguments: dict[str, object],
    bindings: _MessagesBindings,  # mutable-ok: native bridge retains and updates Python argument objects
) -> AnthropicMessagesResponse:
    return cast(AnthropicMessagesResponse, drive_sync(_MessagesHost(arguments, False, bindings)))


async def _drive_async(  # pyright: ignore[reportUnusedFunction]  # called by the native extension
    arguments: dict[str, object],
    bindings: _MessagesBindings,  # mutable-ok: native bridge retains and updates Python argument objects
) -> AnthropicMessagesResponse:
    return cast(AnthropicMessagesResponse, await drive_async(_MessagesHost(arguments, True, bindings)))


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
