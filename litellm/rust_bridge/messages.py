from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Protocol, cast, runtime_checkable

import httpx

from litellm.rust_bridge._lifecycle import (
    LIFECYCLE_STARTED_KEY,
    LOGGING_OBJECT_KEY,
    LifecycleOwner,
    NativeLifecycle,
    NativeLifecycleBindings,
    NativeOutcome,
    TerminalAction,
    advance_host,
    build_call_arguments,
    deployment_failure,
    deployment_pre,
    deployment_success,
    drive_async,
    drive_sync,
    host_result,
    invoke_terminal,
    map_native_error,
    owns_lifecycle,
    restore_correlation_context,
)
from litellm.rust_bridge._lifecycle import (
    initialize_logging as initialize_lifecycle_logging,
)
from litellm.rust_bridge.bindings import NativeBinding, native_exception_types
from litellm.rust_bridge.provenance import mark_native_response
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.types.router import GenericLiteLLMParams


class RustMessages(Protocol):
    def __call__(
        self, arguments: dict[str, object]
    ) -> AnthropicMessagesResponse: ...  # mutable-ok: native bridge retains and updates Python argument objects


class RustAmessages(Protocol):
    def __call__(
        self, arguments: dict[str, object]
    ) -> Awaitable[
        AnthropicMessagesResponse | AsyncIterator[bytes]
    ]: ...  # mutable-ok: native bridge retains and updates Python argument objects


@runtime_checkable
class _MessagesLogging(Protocol):
    stream: bool
    completion_start_time: datetime | None
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
    body: Final = arguments.get("body")
    streaming: Final = isinstance(body, Mapping) and body.get("stream") is True
    from litellm.litellm_core_utils.litellm_logging import Logging

    logger: Final = initialize_lifecycle_logging(
        arguments, asynchronous, "anthropic_messages" if streaming else "messages"
    )
    if (
        streaming
        and isinstance(logger, Logging)
        and isinstance(body, Mapping)
        and not hasattr(logger, "optional_params")
    ):
        logger.optional_params = dict(body)  # mutable-ok: Logging requires mutable provider options
    return logger


class MessagesStream(AsyncIterator[bytes]):
    def __init__(self, source: AsyncIterator[bytes], arguments: Mapping[str, object], logger: object) -> None:
        self.source: Final = source
        self.arguments: Mapping[str, object] = arguments
        self.logger: _MessagesLogging | None = logger if isinstance(logger, _MessagesLogging) else None
        self.started: bool = False

    async def __anext__(self) -> bytes:
        try:
            chunk: Final = await anext(self.source)
        except StopAsyncIteration:
            self._release()
            raise
        except Exception as error:
            mapped: Final = map_native_error(error, self.arguments, "messages")
            self._release()
            raise mapped
        if not self.started:
            self.started = True
            if self.logger is not None:
                started: Final = datetime.now()  # noqa: DTZ005  # Logging uses naive local timestamps
                self.logger.completion_start_time = started
                self.logger.model_call_details["completion_start_time"] = started
        return chunk

    async def aclose(self) -> None:
        from litellm.llms.anthropic.experimental_pass_through.messages.streaming_iterator import aclose_if_supported

        try:
            await aclose_if_supported(self.source)
        finally:
            self._release()

    def _release(self) -> None:
        self.arguments = MappingProxyType({})
        self.logger = None


def _arguments(
    arguments: Mapping[str, object] | None,
    model: str,
    body: dict[str, object],  # mutable-ok: native bridge retains and updates Python argument objects
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,  # mutable-ok: native bridge retains and updates Python argument objects
    timeout: float | httpx.Timeout | None,
    logging_obj: object | None,
    litellm_params: GenericLiteLLMParams | None,
    messages: object,
    lifecycle_owner: LifecycleOwner,
) -> dict[str, object]:  # mutable-ok: native bridge retains and updates Python argument objects
    return build_call_arguments(
        arguments,
        {
            "model": model,
            "body": body,
            "api_key": api_key,
            "api_base": api_base,
            "custom_llm_provider": custom_llm_provider,
            "extra_headers": extra_headers,
            "timeout_seconds": timeout_to_seconds(timeout),
            **({"messages": messages} if messages is not None else {}),
            **({"litellm_params": litellm_params} if litellm_params is not None else {}),
        },
        logging_obj=logging_obj,
        lifecycle_owner=lifecycle_owner,
    )


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
    request_arguments: Mapping[str, object] | None = None,
    logging_obj: object | None = None,
    litellm_params: GenericLiteLLMParams | None = None,
    messages: object = None,
    lifecycle_owner: LifecycleOwner = LifecycleOwner.BRIDGE,
) -> AnthropicMessagesResponse | None:
    implementation: Final = load_rust_messages()
    if implementation is None:
        return None
    call_arguments: Final = _arguments(
        request_arguments if request_arguments is not None else arguments,
        model,
        body,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout,
        logging_obj,
        litellm_params,
        messages,
        lifecycle_owner,
    )
    try:
        return mark_native_response(implementation(arguments=call_arguments))
    except Exception as error:  # noqa: BLE001  # only explicit declines before lifecycle setup may fall back
        exceptions: Final = native_exception_types()
        if (
            exceptions is not None
            and isinstance(error, exceptions[0])
            and not call_arguments.get(LIFECYCLE_STARTED_KEY)
        ):
            return None
        raise map_native_error(error, call_arguments, "messages")


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
    request_arguments: Mapping[str, object] | None = None,
    logging_obj: object | None = None,
    litellm_params: GenericLiteLLMParams | None = None,
    messages: object = None,
    lifecycle_owner: LifecycleOwner = LifecycleOwner.BRIDGE,
) -> AnthropicMessagesResponse | AsyncIterator[bytes] | None:
    implementation: Final = load_rust_amessages()
    if implementation is None:
        return None
    call_arguments: Final = _arguments(
        request_arguments if request_arguments is not None else arguments,
        model,
        body,
        api_key,
        api_base,
        custom_llm_provider,
        extra_headers,
        timeout,
        logging_obj,
        litellm_params,
        messages,
        lifecycle_owner,
    )
    try:
        response: Final = await implementation(arguments=call_arguments)
        if isinstance(response, AsyncIterator):
            return MessagesStream(response, call_arguments, call_arguments.get(LOGGING_OBJECT_KEY))
        return mark_native_response(response)
    except Exception as error:  # noqa: BLE001  # only explicit declines before lifecycle setup may fall back
        exceptions: Final = native_exception_types()
        if (
            exceptions is not None
            and isinstance(error, exceptions[0])
            and not call_arguments.get(LIFECYCLE_STARTED_KEY)
        ):
            return None
        raise map_native_error(error, call_arguments, "messages")


class _MessagesLifecycle(NativeLifecycle, Protocol):
    def failed_after_provider_response(self) -> bool: ...


class _MessagesBindings(NativeLifecycleBindings, Protocol):
    Lifecycle: Callable[[bool, bool], _MessagesLifecycle]
    build_request: Callable[
        [dict[str, object], object], object
    ]  # mutable-ok: native bridge retains and updates Python argument objects
    pre_call: Callable[[object], None]
    send: Callable[[object, object], Awaitable[AnthropicMessagesResponse | AsyncIterator[bytes]]]
    send_sync: Callable[[object], AnthropicMessagesResponse]


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
        self.lifecycle_owned: Final = owns_lifecycle(arguments)
        self.arguments[LIFECYCLE_STARTED_KEY] = True
        self.state: object | None = None
        self.response: object = None
        self.error: BaseException | None = None
        self.start: datetime = datetime.now()  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract
        self.end: datetime | None = None
        self.stream_transferred: bool = False

    def invoke(self) -> tuple[bool, object]:
        return self.bindings.invoke(self.machine, self)

    def setup(self) -> None:
        self.logger = initialize_logging(self.arguments, self.asynchronous)
        self.arguments[LOGGING_OBJECT_KEY] = self.logger
        body: Final = self.arguments.get("body")
        if isinstance(body, Mapping) and body.get("stream") is True and isinstance(self.logger, _MessagesLogging):
            self.logger.stream = True
            self.logger.model_call_details["stream"] = True

    async def deployment_pre(self) -> None:
        if not self.lifecycle_owned:
            return
        self.current = await deployment_pre(self.current, "anthropic_messages")
        self.current[LOGGING_OBJECT_KEY] = self.logger

    def build_request(self) -> None:
        if self.logger is None:
            raise RuntimeError("messages logging was not initialized")
        self.state = self.bindings.build_request(self.current, self.logger)

    def pre_call(self) -> None:
        self.bindings.pre_call(self.state)

    def send_sync(self) -> None:
        self.response = self.bindings.send_sync(self.state)
        self.end = datetime.now()  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract

    async def send(self) -> None:
        self.response = await self.bindings.send(self.state, self)
        self.stream_transferred = isinstance(self.response, AsyncIterator)
        self.end = datetime.now()  # noqa: DTZ005  # Logging preserves the legacy naive timestamp contract

    async def deployment_success(self) -> None:
        from litellm.types.utils import CallTypes

        if self.stream_transferred:
            return
        if self.lifecycle_owned:
            self.response = await deployment_success(self.current, self.response, CallTypes.aanthropic_messages)

    async def complete_stream(self, response: object, error: str | None, end_time: float) -> None:
        from litellm.exceptions import APIError

        if not isinstance(self.logger, _MessagesLogging):
            raise TypeError("messages stream logger was released before completion")
        logger: Final = self.logger
        end: Final = datetime.fromtimestamp(end_time, tz=self.start.tzinfo)
        roots: Final = (self.arguments, self.current, self.state)
        try:
            if error is not None:
                failure: Final = APIError(
                    status_code=500,
                    message=error,
                    llm_provider=str(self.current.get("custom_llm_provider") or "anthropic"),
                    model=str(self.current.get("model") or ""),
                )
                invoke_terminal("sync_failure", roots, logger, None, failure, self.start, end)
                pending: Final = invoke_terminal("async_failure", roots, logger, None, failure, self.start, end)
                if isinstance(pending, Awaitable):
                    await pending
                return
            complete: Final = logger._handle_anthropic_messages_response_logging(response)  # pyright: ignore[reportPrivateUsage]  # existing Messages logging transform
            logger.model_call_details["complete_streaming_response"] = complete
            try:
                invoke_terminal("async_success", roots, logger, None, complete, self.start, end)
            finally:
                invoke_terminal("sync_success_if_needed", roots, logger, None, complete, self.start, end)
        finally:
            restore_correlation_context(logger)
            self.state = None
            self.logger = None
            self.arguments = {}  # mutable-ok: native driver requires dict fields after releasing request roots
            self.current = {}  # mutable-ok: native driver requires dict fields after releasing request roots

    async def deployment_failure(self) -> None:
        if not self.lifecycle_owned:
            return
        await deployment_failure(self.current, self.error, "anthropic_messages")

    def terminal(self, action: TerminalAction, value: object) -> object:
        if self.stream_transferred or not self.lifecycle_owned:
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

    async def async_failure(self) -> object:
        result: Final = self.terminal("async_failure", self.error)
        return await result if isinstance(result, Awaitable) else result

    def restore(self) -> None:
        if not self.stream_transferred and self.lifecycle_owned:
            restore_correlation_context(self.logger)

    def advance(self, outcome: NativeOutcome, error: BaseException | None = None) -> None:
        advance_host(self, outcome, map_native_error(error, self.arguments, "messages"))

    def result(self) -> object:
        if self.machine.complete():
            response: Final = self.response
            if self.stream_transferred:
                self.response = None
            return response
        return host_result(self)


def _drive_sync(  # pyright: ignore[reportUnusedFunction]  # called by the native extension
    arguments: dict[str, object],
    bindings: _MessagesBindings,  # mutable-ok: native bridge retains and updates Python argument objects
) -> AnthropicMessagesResponse:
    return cast(AnthropicMessagesResponse, drive_sync(_MessagesHost(arguments, False, bindings)))


async def _drive_async(  # pyright: ignore[reportUnusedFunction]  # called by the native extension
    arguments: dict[str, object],
    bindings: _MessagesBindings,  # mutable-ok: native bridge retains and updates Python argument objects
) -> AnthropicMessagesResponse | AsyncIterator[bytes]:
    return cast(
        AnthropicMessagesResponse | AsyncIterator[bytes], await drive_async(_MessagesHost(arguments, True, bindings))
    )


__all__ = (
    "amessages",
    "initialize_logging",
    "invoke_terminal",
    "load_rust_amessages",
    "load_rust_messages",
    "messages",
    "set_rust_messages",
)
