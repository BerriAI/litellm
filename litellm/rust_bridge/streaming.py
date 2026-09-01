from __future__ import annotations

import json
from collections.abc import AsyncIterator, Generator, Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NoReturn, Protocol, TypeAlias, runtime_checkable

import httpx

from litellm.exceptions import APIError
from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.timeouts import timeout_to_seconds

if TYPE_CHECKING:
    from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject, ResponsesAPIStreamingResponse

StreamApi: TypeAlias = Literal["chat_completions", "messages", "responses"]
StreamTransport: TypeAlias = Literal["http", "websocket"]
Event: TypeAlias = Mapping[str, object]


@runtime_checkable
class RustEventStream(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]: ...

    def next_event(self) -> Event | None: ...

    async def anext_event(self) -> Event | None: ...

    def close(self) -> None: ...

    async def aclose(self) -> None: ...


class RustStreamOpen(Protocol):
    def __call__(
        self,
        request: Mapping[str, object],
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
        litellm_metadata: Mapping[str, object] | None,
    ) -> RustEventStream: ...


class RustAsyncStreamOpen(Protocol):
    async def __call__(
        self,
        request: Mapping[str, object],
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
        litellm_metadata: Mapping[str, object] | None,
    ) -> RustEventStream: ...


class RustStreamCapability(Protocol):
    def __call__(self, api: str, provider: str, transport: str) -> bool: ...


@runtime_checkable
class ObjectAwaitable(Protocol):
    def __await__(self) -> Generator[object, None, object]: ...


class _Unset:
    pass


_UNSET: Final = _Unset()


@dataclass(slots=True)
class _RustStreamingState:
    capability: RustStreamCapability | None = None
    chat: RustStreamOpen | None = None
    achat: RustAsyncStreamOpen | None = None
    messages: RustStreamOpen | None = None
    amessages: RustAsyncStreamOpen | None = None
    responses: RustStreamOpen | None = None
    aresponses: RustAsyncStreamOpen | None = None


_STATE: Final = _RustStreamingState()


def set_rust_streaming(
    *,
    capability: RustStreamCapability | None | _Unset = _UNSET,
    chat: RustStreamOpen | None | _Unset = _UNSET,
    achat: RustAsyncStreamOpen | None | _Unset = _UNSET,
    messages: RustStreamOpen | None | _Unset = _UNSET,
    amessages: RustAsyncStreamOpen | None | _Unset = _UNSET,
    responses: RustStreamOpen | None | _Unset = _UNSET,
    aresponses: RustAsyncStreamOpen | None | _Unset = _UNSET,
) -> None:
    if not isinstance(capability, _Unset):
        _STATE.capability = capability
    if not isinstance(chat, _Unset):
        _STATE.chat = chat
    if not isinstance(achat, _Unset):
        _STATE.achat = achat
    if not isinstance(messages, _Unset):
        _STATE.messages = messages
    if not isinstance(amessages, _Unset):
        _STATE.amessages = amessages
    if not isinstance(responses, _Unset):
        _STATE.responses = responses
    if not isinstance(aresponses, _Unset):
        _STATE.aresponses = aresponses


def supports_streaming(api: StreamApi, provider: str, transport: StreamTransport = "http") -> bool:
    capability: Final = _STATE.capability or _native_capability()
    if capability is None:
        return False
    try:
        return capability(api, provider, transport)
    except Exception:  # noqa: BLE001  # native extensions can raise dynamically defined exceptions
        return False


def _native_attribute(name: str) -> object | None:
    native: Final = get_native_bridge()
    return None if native is None else getattr(native, name, None)


def _native_capability() -> RustStreamCapability | None:
    capability: Final = _native_attribute("supports_streaming")
    if not callable(capability):
        return None

    def supports(api: str, provider: str, transport: str) -> bool:
        return capability(api, provider, transport) is True

    return supports


def _native_sync_opener(name: str) -> RustStreamOpen | None:
    opener: Final = _native_attribute(name)
    if not callable(opener):
        return None

    def open_stream(
        request: Mapping[str, object],
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
        litellm_metadata: Mapping[str, object] | None,
    ) -> RustEventStream:
        stream: Final = opener(
            request,
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_seconds,
            litellm_metadata,
        )
        if not isinstance(stream, RustEventStream):
            raise TypeError("native stream opener returned an invalid stream")
        return stream

    return open_stream


def _native_async_opener(name: str) -> RustAsyncStreamOpen | None:
    opener: Final = _native_attribute(name)
    if not callable(opener):
        return None

    async def open_stream(
        request: Mapping[str, object],
        provider: str,
        credentials: Mapping[str, str] | None,
        api_base: str | None,
        extra_headers: Mapping[str, str] | None,
        timeout_seconds: float | None,
        litellm_metadata: Mapping[str, object] | None,
    ) -> RustEventStream:
        pending: Final = opener(
            request,
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_seconds,
            litellm_metadata,
        )
        if not isinstance(pending, ObjectAwaitable):
            raise TypeError("native async stream opener returned a non-awaitable")
        stream: Final[object] = await pending
        if not isinstance(stream, RustEventStream):
            raise TypeError("native async stream opener returned an invalid stream")
        return stream

    return open_stream


def _sync_opener(api: StreamApi) -> RustStreamOpen | None:
    match api:
        case "chat_completions":
            return _STATE.chat or _native_sync_opener("chat_completions_stream")
        case "messages":
            return _STATE.messages or _native_sync_opener("messages_stream")
        case "responses":
            return _STATE.responses or _native_sync_opener("responses_stream")


def _async_opener(api: StreamApi) -> RustAsyncStreamOpen | None:
    match api:
        case "chat_completions":
            return _STATE.achat or _native_async_opener("achat_completions_stream")
        case "messages":
            return _STATE.amessages or _native_async_opener("amessages_stream")
        case "responses":
            return _STATE.aresponses or _native_async_opener("aresponses_stream")


def _native_exceptions() -> tuple[type[BaseException], type[BaseException]] | None:
    native: Final = get_native_bridge()
    if native is None:
        return None
    declined: Final = getattr(native, "RustBridgeDeclined", None)
    upstream: Final = getattr(native, "RustUpstreamError", None)
    if not isinstance(declined, type) or not isinstance(upstream, type):
        return None
    return declined, upstream


def _handle_open_error(error: Exception, provider: str) -> None:
    exceptions: Final = _native_exceptions()
    if exceptions is not None and isinstance(error, exceptions[0]):
        return
    _raise_stream_error(error, provider)


def _raise_stream_error(error: Exception, provider: str) -> NoReturn:
    exceptions: Final = _native_exceptions()
    if exceptions is None or not isinstance(error, exceptions[1]):
        raise error
    args: Final[tuple[object, ...]] = error.args
    status_value: Final = args[0] if args else 0
    message_value: Final = args[1] if len(args) > 1 else str(error)
    status: Final = status_value if isinstance(status_value, int) else 0
    message: Final = message_value if isinstance(message_value, str) else str(message_value)
    raise APIError(
        status_code=status or 500,
        message=f"litellm rust typed stream: {message}",
        llm_provider=provider,
        model="",
    ) from error


class TypedEventStreamAdapter:
    def __init__(self, stream: RustEventStream, provider: str) -> None:
        self._stream: Final = stream
        self._provider: Final = provider
        self.metadata: Final = stream.metadata
        self._mode: Literal["sync", "async"] | None = None

    def _claim(self, mode: Literal["sync", "async"]) -> None:
        if self._mode is None:
            self._mode = mode
            return
        if self._mode != mode:
            raise RuntimeError("native stream cannot mix synchronous and asynchronous consumption")

    def __iter__(self) -> Iterator[Event]:
        self._claim("sync")
        return self

    def __next__(self) -> Event:
        self._claim("sync")
        event: Final = self._next_event()
        if event is None:
            raise StopIteration
        return event

    def _next_event(self) -> Event | None:
        try:
            return self._stream.next_event()
        except Exception as error:  # noqa: BLE001  # native stream errors cross a dynamic extension boundary
            _raise_stream_error(error, self._provider)

    def __aiter__(self) -> AsyncIterator[Event]:
        self._claim("async")
        return self

    async def __anext__(self) -> Event:
        self._claim("async")
        try:
            event: Final = await self._stream.anext_event()
        except Exception as error:  # noqa: BLE001  # native stream errors cross a dynamic extension boundary
            _raise_stream_error(error, self._provider)
        if event is None:
            raise StopAsyncIteration
        return event

    def close(self) -> None:
        self._stream.close()

    async def aclose(self) -> None:
        await self._stream.aclose()


class MessagesSseStreamAdapter:
    def __init__(self, events: TypedEventStreamAdapter) -> None:
        self._events: Final = events
        self.metadata: Final = events.metadata

    def __iter__(self) -> Iterator[bytes]:
        return (_event_to_sse(event) for event in self._events)

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for event in self._events:
            yield _event_to_sse(event)

    def close(self) -> None:
        self._events.close()

    async def aclose(self) -> None:
        await self._events.aclose()


class ResponsesSdkEventStreamAdapter:
    def __init__(self, events: TypedEventStreamAdapter) -> None:
        self._events: Final = events
        self.metadata: Final = events.metadata

    def __iter__(self) -> Iterator[ResponsesAPIStreamingResponse]:
        return (_responses_event_to_sdk(event) for event in self._events)

    async def __aiter__(self) -> AsyncIterator[ResponsesAPIStreamingResponse]:
        async for event in self._events:
            yield _responses_event_to_sdk(event)

    def close(self) -> None:
        self._events.close()

    async def aclose(self) -> None:
        await self._events.aclose()


def _responses_event_to_sdk(event: Event) -> ResponsesAPIStreamingResponse:
    from litellm.types.llms.openai import GenericEvent

    event_type: Final = event.get("type")
    model: Final = _responses_event_models().get(event_type) if isinstance(event_type, str) else None
    return (model or GenericEvent).model_validate(event)


@lru_cache(maxsize=1)
def _responses_event_models() -> Mapping[str, type[BaseLiteLLMOpenAIResponseObject]]:
    from litellm.types.llms import openai as openai_types

    return MappingProxyType(
        {
            "response.created": openai_types.ResponseCreatedEvent,
            "response.in_progress": openai_types.ResponseInProgressEvent,
            "response.completed": openai_types.ResponseCompletedEvent,
            "response.failed": openai_types.ResponseFailedEvent,
            "response.incomplete": openai_types.ResponseIncompleteEvent,
            "response.reasoning_summary_part.added": openai_types.ResponsePartAddedEvent,
            "response.reasoning_summary_text.delta": openai_types.ReasoningSummaryTextDeltaEvent,
            "response.reasoning_summary_text.done": openai_types.ReasoningSummaryTextDoneEvent,
            "response.reasoning_summary_part.done": openai_types.ReasoningSummaryPartDoneEvent,
            "response.output_item.added": openai_types.OutputItemAddedEvent,
            "response.output_item.done": openai_types.OutputItemDoneEvent,
            "response.content_part.added": openai_types.ContentPartAddedEvent,
            "response.content_part.done": openai_types.ContentPartDoneEvent,
            "response.output_text.delta": openai_types.OutputTextDeltaEvent,
            "response.output_text.annotation.added": openai_types.OutputTextAnnotationAddedEvent,
            "response.output_text.done": openai_types.OutputTextDoneEvent,
            "response.refusal.delta": openai_types.RefusalDeltaEvent,
            "response.refusal.done": openai_types.RefusalDoneEvent,
            "response.function_call_arguments.delta": openai_types.FunctionCallArgumentsDeltaEvent,
            "response.function_call_arguments.done": openai_types.FunctionCallArgumentsDoneEvent,
            "response.file_search_call.in_progress": openai_types.FileSearchCallInProgressEvent,
            "response.file_search_call.searching": openai_types.FileSearchCallSearchingEvent,
            "response.file_search_call.completed": openai_types.FileSearchCallCompletedEvent,
            "response.web_search_call.in_progress": openai_types.WebSearchCallInProgressEvent,
            "response.web_search_call.searching": openai_types.WebSearchCallSearchingEvent,
            "response.web_search_call.completed": openai_types.WebSearchCallCompletedEvent,
            "response.mcp_list_tools.in_progress": openai_types.MCPListToolsInProgressEvent,
            "response.mcp_list_tools.completed": openai_types.MCPListToolsCompletedEvent,
            "response.mcp_list_tools.failed": openai_types.MCPListToolsFailedEvent,
            "response.mcp_call.in_progress": openai_types.MCPCallInProgressEvent,
            "response.mcp_call_arguments.delta": openai_types.MCPCallArgumentsDeltaEvent,
            "response.mcp_call_arguments.done": openai_types.MCPCallArgumentsDoneEvent,
            "response.mcp_call.completed": openai_types.MCPCallCompletedEvent,
            "response.mcp_call.failed": openai_types.MCPCallFailedEvent,
            "image_generation.partial_image": openai_types.ImageGenerationPartialImageEvent,
            "error": openai_types.ErrorEvent,
        }
    )


def _event_to_sse(event: Event) -> bytes:
    payload: Final = dict(event)  # mutable-ok: JSON requires a concrete dict
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def open_stream(
    *,
    api: StreamApi,
    provider: str,
    request: Mapping[str, object],
    credentials: Mapping[str, str] | None,
    api_base: str | None,
    extra_headers: Mapping[str, str] | None,
    timeout: float | httpx.Timeout | None,
    litellm_metadata: Mapping[str, object] | None,
) -> TypedEventStreamAdapter | None:
    if not supports_streaming(api, provider):
        return None
    opener: Final = _sync_opener(api)
    if opener is None:
        return None
    try:
        stream: Final = opener(
            request,
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_to_seconds(timeout),
            litellm_metadata,
        )
    except Exception as error:  # noqa: BLE001  # native stream errors cross a dynamic extension boundary
        _handle_open_error(error, provider)
        return None
    return TypedEventStreamAdapter(stream, provider)


async def aopen_stream(
    *,
    api: StreamApi,
    provider: str,
    request: Mapping[str, object],
    credentials: Mapping[str, str] | None,
    api_base: str | None,
    extra_headers: Mapping[str, str] | None,
    timeout: float | httpx.Timeout | None,
    litellm_metadata: Mapping[str, object] | None,
) -> TypedEventStreamAdapter | None:
    if not supports_streaming(api, provider):
        return None
    opener: Final = _async_opener(api)
    if opener is None:
        return None
    try:
        stream: Final = await opener(
            request,
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_to_seconds(timeout),
            litellm_metadata,
        )
    except Exception as error:  # noqa: BLE001  # native stream errors cross a dynamic extension boundary
        _handle_open_error(error, provider)
        return None
    return TypedEventStreamAdapter(stream, provider)
