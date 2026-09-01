from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator, Mapping
from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, TypeAlias, runtime_checkable

import httpx

from litellm.rust_bridge.runtime import (
    UNSET,
    BridgeErrorContext,
    CoreEngine,
    ExecutionResult,
    FallbackMode,
    NativeBinding,
    Unset,
    acall,
    ainvoke,
    async_none,
    call,
    invoke,
)
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
    ) -> RustEventStream: ...


_CHAT: Final = NativeBinding[RustStreamOpen]("chat_completions_stream")
_ACHAT: Final = NativeBinding[RustAsyncStreamOpen]("achat_completions_stream")
_MESSAGES: Final = NativeBinding[RustStreamOpen]("messages_stream")
_AMESSAGES: Final = NativeBinding[RustAsyncStreamOpen]("amessages_stream")
_RESPONSES: Final = NativeBinding[RustStreamOpen]("responses_stream")
_ARESPONSES: Final = NativeBinding[RustAsyncStreamOpen]("aresponses_stream")


def set_rust_streaming(
    *,
    chat: RustStreamOpen | None | Unset = UNSET,
    achat: RustAsyncStreamOpen | None | Unset = UNSET,
    messages: RustStreamOpen | None | Unset = UNSET,
    amessages: RustAsyncStreamOpen | None | Unset = UNSET,
    responses: RustStreamOpen | None | Unset = UNSET,
    aresponses: RustAsyncStreamOpen | None | Unset = UNSET,
) -> None:
    _CHAT.update(chat)
    _ACHAT.update(achat)
    _MESSAGES.update(messages)
    _AMESSAGES.update(amessages)
    _RESPONSES.update(responses)
    _ARESPONSES.update(aresponses)


def _sync_opener(api: StreamApi) -> RustStreamOpen | None:
    match api:
        case "chat_completions":
            return _CHAT.load()
        case "messages":
            return _MESSAGES.load()
        case "responses":
            return _RESPONSES.load()


def _async_opener(api: StreamApi) -> RustAsyncStreamOpen | None:
    match api:
        case "chat_completions":
            return _ACHAT.load()
        case "messages":
            return _AMESSAGES.load()
        case "responses":
            return _ARESPONSES.load()


class TypedEventStreamAdapter:
    def __init__(self, stream: RustEventStream, provider: str) -> None:
        self._stream: Final = stream
        self._provider: Final = provider
        self.metadata: Final = stream.metadata
        self.core_engine: Final = CoreEngine.RUST
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
        return call(
            self._stream.next_event,
            BridgeErrorContext(route="typed stream", provider=self._provider, model=""),
        )

    def __aiter__(self) -> AsyncIterator[Event]:
        self._claim("async")
        return self

    async def __anext__(self) -> Event:
        self._claim("async")
        event: Final = await acall(
            self._stream.anext_event,
            BridgeErrorContext(route="typed stream", provider=self._provider, model=""),
        )
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
) -> ExecutionResult[TypedEventStreamAdapter | None]:
    opener: Final = _sync_opener(api)
    native_call: Final = (
        None
        if opener is None
        else lambda: opener(
            request,
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_to_seconds(timeout),
        )
    )
    return invoke(
        native_call=native_call,
        fallback=lambda: None,
        adapt=lambda stream: TypedEventStreamAdapter(stream, provider),
        mode=FallbackMode.PYTHON,
        context=BridgeErrorContext(route=f"{api} stream", provider=provider, model=""),
    )


async def aopen_stream(
    *,
    api: StreamApi,
    provider: str,
    request: Mapping[str, object],
    credentials: Mapping[str, str] | None,
    api_base: str | None,
    extra_headers: Mapping[str, str] | None,
    timeout: float | httpx.Timeout | None,
) -> ExecutionResult[TypedEventStreamAdapter | None]:
    opener: Final = _async_opener(api)
    native_call: Final = (
        None
        if opener is None
        else lambda: opener(
            request,
            provider,
            credentials,
            api_base,
            extra_headers,
            timeout_to_seconds(timeout),
        )
    )
    return await ainvoke(
        native_call=native_call,
        fallback=async_none,
        adapt=lambda stream: TypedEventStreamAdapter(stream, provider),
        mode=FallbackMode.PYTHON,
        context=BridgeErrorContext(route=f"{api} stream", provider=provider, model=""),
    )
