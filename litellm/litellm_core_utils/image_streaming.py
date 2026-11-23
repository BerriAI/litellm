from __future__ import annotations

import datetime
import traceback
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Final, Protocol, overload

import anyio
import httpx
from openai import APIError, AsyncStream, Stream
from openai.types.image_edit_stream_event import (
    ImageEditCompletedEvent,
    ImageEditPartialImageEvent,
    ImageEditStreamEvent,
)
from openai.types.image_gen_stream_event import (
    ImageGenCompletedEvent,
    ImageGenPartialImageEvent,
    ImageGenStreamEvent,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.litellm_core_utils.audio_utils.transcription_streaming import (
    TranscriptionStreamExceptionMapper,
    litellm_exception_mapper,
)
from litellm.litellm_core_utils.core_helpers import (
    process_response_headers,  # pyright: ignore[reportUnknownVariableType]  # legacy helper accepts unparameterized dict
)
from litellm.types.utils import ImageObject, ImageResponse, ImageUsage

__all__ = [
    "AsyncImageSSEStream",
    "ImageStreamEvent",
    "ImageStreamExceptionMapper",
    "ImageStreamLogging",
    "LoggingAsyncImageStream",
    "LoggingImageStream",
    "SyncImageSSEStream",
    "format_image_sse_event",
    "image_stream_flag",
    "image_stream_requested",
    "is_image_stream_response",
    "litellm_exception_mapper",
    "wrap_image_stream",
]

ImageStreamEvent = ImageGenStreamEvent | ImageEditStreamEvent
ImageStreamExceptionMapper = TranscriptionStreamExceptionMapper

_STREAM_FLAG: Final = TypeAdapter(bool)
_JSON_OBJECT_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])
_STRING_MAPPING_ADAPTER: Final[TypeAdapter[dict[str, object]]] = TypeAdapter(dict[str, object])
_IMAGE_STREAM_EVENT_ADAPTER: Final[
    TypeAdapter[
        ImageGenPartialImageEvent | ImageGenCompletedEvent | ImageEditPartialImageEvent | ImageEditCompletedEvent
    ]
] = TypeAdapter(
    ImageGenPartialImageEvent | ImageGenCompletedEvent | ImageEditPartialImageEvent | ImageEditCompletedEvent
)


def image_stream_flag(value: object) -> bool | None:
    """Parse boolean or string-encoded stream flags into bool or None."""
    if not isinstance(value, (bool, str, int)):
        return None
    try:
        return _STREAM_FLAG.validate_python(value)
    except ValidationError:
        return None


def image_stream_requested(value: object) -> bool:
    """Return True only when an image request explicitly asks for streaming."""
    return image_stream_flag(value) is True


class ImageStreamLogging(Protocol):
    def success_handler(
        self,
        result: ImageResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    async def async_success_handler(
        self,
        result: ImageResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    def handle_sync_success_callbacks_for_async_calls(
        self,
        result: ImageResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    def failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
        *,
        for_async_request: bool = False,
    ) -> None: ...

    async def async_failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...


class _SSEFrameAccumulator:
    """Incrementally collect SSE field lines into `(event_name, data)` frames."""

    def __init__(self) -> None:
        self._event: str | None = None
        self._data_lines: list[str] = []

    def feed_line(self, line: str) -> tuple[str | None, str] | None:
        stripped: Final = line.removesuffix("\r")
        if stripped == "":
            return self.flush()
        if stripped.startswith(":"):
            return None
        field, _, rest = stripped.partition(":")
        value: Final = rest.removeprefix(" ")
        if field == "event":
            self._event = value
        elif field == "data":
            self._data_lines.append(value)
        return None

    def flush(self) -> tuple[str | None, str] | None:
        event_name: Final = self._event
        self._event = None
        if not self._data_lines:
            return None
        raw_data: Final = "\n".join(self._data_lines)
        self._data_lines.clear()
        return (event_name, raw_data)


def _sse_error_message(error_value: object, fallback: str) -> str:
    if isinstance(error_value, Mapping):
        try:
            error_mapping: Final = _STRING_MAPPING_ADAPTER.validate_python(error_value)
        except ValidationError:
            return fallback
        message: Final[object] = error_mapping.get("message")
        if isinstance(message, str) and message:
            return message
    if isinstance(error_value, str) and error_value:
        return error_value
    return fallback


def _parse_image_sse_frame(
    event_name: str | None,
    raw_data: str,
    request: httpx.Request | None = None,
) -> ImageStreamEvent | None:
    if raw_data.strip() == "[DONE]":
        return None
    dummy_request: Final = request or httpx.Request("POST", "https://api.openai.com/v1/images/edits")
    try:
        payload: dict[str, object] = _JSON_OBJECT_ADAPTER.validate_json(raw_data)
    except ValidationError as exc:
        raise APIError(message=raw_data, request=dummy_request, body=None) from exc

    if event_name == "error" or "error" in payload:
        raise APIError(
            message=_sse_error_message(payload.get("error"), raw_data),
            request=dummy_request,
            body=payload.get("error", payload),
        )

    if "type" not in payload and event_name:
        payload = {**payload, "type": event_name}
    return _IMAGE_STREAM_EVENT_ADAPTER.validate_python(payload)


def _processed_response_headers(headers: httpx.Headers) -> dict[str, object]:
    return _STRING_MAPPING_ADAPTER.validate_python(process_response_headers(headers))


def _initial_hidden_params(stream: object, headers: httpx.Headers) -> dict[str, object]:
    raw_hidden: Final[object] = getattr(stream, "_hidden_params", None)
    base: dict[str, object] = {}
    if isinstance(raw_hidden, Mapping):
        try:
            base = _STRING_MAPPING_ADAPTER.validate_python(raw_hidden)
        except ValidationError:
            base = {}
    base["additional_headers"] = _processed_response_headers(headers)
    return base


class SyncImageSSEStream(Stream[ImageStreamEvent]):
    """Wrap a streaming `httpx.Response` as a typed `openai.Stream` for image SSE frames."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self._hidden_params: dict[str, object] = {
            "additional_headers": _processed_response_headers(response.headers),
        }
        self._iterator = self._iter_image_events()

    @property
    def hidden_params(self) -> dict[str, object]:
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, object]) -> None:
        self._hidden_params = hidden_params

    def _iter_image_events(self) -> Iterator[ImageStreamEvent]:
        accumulator: Final = _SSEFrameAccumulator()
        request: Final = getattr(self.response, "_request", None)
        try:
            for line in self.response.iter_lines():
                frame = accumulator.feed_line(line)
                if frame is None:
                    continue
                event = _parse_image_sse_frame(*frame, request=request)
                if event is not None:
                    yield event
            trailing: Final = accumulator.flush()
            if trailing is not None:
                event = _parse_image_sse_frame(*trailing, request=request)
                if event is not None:
                    yield event
        finally:
            self.response.close()


class AsyncImageSSEStream(AsyncStream[ImageStreamEvent]):
    """Wrap an async streaming `httpx.Response` as a typed `openai.AsyncStream` for image SSE frames."""

    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self._hidden_params: dict[str, object] = {
            "additional_headers": _processed_response_headers(response.headers),
        }
        self._iterator = self._iter_image_events()

    @property
    def hidden_params(self) -> dict[str, object]:
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, object]) -> None:
        self._hidden_params = hidden_params

    async def _iter_image_events(self) -> AsyncIterator[ImageStreamEvent]:
        accumulator: Final = _SSEFrameAccumulator()
        request: Final = getattr(self.response, "_request", None)
        try:
            async for line in self.response.aiter_lines():
                frame = accumulator.feed_line(line)
                if frame is None:
                    continue
                event = _parse_image_sse_frame(*frame, request=request)
                if event is not None:
                    yield event
            trailing: Final = accumulator.flush()
            if trailing is not None:
                event = _parse_image_sse_frame(*trailing, request=request)
                if event is not None:
                    yield event
        finally:
            await self.response.aclose()


class _ImageEventCollector:
    """Accumulate streamed partial/completed image events into a single `ImageResponse`."""

    def __init__(self) -> None:
        self.latest_partial_event: ImageGenPartialImageEvent | ImageEditPartialImageEvent | None = None
        self.completed_events: list[ImageGenCompletedEvent | ImageEditCompletedEvent] = []

    def add(self, event: ImageStreamEvent) -> None:
        if isinstance(event, (ImageGenCompletedEvent, ImageEditCompletedEvent)):
            self.completed_events.append(event)
        else:
            self.latest_partial_event = event

    def response(self, hidden_params: Mapping[str, object]) -> ImageResponse:
        base_hidden: Final = {k: v for k, v in hidden_params.items() if k != "response_cost"}
        if self.completed_events:
            last: Final = self.completed_events[-1]
            raw_usage: Final[object] = getattr(last, "usage", None)
            usage: Final = (
                ImageUsage.model_validate(raw_usage.model_dump()) if isinstance(raw_usage, BaseModel) else None
            )
            images: Final[list[ImageObject]] = [
                ImageObject(b64_json=event.b64_json) for event in self.completed_events if event.b64_json
            ]
            assembled: Final = ImageResponse(
                created=last.created_at,
                data=images,
                background=last.background,
                output_format=last.output_format,
                quality=last.quality,
                size=last.size,
                usage=usage,
            )
            assembled.hidden_params = dict(base_hidden)
            return assembled

        partial: Final = self.latest_partial_event
        if partial is not None:
            partial_images: Final[list[ImageObject]] = (
                [ImageObject(b64_json=partial.b64_json)] if partial.b64_json else []
            )
            fallback: Final = ImageResponse(
                created=partial.created_at,
                data=partial_images,
                background=partial.background,
                output_format=partial.output_format,
                quality=partial.quality,
                size=partial.size,
            )
            fallback.hidden_params = dict(base_hidden)
            return fallback

        empty: Final = ImageResponse(
            created=int(datetime.datetime.now(datetime.timezone.utc).timestamp()),
            data=[],
        )
        empty.hidden_params = dict(base_hidden)
        return empty


def _sync_cost_to_stream_hidden_params(stream_hidden_params: dict[str, object], response: ImageResponse) -> None:
    response_cost: Final[object] = response.hidden_params.get("response_cost")
    if response_cost is not None:
        stream_hidden_params["response_cost"] = response_cost


class LoggingImageStream(Stream[ImageStreamEvent]):
    """Synchronous image SSE stream wrapper that records cost and logging callbacks on finish."""

    def __init__(
        self,
        stream: Stream[ImageStreamEvent],
        logging_obj: ImageStreamLogging,
        start_time: datetime.datetime,
        map_exception: ImageStreamExceptionMapper,
    ) -> None:
        self.response = stream.response
        self._hidden_params: dict[str, object] = _initial_hidden_params(stream, stream.response.headers)
        self._logging_obj = logging_obj
        self._start_time = start_time
        self._map_exception = map_exception
        self._collector = _ImageEventCollector()
        self._finalized = False
        self._failed = False
        source_iterator: Final = iter(stream)
        self._iterator = self._logging_iterator(source_iterator)

    @property
    def hidden_params(self) -> dict[str, object]:
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, object]) -> None:
        self._hidden_params = hidden_params

    def _logging_iterator(self, source_iterator: Iterator[ImageStreamEvent]) -> Iterator[ImageStreamEvent]:
        try:
            for event in source_iterator:
                self._collector.add(event)
                yield event
        except Exception as exception:
            self._failed = True
            mapped_exception: Final = self._map_exception(exception)
            end_time: Final = datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract
            self._logging_obj.failure_handler(mapped_exception, traceback.format_exc(), self._start_time, end_time)
            raise mapped_exception from exception
        finally:
            self._finalize()

    def _finalize(self) -> None:
        if self._finalized or self._failed:
            return
        self._finalized = True
        assembled: Final = self._collector.response(self._hidden_params)
        self._logging_obj.success_handler(
            assembled,
            self._start_time,
            datetime.datetime.now(),  # noqa: DTZ005  # callback timestamps use the legacy naive contract
        )
        _sync_cost_to_stream_hidden_params(self._hidden_params, assembled)

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._finalize()

    @property
    def logging_obj(self) -> ImageStreamLogging:
        return self._logging_obj


class LoggingAsyncImageStream(AsyncStream[ImageStreamEvent]):
    """Asynchronous image SSE stream wrapper that records cost and logging callbacks on finish."""

    def __init__(
        self,
        stream: AsyncStream[ImageStreamEvent],
        logging_obj: ImageStreamLogging,
        start_time: datetime.datetime,
        map_exception: ImageStreamExceptionMapper,
    ) -> None:
        self.response = stream.response
        self._hidden_params: dict[str, object] = _initial_hidden_params(stream, stream.response.headers)
        self._logging_obj = logging_obj
        self._start_time = start_time
        self._map_exception = map_exception
        self._collector = _ImageEventCollector()
        self._finalized = False
        self._failed = False
        source_iterator: Final = stream.__aiter__()
        self._iterator = self._logging_iterator(source_iterator)

    @property
    def hidden_params(self) -> dict[str, object]:
        return self._hidden_params

    @hidden_params.setter
    def hidden_params(self, hidden_params: dict[str, object]) -> None:
        self._hidden_params = hidden_params

    async def _logging_iterator(
        self, source_iterator: AsyncIterator[ImageStreamEvent]
    ) -> AsyncIterator[ImageStreamEvent]:
        try:
            async for event in source_iterator:
                self._collector.add(event)
                yield event
        except Exception as exception:
            self._failed = True
            mapped_exception: Final = self._map_exception(exception)
            end_time: Final = datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract
            self._logging_obj.failure_handler(
                mapped_exception, traceback.format_exc(), self._start_time, end_time, for_async_request=True
            )
            await self._logging_obj.async_failure_handler(
                mapped_exception,
                traceback.format_exc(),
                self._start_time,
                end_time,
            )
            raise mapped_exception from exception
        finally:
            await self._finalize()

    async def _finalize(self) -> None:
        if self._finalized or self._failed:
            return
        self._finalized = True
        assembled: Final = self._collector.response(self._hidden_params)
        end_time: Final = datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract
        self._logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=assembled,
            start_time=self._start_time,
            end_time=end_time,
        )
        with anyio.CancelScope(shield=True):
            await self._logging_obj.async_success_handler(
                result=assembled,
                start_time=self._start_time,
                end_time=end_time,
            )
        _sync_cost_to_stream_hidden_params(self._hidden_params, assembled)

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await self._finalize()

    async def aclose(self) -> None:
        await self.close()

    @property
    def logging_obj(self) -> ImageStreamLogging:
        return self._logging_obj


@overload
def wrap_image_stream(
    stream: Stream[ImageStreamEvent],
    logging_obj: ImageStreamLogging,
    start_time: datetime.datetime,
    map_exception: ImageStreamExceptionMapper,
) -> LoggingImageStream: ...


@overload
def wrap_image_stream(
    stream: AsyncStream[ImageStreamEvent],
    logging_obj: ImageStreamLogging,
    start_time: datetime.datetime,
    map_exception: ImageStreamExceptionMapper,
) -> LoggingAsyncImageStream: ...


def wrap_image_stream(
    stream: Stream[ImageStreamEvent] | AsyncStream[ImageStreamEvent],
    logging_obj: ImageStreamLogging,
    start_time: datetime.datetime,
    map_exception: ImageStreamExceptionMapper,
) -> LoggingImageStream | LoggingAsyncImageStream:
    """Wrap an OpenAI image stream so LiteLLM callbacks and cost tracking run on completion."""
    if isinstance(stream, AsyncStream):
        return LoggingAsyncImageStream(stream, logging_obj, start_time, map_exception)
    return LoggingImageStream(stream, logging_obj, start_time, map_exception)


def is_image_stream_response(response: object) -> bool:
    """Return True when response is a LiteLLM image generation or edit stream."""
    return isinstance(
        response,
        (SyncImageSSEStream, AsyncImageSSEStream, LoggingImageStream, LoggingAsyncImageStream),
    )


def format_image_sse_event(chunk: object) -> str | None:
    """Serialize an OpenAI image stream event using OpenAI's named SSE event framing."""
    if isinstance(
        chunk,
        (
            ImageGenPartialImageEvent,
            ImageGenCompletedEvent,
            ImageEditPartialImageEvent,
            ImageEditCompletedEvent,
        ),
    ):
        return f"event: {chunk.type}\ndata: {chunk.model_dump_json(exclude_unset=True)}\n\n"
    return None
