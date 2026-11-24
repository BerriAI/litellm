from __future__ import annotations

import asyncio
import datetime
import json
import traceback
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Mapping
from typing import Final, Protocol, TypeAlias

import anyio
import openai
from openai._legacy_response import LegacyAPIResponse
from openai.types import (
    ImageEditCompletedEvent,
    ImageEditPartialImageEvent,
    ImageGenCompletedEvent,
    ImageGenPartialImageEvent,
    ImagesResponse,
)
from pydantic import BaseModel, TypeAdapter, ValidationError

from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.llms.custom_httpx.sse_lines import aiter_sse_lines, iter_sse_lines
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.utils import ImageObject, ImageResponse

_STREAM_FLAG: Final = TypeAdapter(bool)
_EVENT_PAYLOAD: Final = TypeAdapter(dict[str, object])
_EXTRA_BODY: Final = TypeAdapter(Mapping[str, object])
_COMPLETED_SUFFIX: Final = ".completed"
_EVENT_STREAM_CONTENT_TYPE: Final = "text/event-stream"
_KNOWN_EVENTS: Final[Mapping[str, type[BaseModel]]] = {
    "image_generation.partial_image": ImageGenPartialImageEvent,
    "image_generation.completed": ImageGenCompletedEvent,
    "image_edit.partial_image": ImageEditPartialImageEvent,
    "image_edit.completed": ImageEditCompletedEvent,
}

ImageStreamExceptionMapper: TypeAlias = Callable[[Exception], Exception]


def image_stream_requested(value: object) -> bool:
    try:
        return _STREAM_FLAG.validate_python(value)
    except ValidationError:
        return False


def is_event_stream(headers: Mapping[str, str]) -> bool:
    return _EVENT_STREAM_CONTENT_TYPE in headers.get("content-type", "")


def is_completed_image_event(event: BaseModel) -> bool:
    event_type: Final = getattr(event, "type", None)
    return isinstance(event_type, str) and event_type.endswith(_COMPLETED_SUFFIX)


def image_response_from_completed_event(event: BaseModel) -> ImageResponse:
    payload: Final = event.model_dump(exclude_none=True)
    return ImageResponse(
        created=payload.get("created_at"),
        data=[ImageObject(b64_json=payload.get("b64_json"))],
        **{
            field: payload[field]
            for field in ("usage", "size", "quality", "output_format", "background")
            if field in payload
        },
    )


def image_event_sse_frame(event: BaseModel) -> str:
    return f"event: {getattr(event, 'type', 'message')}\ndata: {event.model_dump_json(exclude_unset=True)}\n\n"


def completed_image_events(response: ImageResponse, *, event_prefix: str) -> tuple[BaseModel, ...]:
    """One completed event per image of a response that was produced unstreamed. The last one carries the usage."""
    images: Final = tuple(response.data or ())
    shared: Final = {
        "type": f"{event_prefix}{_COMPLETED_SUFFIX}",
        "created_at": response.created,
        **response.model_dump(include={"size", "quality", "output_format", "background"}, exclude_none=True),
    }
    usage: Final = response.model_dump(include={"usage"}, exclude_none=True)
    return tuple(
        openai.BaseModel.model_validate(
            {
                **shared,
                **(usage if position == len(images) - 1 else {}),
                **image.model_dump(include={"b64_json", "url"}, exclude_none=True),
            }
        )
        for position, image in enumerate(images)
    ) or (openai.BaseModel.model_validate({**shared, **usage}),)


def _event_from_sse_data(line: str) -> BaseModel | None:
    if not line.startswith("data:"):
        return None
    data: Final = line[len("data:") :].strip()
    if not data or data == "[DONE]":
        return None
    payload: Final = _EVENT_PAYLOAD.validate_python(json.loads(data))
    error: Final = payload.get("error")
    if error is not None or payload.get("type") == "error":
        raise OpenAIError(status_code=500, message=json.dumps(payload if error is None else error))
    event_type: Final = payload.get("type")
    known_event: Final = _KNOWN_EVENTS.get(event_type) if isinstance(event_type, str) else None
    if known_event is None:
        return openai.BaseModel.model_validate(payload)
    try:
        return known_event.model_validate(payload)
    except ValidationError:
        return openai.BaseModel.model_validate(payload)


def image_events_from_sse(text_chunks: Iterator[str]) -> Iterator[BaseModel]:
    for line in iter_sse_lines(text_chunks):
        event = _event_from_sse_data(line)
        if event is not None:
            yield event


async def aimage_events_from_sse(text_chunks: AsyncIterator[str]) -> AsyncIterator[BaseModel]:
    async for line in aiter_sse_lines(text_chunks):
        event = _event_from_sse_data(line)
        if event is not None:
            yield event


class ImageStreamLogging(Protocol):
    start_time: datetime.datetime

    def success_handler(
        self, result: ImageResponse, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None: ...

    async def async_success_handler(
        self, result: ImageResponse, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None: ...

    def handle_sync_success_callbacks_for_async_calls(
        self, result: ImageResponse, start_time: datetime.datetime, end_time: datetime.datetime
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


def _now() -> datetime.datetime:
    return datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract


def _ended_before_completion() -> OpenAIError:
    return OpenAIError(status_code=502, message="The image stream ended before its completed event")


class ImageStream:
    """The provider's image events. The call is logged, and billed, when the completed event arrives.

    Closing early reads the rest of the stream first, since the provider bills the image either way."""

    def __init__(
        self,
        events: Iterator[BaseModel],
        close: Callable[[], None],
        headers: Mapping[str, str],
        logging_obj: ImageStreamLogging,
        map_exception: ImageStreamExceptionMapper,
    ) -> None:
        self._events: Final = events
        self._close: Final = close
        self._hidden_params: dict[str, object] = {  # mutable-ok: ResponseMetadata fills in the call's metadata
            "additional_headers": process_response_headers(dict(headers))
        }
        self._logging_obj: Final = logging_obj
        self._map_exception: Final = map_exception
        self._settled = False

    @property
    def logging_obj(self) -> ImageStreamLogging:
        return self._logging_obj

    def __iter__(self) -> ImageStream:
        return self

    def __next__(self) -> BaseModel:
        try:
            event: Final = next(self._events)
        except StopIteration:
            if self._settled:
                raise
            cut: Final = self._map_exception(_ended_before_completion())
            self._fail(cut)
            raise cut from None
        except Exception as exception:  # noqa: BLE001  # stream boundary: every provider failure is mapped and logged
            mapped: Final = self._map_exception(exception)
            self._fail(mapped)
            raise mapped from exception
        if is_completed_image_event(event):
            self._succeed(event)
        return event

    def _succeed(self, event: BaseModel) -> None:
        if self._settled:
            return
        self._settled = True
        self._logging_obj.success_handler(
            image_response_from_completed_event(event), self._logging_obj.start_time, _now()
        )

    def _fail(self, exception: Exception) -> None:
        if self._settled:
            return
        self._settled = True
        self._logging_obj.failure_handler(exception, traceback.format_exc(), self._logging_obj.start_time, _now())

    def close(self) -> None:
        try:
            while not self._settled:
                next(self)
        except Exception:  # noqa: BLE001, S110  # __next__ already logged why the rest could not be read
            pass
        finally:
            self._close()


class AsyncImageStream:
    """Async twin of `ImageStream`."""

    def __init__(
        self,
        events: AsyncIterator[BaseModel],
        close: Callable[[], Awaitable[None]],
        headers: Mapping[str, str],
        logging_obj: ImageStreamLogging,
        map_exception: ImageStreamExceptionMapper,
    ) -> None:
        self._events: Final = events
        self._close: Final = close
        self._hidden_params: dict[str, object] = {  # mutable-ok: ResponseMetadata fills in the call's metadata
            "additional_headers": process_response_headers(dict(headers))
        }
        self._logging_obj: Final = logging_obj
        self._map_exception: Final = map_exception
        self._settled = False
        self._pending: asyncio.Future[BaseModel] | None = None

    @property
    def logging_obj(self) -> ImageStreamLogging:
        return self._logging_obj

    def __aiter__(self) -> AsyncImageStream:
        return self

    async def __anext__(self) -> BaseModel:
        try:
            event: Final = await self._read()
        except StopAsyncIteration:
            if self._settled:
                raise
            cut: Final = self._map_exception(_ended_before_completion())
            await self._fail(cut)
            raise cut from None
        except Exception as exception:  # noqa: BLE001  # stream boundary: every provider failure is mapped and logged
            mapped: Final = self._map_exception(exception)
            await self._fail(mapped)
            raise mapped from exception
        if is_completed_image_event(event):
            await self._succeed(event)
        return event

    async def _read(self) -> BaseModel:
        """A caller cancelled mid-read leaves the read running, so `aclose` can still collect the billed image."""
        pending: Final = self._pending or asyncio.ensure_future(self._events.__anext__())
        self._pending = pending
        try:
            return await asyncio.shield(pending)
        finally:
            if pending.done():
                self._pending = None

    async def _succeed(self, event: BaseModel) -> None:
        if self._settled:
            return
        self._settled = True
        response: Final = image_response_from_completed_event(event)
        end_time: Final = _now()
        with anyio.CancelScope(shield=True):
            self._logging_obj.handle_sync_success_callbacks_for_async_calls(
                result=response, start_time=self._logging_obj.start_time, end_time=end_time
            )
            await self._logging_obj.async_success_handler(
                result=response, start_time=self._logging_obj.start_time, end_time=end_time
            )

    async def _fail(self, exception: Exception) -> None:
        if self._settled:
            return
        self._settled = True
        end_time: Final = _now()
        with anyio.CancelScope(shield=True):
            self._logging_obj.failure_handler(
                exception, traceback.format_exc(), self._logging_obj.start_time, end_time, for_async_request=True
            )
            await self._logging_obj.async_failure_handler(
                exception, traceback.format_exc(), self._logging_obj.start_time, end_time
            )

    async def aclose(self) -> None:
        with anyio.CancelScope(shield=True):
            try:
                while not self._settled:
                    await self.__anext__()
            except Exception:  # noqa: BLE001, S110  # __anext__ already logged why the rest could not be read
                pass
            finally:
                await self._close()


def image_generation_stream_requested(stream: object, extra_body: object) -> bool:
    try:
        body_stream: Final = _EXTRA_BODY.validate_python(extra_body).get("stream")
    except ValidationError:
        return image_stream_requested(stream)
    return image_stream_requested(stream) or image_stream_requested(body_stream)


def openai_image_generation_stream(
    raw_response: LegacyAPIResponse[ImagesResponse],
    logging_obj: ImageStreamLogging,
    map_exception: ImageStreamExceptionMapper,
) -> ImageStream | None:
    if not is_event_stream(raw_response.headers):
        return None
    return ImageStream(
        events=image_events_from_sse(raw_response.http_response.iter_text()),
        close=raw_response.http_response.close,
        headers=raw_response.headers,
        logging_obj=logging_obj,
        map_exception=map_exception,
    )


def openai_image_generation_async_stream(
    raw_response: LegacyAPIResponse[ImagesResponse],
    logging_obj: ImageStreamLogging,
    map_exception: ImageStreamExceptionMapper,
) -> AsyncImageStream | None:
    if not is_event_stream(raw_response.headers):
        return None
    return AsyncImageStream(
        events=aimage_events_from_sse(raw_response.http_response.aiter_text()),
        close=raw_response.http_response.aclose,
        headers=raw_response.headers,
        logging_obj=logging_obj,
        map_exception=map_exception,
    )
