import datetime
import json
import traceback
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import TYPE_CHECKING, Final, Optional, Union, cast

import anyio
import httpx
from pydantic import BaseModel

from litellm.types.utils import ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def image_stream_requested(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1")
    return False


def _event_payload(chunk: object) -> Mapping[str, object] | None:
    if isinstance(chunk, BaseModel):
        return chunk.model_dump()
    if isinstance(chunk, Mapping):
        return chunk
    return None


def _image_response_from_completed(payload: Mapping[str, object]) -> ImageResponse | None:
    b64_json: Final = payload.get("b64_json")
    if not isinstance(b64_json, str):
        return None
    created: Final = payload.get("created_at")
    usage: Final = payload.get("usage")
    size: Final = payload.get("size")
    quality: Final = payload.get("quality")
    return ImageResponse(
        created=created if isinstance(created, int) else 0,
        data=[{"b64_json": b64_json}],
        usage=usage if isinstance(usage, Mapping) else None,
        size=size if isinstance(size, str) else None,
        quality=quality if isinstance(quality, str) else None,
    )


async def image_event_stream(response: httpx.Response) -> AsyncIterator[dict[str, object]]:
    async for line in response.aiter_lines():
        if not line.startswith("data:"):
            continue
        data: Final = line[len("data:") :].strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload: Final = json.loads(data)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            yield payload


class ImageStreamingResponse:
    def __init__(
        self,
        stream_iterator: Union[AsyncIterator[object], Iterator[object]],
        logging_obj: Optional["LiteLLMLoggingObj"],
        start_time: datetime.datetime | None = None,
    ) -> None:
        self.stream_iterator = stream_iterator
        self.logging_obj = logging_obj
        self.standard_logging_object: object | None = None
        self._hidden_params: dict[str, object] = {}
        self._start_time: Final = (
            start_time
            or (
                logging_obj.start_time
                if logging_obj is not None and getattr(logging_obj, "start_time", None)
                else datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract
            )
        )
        self._logging_completed = False
        self._completed_payload: Mapping[str, object] | None = None

    def __aiter__(self) -> "ImageStreamingResponse":
        return self

    async def __anext__(self) -> object:
        try:
            chunk: Final = await cast(AsyncIterator[object], self.stream_iterator).__anext__()
        except StopAsyncIteration:
            await self._log_success_async()
            raise
        except Exception as e:
            await self._log_failure_async(e)
            raise
        self._capture(chunk)
        return chunk

    def __iter__(self) -> "ImageStreamingResponse":
        return self

    def __next__(self) -> object:
        try:
            chunk: Final = next(cast(Iterator[object], self.stream_iterator))
        except StopIteration:
            self._log_success_sync()
            raise
        except Exception as e:
            self._log_failure_sync(e)
            raise
        self._capture(chunk)
        return chunk

    async def aclose(self) -> None:
        self._logging_completed = True
        if hasattr(self.stream_iterator, "aclose"):
            await cast(AsyncIterator[object], self.stream_iterator).aclose()

    def close(self) -> None:
        self._logging_completed = True
        if hasattr(self.stream_iterator, "close"):
            cast(Iterator[object], self.stream_iterator).close()

    def _capture(self, chunk: object) -> None:
        payload: Final = _event_payload(chunk)
        if payload is None:
            return
        event_type: Final = payload.get("type")
        if isinstance(event_type, str) and event_type.endswith(".completed"):
            self._completed_payload = payload

    def _billed_response(self) -> ImageResponse | None:
        if self._completed_payload is None:
            return None
        return _image_response_from_completed(self._completed_payload)

    async def _log_success_async(self) -> None:
        if self._logging_completed or self.logging_obj is None:
            return
        self._logging_completed = True
        result: Final = self._billed_response()
        if result is None:
            return
        end_time: Final = datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract
        with anyio.CancelScope(shield=True):
            await self.logging_obj.async_success_handler(
                result=result,
                start_time=self._start_time,
                end_time=end_time,
            )
        self.logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=result,
            start_time=self._start_time,
            end_time=end_time,
        )

    def _log_success_sync(self) -> None:
        if self._logging_completed or self.logging_obj is None:
            return
        self._logging_completed = True
        result: Final = self._billed_response()
        if result is None:
            return
        self.logging_obj.success_handler(
            result=result,
            start_time=self._start_time,
            end_time=datetime.datetime.now(),  # noqa: DTZ005  # callback timestamps use the legacy naive contract
        )

    async def _log_failure_async(self, error: Exception) -> None:
        if self._logging_completed or self.logging_obj is None:
            return
        self._logging_completed = True
        end_time: Final = datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract
        traceback_str: Final = traceback.format_exc()
        self.logging_obj.failure_handler(error, traceback_str, self._start_time, end_time)
        await self.logging_obj.async_failure_handler(error, traceback_str, self._start_time, end_time)

    def _log_failure_sync(self, error: Exception) -> None:
        if self._logging_completed or self.logging_obj is None:
            return
        self._logging_completed = True
        self.logging_obj.failure_handler(
            error,
            traceback.format_exc(),
            self._start_time,
            datetime.datetime.now(),  # noqa: DTZ005  # callback timestamps use the legacy naive contract
        )
