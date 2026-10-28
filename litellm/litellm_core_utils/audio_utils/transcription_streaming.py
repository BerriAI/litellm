from __future__ import annotations

import datetime
import json
import traceback
from collections.abc import AsyncIterator, Iterator
from typing import Final, Protocol

import anyio
from openai import AsyncStream, Stream
from openai.types.audio import (
    TranscriptionStreamEvent,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
)

from litellm.litellm_core_utils.core_helpers import process_response_headers
from litellm.types.utils import TranscriptionResponse

TRANSCRIPTION_RAN_UNSTREAMED: Final = "transcription_ran_unstreamed"


def transcription_stream_requested(value: object) -> bool:
    return str(value).lower() in ("1", "true")


TRANSCRIPTION_STREAM_DONE_FRAME: Final = "data: [DONE]\n\n"


def guarded_transcription_events(response: TranscriptionResponse) -> tuple[str, ...]:
    text: Final = response.text or ""
    usage: Final = response.usage
    languages: Final = TranscriptionResponse.detected_languages(getattr(response, "languages", None))
    done_event: Final = {
        "type": "transcript.text.done",
        "text": text,
        **({} if usage is None else {"usage": usage.model_dump(exclude_none=True)}),
        **({} if languages is None else {"languages": [language.model_dump() for language in languages]}),
    }
    return (
        *(
            f"data: {json.dumps(event, separators=(',', ':'))}\n\n"
            for event in ({"type": "transcript.text.delta", "delta": text}, done_event)
        ),
        TRANSCRIPTION_STREAM_DONE_FRAME,
    )


class TranscriptionStreamLogging(Protocol):
    def success_handler(
        self,
        result: TranscriptionResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    async def async_success_handler(
        self,
        result: TranscriptionResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    def handle_sync_success_callbacks_for_async_calls(
        self,
        result: TranscriptionResponse,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    def failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...

    async def async_failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None: ...


class _TranscriptionEventCollector:
    def __init__(self, duration: float | None) -> None:
        self.duration = duration
        self.text_deltas: list[str] = []  # mutable-ok: streaming deltas accumulate until the terminal event
        self.done_event: TranscriptionTextDoneEvent | None = None

    def add(self, event: TranscriptionStreamEvent) -> None:
        if isinstance(event, TranscriptionTextDeltaEvent):
            self.text_deltas.append(event.delta)
        elif isinstance(event, TranscriptionTextDoneEvent):
            self.done_event = event

    def response(self) -> TranscriptionResponse:
        done_event: Final = self.done_event
        response: Final = TranscriptionResponse(
            text=done_event.text if done_event is not None else "".join(self.text_deltas),
            usage=done_event.usage.model_dump() if done_event is not None and done_event.usage is not None else None,
            languages=getattr(done_event, "languages", None) if done_event is not None else None,
        )
        if self.duration is not None:
            response.set_audio_transcription_duration(self.duration)
        return response


class LoggingTranscriptionStream(Stream[TranscriptionStreamEvent]):
    def __init__(
        self,
        stream: Stream[TranscriptionStreamEvent],
        logging_obj: TranscriptionStreamLogging,
        start_time: datetime.datetime,
    ) -> None:
        self.__dict__.update(stream.__dict__)
        self._hidden_params: dict[str, object] = {  # mutable-ok: ResponseMetadata fills in the call's metadata
            "additional_headers": process_response_headers(stream.response.headers)
        }
        self._logging_obj = logging_obj
        self._start_time = start_time
        self._collector = _TranscriptionEventCollector(getattr(stream, "_litellm_audio_duration", None))
        self._finalized = False
        self._failed = False
        source_iterator: Final = self._iterator
        self._iterator = self._logging_iterator(source_iterator)

    def _logging_iterator(
        self, source_iterator: Iterator[TranscriptionStreamEvent]
    ) -> Iterator[TranscriptionStreamEvent]:
        try:
            for event in source_iterator:
                self._collector.add(event)
                yield event
        except Exception as exception:
            self._failed = True
            end_time: Final = datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract
            self._logging_obj.failure_handler(exception, traceback.format_exc(), self._start_time, end_time)
            raise
        finally:
            self._finalize()

    def _finalize(self) -> None:
        if self._finalized or self._failed:
            return
        self._finalized = True
        self._logging_obj.success_handler(
            self._collector.response(),
            self._start_time,
            datetime.datetime.now(),  # noqa: DTZ005  # callback timestamps use the legacy naive contract
        )

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._finalize()

    @property
    def logging_obj(self) -> TranscriptionStreamLogging:
        return self._logging_obj


class LoggingAsyncTranscriptionStream(AsyncStream[TranscriptionStreamEvent]):
    def __init__(
        self,
        stream: AsyncStream[TranscriptionStreamEvent],
        logging_obj: TranscriptionStreamLogging,
        start_time: datetime.datetime,
    ) -> None:
        self.__dict__.update(stream.__dict__)
        self._hidden_params: dict[str, object] = {  # mutable-ok: ResponseMetadata fills in the call's metadata
            "additional_headers": process_response_headers(stream.response.headers)
        }
        self._logging_obj = logging_obj
        self._start_time = start_time
        self._collector = _TranscriptionEventCollector(getattr(stream, "_litellm_audio_duration", None))
        self._finalized = False
        self._failed = False
        source_iterator: Final = self._iterator
        self._iterator = self._logging_iterator(source_iterator)

    async def _logging_iterator(
        self, source_iterator: AsyncIterator[TranscriptionStreamEvent]
    ) -> AsyncIterator[TranscriptionStreamEvent]:
        try:
            async for event in source_iterator:
                self._collector.add(event)
                yield event
        except Exception as exception:
            self._failed = True
            end_time: Final = datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract
            self._logging_obj.failure_handler(exception, traceback.format_exc(), self._start_time, end_time)
            await self._logging_obj.async_failure_handler(
                exception,
                traceback.format_exc(),
                self._start_time,
                end_time,
            )
            raise
        finally:
            await self._finalize()

    async def _finalize(self) -> None:
        if self._finalized or self._failed:
            return
        self._finalized = True
        response: Final = self._collector.response()
        end_time: Final = datetime.datetime.now()  # noqa: DTZ005  # callback timestamps use the legacy naive contract
        self._logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=response,
            start_time=self._start_time,
            end_time=end_time,
        )
        with anyio.CancelScope(shield=True):
            await self._logging_obj.async_success_handler(
                result=response,
                start_time=self._start_time,
                end_time=end_time,
            )

    async def close(self) -> None:
        try:
            await super().close()
        finally:
            await self._finalize()

    @property
    def logging_obj(self) -> TranscriptionStreamLogging:
        return self._logging_obj


def wrap_transcription_stream(
    stream: Stream[TranscriptionStreamEvent] | AsyncStream[TranscriptionStreamEvent],
    logging_obj: TranscriptionStreamLogging,
    start_time: datetime.datetime,
) -> LoggingTranscriptionStream | LoggingAsyncTranscriptionStream:
    if isinstance(stream, AsyncStream):
        return LoggingAsyncTranscriptionStream(stream, logging_obj, start_time)
    return LoggingTranscriptionStream(stream, logging_obj, start_time)
