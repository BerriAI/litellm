from __future__ import annotations

import datetime
import traceback
from collections.abc import AsyncIterator, Iterator
from typing import Protocol

from openai import AsyncStream, Stream
from openai.types.audio import (
    TranscriptionStreamEvent,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
)

from litellm.types.utils import TranscriptionResponse


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
        self.text_deltas: list[str] = []
        self.done_event: TranscriptionTextDoneEvent | None = None

    def add(self, event: TranscriptionStreamEvent) -> None:
        if isinstance(event, TranscriptionTextDeltaEvent):
            self.text_deltas.append(event.delta)
        elif isinstance(event, TranscriptionTextDoneEvent):
            self.done_event = event

    def response(self) -> TranscriptionResponse:
        done_event = self.done_event
        response = TranscriptionResponse(
            text=done_event.text if done_event is not None else "".join(self.text_deltas),
            usage=done_event.usage.model_dump() if done_event is not None and done_event.usage is not None else None,
            languages=(
                [language.model_dump() for language in done_event.languages]
                if done_event is not None and done_event.languages is not None
                else None
            ),
        )
        if self.duration is not None:
            response._hidden_params["audio_transcription_duration"] = self.duration
        return response


class LoggingTranscriptionStream(Stream[TranscriptionStreamEvent]):
    def __init__(
        self,
        stream: Stream[TranscriptionStreamEvent],
        logging_obj: TranscriptionStreamLogging,
        start_time: datetime.datetime,
    ) -> None:
        self.__dict__.update(stream.__dict__)
        self._logging_obj = logging_obj
        self._start_time = start_time
        self._collector = _TranscriptionEventCollector(getattr(stream, "_litellm_audio_duration", None))
        self._finalized = False
        self._failed = False
        source_iterator = self._iterator
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
            end_time = datetime.datetime.now()
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
            datetime.datetime.now(),
        )

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._finalize()


class LoggingAsyncTranscriptionStream(AsyncStream[TranscriptionStreamEvent]):
    def __init__(
        self,
        stream: AsyncStream[TranscriptionStreamEvent],
        logging_obj: TranscriptionStreamLogging,
        start_time: datetime.datetime,
    ) -> None:
        self.__dict__.update(stream.__dict__)
        self._logging_obj = logging_obj
        self._start_time = start_time
        self._collector = _TranscriptionEventCollector(getattr(stream, "_litellm_audio_duration", None))
        self._finalized = False
        self._failed = False
        source_iterator = self._iterator
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
            end_time = datetime.datetime.now()
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
        response = self._collector.response()
        end_time = datetime.datetime.now()
        self._logging_obj.handle_sync_success_callbacks_for_async_calls(
            result=response,
            start_time=self._start_time,
            end_time=end_time,
        )
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


def wrap_transcription_stream(
    stream: Stream[TranscriptionStreamEvent] | AsyncStream[TranscriptionStreamEvent],
    logging_obj: TranscriptionStreamLogging,
    start_time: datetime.datetime,
) -> LoggingTranscriptionStream | LoggingAsyncTranscriptionStream:
    if isinstance(stream, AsyncStream):
        return LoggingAsyncTranscriptionStream(stream, logging_obj, start_time)
    return LoggingTranscriptionStream(stream, logging_obj, start_time)
