import datetime
from collections.abc import AsyncIterator
from typing import Final

import anyio
import anyio.lowlevel
import httpx
import pytest
from openai import AsyncOpenAI

from litellm.litellm_core_utils.audio_utils.transcription_streaming import wrap_transcription_stream
from litellm.types.utils import TranscriptionResponse

_STARTED_AT: Final = datetime.datetime(2026, 10, 28, 12, 0, 0)  # noqa: DTZ001  # callback timestamps use the legacy naive contract


class _FirstDeltaThenStall(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"type":"transcript.text.delta","delta":"Good"}\n\n'
        await anyio.sleep_forever()


class _RecordingLogging:
    def __init__(self) -> None:
        self.logged: tuple[TranscriptionResponse, ...] = ()

    def success_handler(
        self, result: TranscriptionResponse, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        return None

    async def async_success_handler(
        self, result: TranscriptionResponse, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        await anyio.lowlevel.checkpoint()
        self.logged = (*self.logged, result)

    def handle_sync_success_callbacks_for_async_calls(
        self, result: TranscriptionResponse, start_time: datetime.datetime, end_time: datetime.datetime
    ) -> None:
        return None

    def failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        return None

    async def async_failure_handler(
        self,
        exception: Exception,
        traceback_exception: str,
        start_time: datetime.datetime,
        end_time: datetime.datetime,
    ) -> None:
        return None


@pytest.mark.asyncio
async def test_a_stream_the_client_drops_mid_transcript_is_still_logged_as_a_success() -> None:
    upstream: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://upstream.example/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200, headers={"content-type": "text/event-stream"}, stream=_FirstDeltaThenStall()
                )
            )
        ),
    )
    stream: Final = await upstream.audio.transcriptions.create(
        model="gpt-transcribe", file=("sample.wav", b"audio"), stream=True
    )
    logging_obj: Final = _RecordingLogging()
    wrapped: Final = wrap_transcription_stream(stream, logging_obj, _STARTED_AT)

    with anyio.CancelScope() as client_connection:
        async for _ in wrapped:
            client_connection.cancel()

    assert [result.text for result in logging_obj.logged] == ["Good"]
