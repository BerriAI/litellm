import datetime
import io
import math
import wave
from collections.abc import AsyncIterator, Iterator
from typing import Final

import anyio
import anyio.lowlevel
import httpx
import pytest
import respx
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.audio_utils.transcription_streaming import (
    litellm_exception_mapper,
    wrap_transcription_stream,
)
from litellm.types.utils import TranscriptionResponse

_STARTED_AT: Final = datetime.datetime(2026, 10, 28, 12, 0, 0)  # noqa: DTZ001  # callback timestamps use the legacy naive contract


class _FirstDeltaThenStall(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"type":"transcript.text.delta","delta":"Good"}\n\n'
        await anyio.sleep_forever()


class _FirstDeltaThenFails(httpx.AsyncByteStream):
    def __init__(self, failure: httpx.TransportError) -> None:
        self._failure = failure

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"type":"transcript.text.delta","delta":"Good"}\n\n'
        raise self._failure


def _transcription_router(failing_api_base: str, healthy_api_base: str) -> Router:
    return Router(
        model_list=[
            {
                "model_name": "transcriber",
                "litellm_params": {
                    "model": "openai/gpt-transcribe",
                    "api_key": "sk-test",
                    "api_base": api_base,
                    "order": order,
                },
            }
            for order, api_base in ((1, failing_api_base), (2, healthy_api_base))
        ],
        allowed_fails=0,
        cooldown_time=60,
    )


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
        *,
        for_async_request: bool = False,
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
    wrapped: Final = wrap_transcription_stream(stream, logging_obj, _STARTED_AT, litellm_exception_mapper({}))

    with anyio.CancelScope() as client_connection:
        async for _ in wrapped:
            client_connection.cancel()

    assert [result.text for result in logging_obj.logged] == ["Good"]


@pytest.mark.asyncio
async def test_a_deployment_whose_stream_times_out_mid_transcript_goes_into_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router: Final = _transcription_router("https://stalls.example/v1", "https://healthy.example/v1")

    with respx.mock(assert_all_called=False) as upstream:
        upstream.post("https://stalls.example/v1/audio/transcriptions").mock(
            side_effect=lambda request: httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                stream=_FirstDeltaThenFails(httpx.ReadTimeout("read timed out")),
            )
        )
        upstream.post("https://healthy.example/v1/audio/transcriptions").mock(
            return_value=httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=b'data: {"type":"transcript.text.done","text":"Good morning"}\n\n',
            )
        )
        stalled: Final = await router.atranscription(model="transcriber", file=("sample.wav", b"audio"), stream=True)
        with pytest.raises(litellm.Timeout):
            async for _ in stalled:
                pass
        await anyio.sleep(0.1)
        retried: Final = await router.atranscription(model="transcriber", file=("sample.wav", b"audio"), stream=True)
        events: Final = [event async for event in retried]

    assert [getattr(event, "text", None) for event in events] == ["Good morning"]


@pytest.mark.asyncio
async def test_a_stream_the_upstream_drops_mid_transcript_raises_a_litellm_error_to_the_caller() -> None:
    upstream: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://upstream.example/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=_FirstDeltaThenFails(httpx.RemoteProtocolError("peer closed connection")),
                )
            )
        ),
    )
    stream: Final = await litellm.atranscription(
        model="openai/gpt-transcribe", file=("sample.wav", b"audio"), stream=True, client=upstream
    )

    with pytest.raises(litellm.APIConnectionError, match="peer closed connection"):
        async for _ in stream:
            pass


class _FailureHookCounter(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.failures: tuple[str, ...] = ()

    def log_failure_event(self, kwargs: object, response_obj: object, start_time: object, end_time: object) -> None:
        self.failures = (*self.failures, "sync_failure")

    async def async_log_failure_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.failures = (*self.failures, "async_failure")


@pytest.mark.asyncio
async def test_an_async_stream_that_fails_mid_transcript_is_logged_as_one_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter: Final = _FailureHookCounter()
    monkeypatch.setattr(litellm, "callbacks", [counter])
    upstream: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://upstream.example/v1",
        http_client=httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    headers={"content-type": "text/event-stream"},
                    stream=_FirstDeltaThenFails(httpx.ReadError("connection reset")),
                )
            )
        ),
    )
    stream: Final = await litellm.atranscription(
        model="openai/gpt-transcribe", file=("sample.wav", b"audio"), stream=True, client=upstream
    )

    with pytest.raises(litellm.APIConnectionError):
        async for _ in stream:
            pass

    assert counter.failures == ("async_failure",)


class _FirstDeltaThenResetSync(httpx.SyncByteStream):
    def __iter__(self) -> Iterator[bytes]:
        yield b'data: {"type":"transcript.text.delta","delta":"Good"}\n\n'
        raise httpx.ReadError("connection reset")


class _FailureCostRecorder(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.costs: tuple[object, ...] = ()

    def log_failure_event(self, kwargs: object, response_obj: object, start_time: object, end_time: object) -> None:
        self.costs = (*self.costs, kwargs.get("response_cost") if isinstance(kwargs, dict) else None)

    async def async_log_failure_event(
        self, kwargs: object, response_obj: object, start_time: object, end_time: object
    ) -> None:
        self.costs = (*self.costs, kwargs.get("response_cost") if isinstance(kwargs, dict) else None)


def _silent_wav(seconds: int) -> bytes:
    buffer: Final = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 16_000 * seconds)
    return buffer.getvalue()


def _reset_mid_transcript(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        headers={"content-type": "text/event-stream"},
        stream=_FirstDeltaThenFails(httpx.ReadError("connection reset")),
    )


def _reset_mid_transcript_sync(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_FirstDeltaThenResetSync())


def _per_second_rate(model: str) -> float:
    rate: Final = litellm.model_cost[model]["input_cost_per_second"]
    assert isinstance(rate, float)
    return rate


@pytest.mark.asyncio
async def test_an_async_stream_that_fails_mid_transcript_bills_the_uploaded_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder: Final = _FailureCostRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    upstream: Final = AsyncOpenAI(
        api_key="sk-test",
        base_url="https://upstream.example/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(_reset_mid_transcript)),
    )
    stream: Final = await litellm.atranscription(
        model="openai/gpt-transcribe", file=("sample.wav", _silent_wav(4)), stream=True, client=upstream
    )

    with pytest.raises(litellm.APIConnectionError):
        async for _ in stream:
            pass

    assert len(recorder.costs) == 1
    assert isinstance(recorder.costs[0], float)
    assert math.isclose(recorder.costs[0], 4 * _per_second_rate("gpt-transcribe"))


def test_a_sync_stream_that_fails_mid_transcript_bills_the_uploaded_seconds(monkeypatch: pytest.MonkeyPatch) -> None:
    recorder: Final = _FailureCostRecorder()
    monkeypatch.setattr(litellm, "callbacks", [recorder])
    upstream: Final = OpenAI(
        api_key="sk-test",
        base_url="https://upstream.example/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(_reset_mid_transcript_sync)),
    )
    stream: Final = litellm.transcription(
        model="openai/gpt-transcribe", file=("sample.wav", _silent_wav(4)), stream=True, client=upstream
    )

    with pytest.raises(litellm.APIConnectionError):
        for _ in stream:
            pass

    assert len(recorder.costs) == 1
    assert isinstance(recorder.costs[0], float)
    assert math.isclose(recorder.costs[0], 4 * _per_second_rate("gpt-transcribe"))
