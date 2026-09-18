import json
from collections.abc import AsyncIterator, Sequence
from datetime import timedelta
from typing import Final

import pytest
from google.cloud.speech_v2.types import (
    RecognitionResponseMetadata,
    SpeechRecognitionAlternative,
    StreamingRecognitionResult,
    StreamingRecognizeRequest,
    StreamingRecognizeResponse,
)
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from litellm.llms.vertex_ai.audio_transcription.realtime_backend import SpeechStreamingBackend
from litellm.llms.vertex_ai.audio_transcription.realtime_transformation import SpeechStreamingTarget

TARGET: Final = SpeechStreamingTarget(
    api_endpoint="us-speech.googleapis.com",
    recognizer="projects/proj-1/locations/us/recognizers/_",
    access_token="token",
)
CONFIGURE: Final = json.dumps(
    {"kind": "configure", "model": "chirp_3", "language_codes": ["en-US"], "sample_rate_hertz": 16_000}
)
FINISH_TURN: Final = json.dumps({"kind": "finish_turn"})
DISCARD_TURN: Final = json.dumps({"kind": "discard_turn"})
ScriptItem = StreamingRecognizeResponse | Exception


def _response(
    transcript: str | None,
    *,
    is_final: bool = False,
    billed: float = 0.0,
    event: str = "SPEECH_EVENT_TYPE_UNSPECIFIED",
) -> StreamingRecognizeResponse:
    results = (
        []
        if transcript is None
        else [
            StreamingRecognitionResult(
                alternatives=[SpeechRecognitionAlternative(transcript=transcript)], is_final=is_final
            )
        ]
    )
    return StreamingRecognizeResponse(
        results=results,
        speech_event_type=event,
        metadata=RecognitionResponseMetadata(total_billed_duration=timedelta(seconds=billed)),
    )


class _FakeTransport:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class _FakeSpeechClient:
    def __init__(self, *scripts: Sequence[ScriptItem]) -> None:
        self.transport: Final = _FakeTransport()
        self.streams: Final[list[list[StreamingRecognizeRequest]]] = []
        self._scripts: Final = [list(script) for script in scripts]

    async def streaming_recognize(
        self, requests: AsyncIterator[StreamingRecognizeRequest] | None = None
    ) -> AsyncIterator[StreamingRecognizeResponse]:
        assert requests is not None
        script: Final = self._scripts.pop(0) if self._scripts else []
        received: Final[list[StreamingRecognizeRequest]] = []
        self.streams.append(received)
        return self._respond(requests, script, received)

    async def _respond(
        self,
        requests: AsyncIterator[StreamingRecognizeRequest],
        script: list[ScriptItem],
        received: list[StreamingRecognizeRequest],
    ) -> AsyncIterator[StreamingRecognizeResponse]:
        async for request in requests:
            received.append(request)
            if request.audio and script:
                yield self._next(script)
        while script:
            yield self._next(script)

    @staticmethod
    def _next(script: list[ScriptItem]) -> StreamingRecognizeResponse:
        item: Final = script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _backend(client: _FakeSpeechClient, **kwargs: object) -> SpeechStreamingBackend:
    return SpeechStreamingBackend(TARGET, client_factory=lambda target: client, **kwargs)


async def _recv(backend: SpeechStreamingBackend) -> dict[str, object]:
    message: Final = await backend.recv()
    assert isinstance(message, str)
    return json.loads(message)


async def _configure(backend: SpeechStreamingBackend) -> None:
    await backend.send(CONFIGURE)
    assert await _recv(backend) == {"kind": "configured"}


def _audio(stream: list[StreamingRecognizeRequest]) -> list[bytes]:
    return [bytes(request.audio) for request in stream[1:]]


@pytest.mark.asyncio
async def test_audio_streams_through_one_recognize_call_with_the_config_first():
    client = _FakeSpeechClient([_response("hello"), _response("hello world", is_final=True, billed=2.0)])
    async with _backend(client) as backend:
        await _configure(backend)
        await backend.send(b"\x01\x02")
        await backend.send(b"\x03\x04")
        await backend.send(FINISH_TURN)
        first, second, finished = [await _recv(backend) for _ in range(3)]
    assert first == {
        "kind": "response",
        "speech_event": "none",
        "results": [{"transcript": "hello", "is_final": False}],
        "billed_seconds": 0.0,
    }
    assert second["results"] == [{"transcript": "hello world", "is_final": True}]
    assert second["billed_seconds"] == 2.0
    assert finished == {"kind": "turn_finished"}
    (requests,) = client.streams
    assert requests[0].recognizer == TARGET.recognizer
    config = requests[0].streaming_config
    assert config.config.model == "chirp_3"
    assert list(config.config.language_codes) == ["en-US"]
    assert config.config.explicit_decoding_config.sample_rate_hertz == 16_000
    assert config.config.explicit_decoding_config.audio_channel_count == 1
    assert config.config.explicit_decoding_config.encoding.name == "LINEAR16"
    assert config.streaming_features.interim_results
    assert config.streaming_features.enable_voice_activity_events
    assert _audio(requests) == [b"\x01\x02", b"\x03\x04"]
    assert client.transport.closed


@pytest.mark.asyncio
async def test_voice_activity_events_are_relayed():
    client = _FakeSpeechClient([_response(None, event="SPEECH_ACTIVITY_BEGIN"), _response(None, event="SPEECH_ACTIVITY_END")])
    async with _backend(client) as backend:
        await _configure(backend)
        await backend.send(b"\x00\x00")
        await backend.send(b"\x00\x00")
        begin, end = [await _recv(backend) for _ in range(2)]
    assert (begin["speech_event"], begin["results"]) == ("begin", [])
    assert end["speech_event"] == "end"


@pytest.mark.asyncio
async def test_audio_before_configure_is_rejected():
    backend = _backend(_FakeSpeechClient())
    with pytest.raises(RuntimeError, match="before the Speech-to-Text stream was configured"):
        await backend.send(b"\x00\x00")


@pytest.mark.asyncio
async def test_stream_failure_closes_the_session_with_1011_and_the_reason():
    client = _FakeSpeechClient([PermissionError("IAM_PERMISSION_DENIED: speech.recognizers.recognize")])
    async with _backend(client) as backend:
        await _configure(backend)
        await backend.send(b"\x00\x00")
        with pytest.raises(ConnectionClosedError) as excinfo:
            await backend.recv()
    assert excinfo.value.rcvd is not None
    assert excinfo.value.rcvd.code == 1011
    assert "IAM_PERMISSION_DENIED" in excinfo.value.rcvd.reason
    assert client.transport.closed


@pytest.mark.asyncio
async def test_close_discards_the_open_turn_then_reports_a_normal_closure():
    client = _FakeSpeechClient([_response("hi")])
    backend = _backend(client)
    await _configure(backend)
    await backend.send(b"\x00\x00")
    assert (await _recv(backend))["results"][0]["transcript"] == "hi"
    await backend.close()
    assert await _recv(backend) == {"kind": "turn_discarded"}
    with pytest.raises(ConnectionClosedOK):
        await backend.recv()
    assert client.transport.closed


@pytest.mark.asyncio
async def test_turn_commands_without_audio_answer_immediately():
    backend = _backend(_FakeSpeechClient())
    await _configure(backend)
    await backend.send(FINISH_TURN)
    assert await _recv(backend) == {"kind": "turn_finished"}
    await backend.send(DISCARD_TURN)
    assert await _recv(backend) == {"kind": "turn_discarded"}


@pytest.mark.asyncio
async def test_discard_turn_cancels_the_open_stream_and_the_next_turn_starts_fresh():
    client = _FakeSpeechClient([_response("draft")], [_response("again", is_final=True)])
    async with _backend(client) as backend:
        await _configure(backend)
        await backend.send(b"\x01\x01")
        assert (await _recv(backend))["results"][0]["transcript"] == "draft"
        await backend.send(DISCARD_TURN)
        assert await _recv(backend) == {"kind": "turn_discarded"}
        await backend.send(b"\x02\x02")
        assert (await _recv(backend))["results"][0]["transcript"] == "again"
    assert [_audio(stream) for stream in client.streams] == [[b"\x01\x01"], [b"\x02\x02"]]


@pytest.mark.asyncio
async def test_billed_seconds_accumulate_across_turns():
    client = _FakeSpeechClient([_response("one", is_final=True, billed=2.0)], [_response("two", is_final=True, billed=3.0)])
    async with _backend(client) as backend:
        await _configure(backend)
        await backend.send(b"\x00\x00")
        await backend.send(FINISH_TURN)
        first = await _recv(backend)
        assert await _recv(backend) == {"kind": "turn_finished"}
        await backend.send(b"\x00\x00")
        await backend.send(FINISH_TURN)
        second = await _recv(backend)
        assert await _recv(backend) == {"kind": "turn_finished"}
    assert (first["billed_seconds"], second["billed_seconds"]) == (2.0, 5.0)
    assert len(client.streams) == 2


@pytest.mark.asyncio
async def test_streams_rotate_before_the_five_minute_limit_without_losing_audio():
    now = [0.0]
    client = _FakeSpeechClient(
        [_response("first"), _response("first half", is_final=True, billed=239.0)],
        [_response("second")],
    )
    async with _backend(client, clock=lambda: now[0], rotation_seconds=240.0) as backend:
        await _configure(backend)
        await backend.send(b"\x01\x01")
        assert (await _recv(backend))["results"][0]["transcript"] == "first"
        now[0] = 239.0
        await backend.send(b"\x02\x02")
        assert (await _recv(backend))["results"][0]["transcript"] == "first half"
        now[0] = 240.0
        await backend.send(b"\x03\x03")
        assert await _recv(backend) == {"kind": "turn_finished"}
        second = await _recv(backend)
        assert second["results"][0]["transcript"] == "second"
        assert second["billed_seconds"] == 239.0
        await backend.send(FINISH_TURN)
        assert await _recv(backend) == {"kind": "turn_finished"}
    assert [_audio(stream) for stream in client.streams] == [[b"\x01\x01", b"\x02\x02"], [b"\x03\x03"]]
    assert client.streams[1][0].streaming_config.config.model == "chirp_3"
