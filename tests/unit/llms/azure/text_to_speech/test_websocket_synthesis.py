import base64
import json
from collections.abc import Iterator, Mapping, Sequence
from typing import Final, Self
from unittest.mock import AsyncMock, patch

import pytest
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosedError, InvalidStatus
from websockets.frames import Close
from websockets.http11 import Response

import litellm

# Frame shapes copied from a live swedencentral.tts.speech.microsoft.com session on 2026-11-23
_AUDIO_HEADERS: Final = b"X-RequestId:abc\r\nContent-Type:audio/mpeg\r\nX-StreamId:S1\r\nPath:audio\r\n"
_FIRST_AUDIO: Final = b"\xff\xf3first"
_SECOND_AUDIO: Final = b"\xff\xf3second"


def _text_frame(path: str, body: str) -> str:
    return f"X-RequestId:abc\r\nContent-Type:application/json; charset=utf-8\r\nPath:{path}\r\n\r\n{body}"


def _audio_frame(audio: bytes) -> bytes:
    return len(_AUDIO_HEADERS).to_bytes(2, "big") + _AUDIO_HEADERS + audio


def _metadata_frame(entry_type: str, data: dict[str, object]) -> str:
    return _text_frame("audio.metadata", json.dumps({"Metadata": [{"Type": entry_type, "Data": data}]}))


_SYNTHESIS: Final = (
    _text_frame("turn.start", '{"context": {"serviceTag": "tag"}}'),
    _text_frame("response", '{"audio": {"type": "inline", "streamId": "S1"}}'),
    _audio_frame(_FIRST_AUDIO),
    _metadata_frame("Viseme", {"Offset": 875000, "VisemeId": 12, "IsLastAnimation": True}),
    _audio_frame(_SECOND_AUDIO),
    _metadata_frame("SessionEnd", {"Offset": 28375000}),
    _text_frame("turn.end", "{}"),
)
_EVENTS_WITH_VISEMES: Final = [
    {"type": "speech.audio.delta", "audio": base64.b64encode(_FIRST_AUDIO).decode()},
    {"type": "speech.audio.viseme", "viseme_id": 12, "audio_offset_ms": 87.5},
    {"type": "speech.audio.delta", "audio": base64.b64encode(_SECOND_AUDIO).decode()},
    {"type": "speech.audio.done"},
]
_UNSUPPORTED_VOICE: Final = ConnectionClosedError(rcvd=Close(1007, "Unsupported voice xx-XX-NopeNeural."), sent=None)
_SPEECH_REQUEST: Final = {
    "model": "azure/speech/azure-tts",
    "voice": "de-DE-FlorianMultilingualNeural",
    "input": "Hallo Welt",
    "api_base": "https://eastus.tts.speech.microsoft.com",
    "api_key": "test-key",
    "stream_format": "sse",
}


class _FakeSpeechConnection:
    def __init__(self, replies: Sequence[str | bytes | Exception]) -> None:
        self.url: str | None = None
        self.headers: Mapping[str, str] = {}
        self.sent: Final[list[str]] = []
        self.closed = False
        self._replies: Final = iter(replies)

    def _opened(self, url: str, additional_headers: Mapping[str, str]) -> None:
        self.url = url
        self.headers = additional_headers

    def _reply(self) -> str | bytes | None:
        reply: Final = next(self._replies, None)
        if isinstance(reply, Exception):
            raise reply
        return reply


class _FakeAsyncSpeechConnection(_FakeSpeechConnection):
    async def connect(self, url: str, *, additional_headers: Mapping[str, str], **_: object) -> Self:
        self._opened(url, additional_headers)
        return self

    async def send(self, message: str) -> None:
        self.sent.append(message)

    async def recv(self) -> str | bytes:
        reply: Final = self._reply()
        assert reply is not None, "the client kept reading after the last frame"
        return reply

    def __aiter__(self) -> "_FakeAsyncSpeechConnection":
        return self

    async def __anext__(self) -> str | bytes:
        reply: Final = self._reply()
        if reply is None:
            raise StopAsyncIteration
        return reply

    async def close(self) -> None:
        self.closed = True


class _FakeSyncSpeechConnection(_FakeSpeechConnection):
    def connect(self, url: str, *, additional_headers: Mapping[str, str], **_: object) -> Self:
        self._opened(url, additional_headers)
        return self

    def send(self, message: str) -> None:
        self.sent.append(message)

    def recv(self) -> str | bytes:
        reply: Final = self._reply()
        assert reply is not None, "the client kept reading after the last frame"
        return reply

    def __iter__(self) -> Iterator[str | bytes]:
        return iter(self._reply, None)

    def close(self) -> None:
        self.closed = True


def _events(body: bytes) -> list[dict[str, object]]:
    return [json.loads(line.removeprefix("data: ")) for line in body.decode().split("\n\n") if line]


def _sent_body(message: str) -> str:
    return message.partition("\r\n\r\n")[2]


@pytest.mark.asyncio
async def test_aspeech_sse_returns_audio_and_visemes_as_ordered_events() -> None:
    connection: Final = _FakeAsyncSpeechConnection(_SYNTHESIS)

    with patch("websockets.connect", new=connection.connect):
        response: Final = await litellm.aspeech(**_SPEECH_REQUEST, viseme_enabled=True, response_format="opus")

    assert response.content_type == "text/event-stream"
    assert _events(response.content) == _EVENTS_WITH_VISEMES
    assert connection.url == "wss://eastus.tts.speech.microsoft.com/cognitiveservices/websocket/v1"
    assert connection.headers["Ocp-Apim-Subscription-Key"] == "test-key"
    context_message, ssml_message = connection.sent
    assert context_message.startswith("Path: synthesis.context\r\n")
    assert json.loads(_sent_body(context_message))["synthesis"]["audio"] == {
        "metadataOptions": {"visemeEnabled": True},
        "outputFormat": "ogg-48khz-16bit-mono-opus",
    }
    assert ssml_message.startswith("Path: ssml\r\n")
    assert "<voice name='de-DE-FlorianMultilingualNeural'>" in _sent_body(ssml_message)
    assert "Hallo Welt" in _sent_body(ssml_message)


@pytest.mark.asyncio
async def test_aspeech_sse_without_viseme_enabled_does_not_ask_azure_for_visemes() -> None:
    connection: Final = _FakeAsyncSpeechConnection(_SYNTHESIS)

    with patch("websockets.connect", new=connection.connect):
        await litellm.aspeech(**_SPEECH_REQUEST)

    assert json.loads(_sent_body(connection.sent[0]))["synthesis"]["audio"]["metadataOptions"] == {
        "visemeEnabled": False
    }


@pytest.mark.asyncio
async def test_aspeech_sse_stream_yields_each_event_as_it_arrives() -> None:
    connection: Final = _FakeAsyncSpeechConnection(_SYNTHESIS)

    with patch("websockets.connect", new=connection.connect):
        response: Final = await litellm.aspeech(**_SPEECH_REQUEST, viseme_enabled=True, stream=True)

    assert response.response.is_stream_consumed is False
    chunks: Final = [chunk async for chunk in await response.aiter_bytes()]
    assert [_events(chunk) for chunk in chunks] == [[event] for event in _EVENTS_WITH_VISEMES]


@pytest.mark.asyncio
async def test_aspeech_sse_stream_closes_the_connection_when_the_caller_stops_reading() -> None:
    connection: Final = _FakeAsyncSpeechConnection(_SYNTHESIS)

    with patch("websockets.connect", new=connection.connect):
        response: Final = await litellm.aspeech(**_SPEECH_REQUEST, viseme_enabled=True, stream=True)

    chunks: Final = await response.aiter_bytes()
    assert _events(await anext(chunks)) == _EVENTS_WITH_VISEMES[:1]
    assert connection.closed is False
    await response.aclose()
    assert connection.closed is True


@pytest.mark.asyncio
async def test_aspeech_sse_stops_reading_at_turn_end() -> None:
    connection: Final = _FakeAsyncSpeechConnection((*_SYNTHESIS, _audio_frame(b"audio of a later turn")))

    with patch("websockets.connect", new=connection.connect):
        response: Final = await litellm.aspeech(**_SPEECH_REQUEST, viseme_enabled=True)

    assert _events(response.content) == _EVENTS_WITH_VISEMES


@pytest.mark.asyncio
async def test_aspeech_sse_uses_plain_websocket_for_http_api_base() -> None:
    connection: Final = _FakeAsyncSpeechConnection(_SYNTHESIS)

    with patch("websockets.connect", new=connection.connect):
        await litellm.aspeech(**{**_SPEECH_REQUEST, "api_base": "http://localhost:8080"})

    assert connection.url == "ws://localhost:8080/cognitiveservices/websocket/v1"


@pytest.mark.asyncio
async def test_aspeech_sse_maps_azure_closing_on_a_bad_request_to_bad_request_error() -> None:
    connection: Final = _FakeAsyncSpeechConnection((_SYNTHESIS[0], _UNSUPPORTED_VOICE))

    with patch("websockets.connect", new=connection.connect):
        with pytest.raises(litellm.BadRequestError, match="Unsupported voice xx-XX-NopeNeural"):
            await litellm.aspeech(**_SPEECH_REQUEST)

    assert connection.closed is True


@pytest.mark.asyncio
async def test_aspeech_sse_maps_a_refused_handshake_to_its_http_status() -> None:
    refused: Final = InvalidStatus(Response(401, "Unauthorized", Headers(), b""))

    with patch("websockets.connect", new=AsyncMock(side_effect=refused)):
        with pytest.raises(litellm.AuthenticationError):
            await litellm.aspeech(**_SPEECH_REQUEST)


@pytest.mark.asyncio
async def test_aspeech_viseme_enabled_without_sse_is_rejected_before_calling_azure() -> None:
    audio_only_request: Final = {key: value for key, value in _SPEECH_REQUEST.items() if key != "stream_format"}

    connection: Final = _FakeAsyncSpeechConnection(_SYNTHESIS)

    with patch("websockets.connect", new=connection.connect):
        with pytest.raises(litellm.BadRequestError, match="viseme_enabled requires stream_format='sse'"):
            await litellm.aspeech(**audio_only_request, viseme_enabled=True)

    assert connection.url is None


def test_speech_sse_returns_audio_and_visemes_as_ordered_events() -> None:
    connection: Final = _FakeSyncSpeechConnection(_SYNTHESIS)

    with patch("websockets.sync.client.connect", new=connection.connect):
        response: Final = litellm.speech(**_SPEECH_REQUEST, viseme_enabled=True)

    assert response.content_type == "text/event-stream"
    assert _events(response.content) == _EVENTS_WITH_VISEMES
    assert connection.url == "wss://eastus.tts.speech.microsoft.com/cognitiveservices/websocket/v1"
    assert json.loads(_sent_body(connection.sent[0]))["synthesis"]["audio"]["metadataOptions"] == {
        "visemeEnabled": True
    }


def test_speech_sse_stream_yields_each_event_as_it_arrives_and_closes_the_connection() -> None:
    connection: Final = _FakeSyncSpeechConnection(_SYNTHESIS)

    with patch("websockets.sync.client.connect", new=connection.connect):
        response: Final = litellm.speech(**_SPEECH_REQUEST, viseme_enabled=True, stream=True)

    assert response.response.is_stream_consumed is False
    assert [_events(chunk) for chunk in response.iter_bytes()] == [[event] for event in _EVENTS_WITH_VISEMES]
    assert connection.closed is True


def test_speech_sse_closes_the_connection_when_azure_refuses_the_request() -> None:
    connection: Final = _FakeSyncSpeechConnection((_SYNTHESIS[0], _UNSUPPORTED_VOICE))

    with patch("websockets.sync.client.connect", new=connection.connect):
        with pytest.raises(Exception, match="Unsupported voice xx-XX-NopeNeural") as refusal:
            litellm.speech(**_SPEECH_REQUEST)

    assert getattr(refusal.value, "status_code", None) == 400
    assert connection.closed is True
