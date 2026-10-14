"""Behavior pins for ``proxy_server.py`` audio routes.

Pins (PR2):
    - POST /v1/audio/speech
    - POST /audio/speech
    - POST /v1/audio/transcriptions
    - POST /audio/transcriptions
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import io
from typing import Final

import orjson
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from starlette.background import BackgroundTask
from starlette.responses import StreamingResponse

from litellm.constants import AUDIO_SPEECH_CHUNK_SIZE
from fastapi import Request, Response

from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.llms.openai import HttpxBinaryResponseContent


@pytest.fixture
def patched_speech(monkeypatch, request):
    upstream_content_type = getattr(request, "param", "audio/mpeg")
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(return_value={}),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)

    async def _llm_call():
        return HttpxBinaryResponseContent(
            httpx.Response(
                status_code=200,
                headers={} if upstream_content_type is None else {"content-type": upstream_content_type},
                content=b"\x00\x01\x02",
            )
        )

    async def _fake_route_request(*args, **kwargs):
        return _llm_call()

    monkeypatch.setattr(proxy_server, "route_request", _fake_route_request)
    yield


@pytest.fixture
def patched_speech_error(monkeypatch):
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(return_value={}),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)

    async def _raise(*args, **kwargs):
        raise ValueError("speech boom")

    monkeypatch.setattr(proxy_server, "route_request", _raise)
    yield


@pytest.fixture
def patched_speech_provider_rejection(monkeypatch, patched_speech_error):
    import litellm

    async def _raise(*args, **kwargs):
        raise litellm.BadRequestError(
            message=(
                "Gemini TTS only produces raw PCM16 audio, so response_format='mp3' is not supported."
                " Supported response formats: pcm, wav."
            ),
            model="gemini-3.1-flash-tts-preview",
            llm_provider="gemini",
        )

    monkeypatch.setattr(proxy_server, "route_request", _raise)
    yield


@pytest.fixture
def patched_transcription(monkeypatch):
    router = MagicMock()
    router.model_names = ["whisper-1"]
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(return_value={}),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)
    monkeypatch.setattr(
        proxy_server, "check_file_size_under_limit", lambda **kwargs: True
    )

    async def _form_data(request):
        from starlette.datastructures import FormData, UploadFile

        upload = UploadFile(
            filename="audio.mp3",
            file=io.BytesIO(b"\x00\x01\x02"),
        )
        return FormData([("file", upload), ("model", "whisper-1")])

    monkeypatch.setattr(proxy_server, "get_form_data", _form_data)

    async def _llm_call():
        return {"text": "hello world"}

    async def _fake_route_request(*args, **kwargs):
        return _llm_call()

    monkeypatch.setattr(proxy_server, "route_request", _fake_route_request)
    yield


@pytest.fixture
def patched_transcription_error(monkeypatch, patched_transcription):
    async def _raise(*args, **kwargs):
        raise ValueError("transcription boom")

    monkeypatch.setattr(proxy_server, "route_request", _raise)
    yield


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_audio_speech_happy_path(client, auth_as, patched_speech, path):
    """Pins ``POST /v1/audio/speech`` and ``POST /audio/speech`` (happy)."""
    payload = {"model": "tts-1", "input": "Hi", "voice": "alloy"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    response_summary = {
        "status_code": response.status_code,
        "content_type": response.headers.get("content-type", ""),
        "body_bytes": response.content,
    }
    assert response_summary == {
        "status_code": 200,
        "content_type": "audio/mpeg",
        "body_bytes": b"\x00\x01\x02",
    }


@pytest.mark.parametrize(
    ("patched_speech", "response_format", "expected_content_type"),
    [
        ("audio/wav", "wav", "audio/wav"),
        ("audio/flac", "flac", "audio/flac"),
        ("audio/pcm", "pcm", "audio/pcm"),
        ("audio/wav", "mp3", "audio/wav"),
        ("application/json", "flac", "audio/flac"),
        (None, "wav", "audio/wav"),
        (None, None, "audio/mpeg"),
    ],
    indirect=["patched_speech"],
)
def test_audio_speech_content_type_matches_audio_format(
    client, auth_as, patched_speech, response_format, expected_content_type
):
    """Regression for LIT-6482: /v1/audio/speech mislabeled wav/flac/pcm as audio/mpeg."""
    payload = {
        "model": "tts-1",
        "input": "Hi",
        "voice": "alloy",
        **({} if response_format is None else {"response_format": response_format}),
    }
    with auth_as():
        response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 200
    assert response.headers.get("content-type", "").split(";")[0] == expected_content_type


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_audio_speech_error(client, auth_as, patched_speech_error, path):
    """Pins ``POST /v1/audio/speech`` and ``POST /audio/speech`` (error)."""
    payload = {"model": "tts-1", "input": "Hi", "voice": "alloy"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 500
    assert len(response.content) > 0


def test_audio_speech_bad_request_maps_to_400(client, auth_as, patched_speech_provider_rejection):
    """Regression for LIT-6501: a BadRequestError from the speech path surfaced as a generic 500."""
    payload = {"model": "gemini-tts", "input": "Hi", "voice": "Kore", "response_format": "mp3"}
    with auth_as():
        response = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 400
    error = response.json()["error"]
    assert "response_format='mp3'" in error["message"]
    assert "pcm" in error["message"]
    assert "wav" in error["message"]


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"])
def test_audio_transcription_happy_path(client, auth_as, patched_transcription, path):
    """Pins ``POST /v1/audio/transcriptions`` / ``POST /audio/transcriptions`` (happy)."""
    files = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data = {"model": "whisper-1"}
    with auth_as():
        response = client.post(path, files=files, data=data)
    assert response.status_code == 200
    body = response.json()
    assert body == {"text": "hello world"}
    response_summary = {
        "status_code": response.status_code,
        "text_field": body["text"],
        "media_type_hint": response.headers.get("content-type", "").split(";")[0],
    }
    assert response_summary == {
        "status_code": 200,
        "text_field": "hello world",
        "media_type_hint": "application/json",
    }


@pytest.mark.parametrize("path", ["/v1/audio/transcriptions", "/audio/transcriptions"])
def test_audio_transcription_error(client, auth_as, patched_transcription_error, path):
    """Pins ``POST /v1/audio/transcriptions`` / ``POST /audio/transcriptions`` (error)."""
    files = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data = {"model": "whisper-1"}
    with auth_as():
        response = client.post(path, files=files, data=data)
    assert response.status_code == 500
    assert len(response.content) > 0


class _RecordingAudioStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes], produced: list[bytes]):
        self._chunks = chunks
        self._produced = produced
        self.closed = False

    async def __aiter__(self):
        for chunk in self._chunks:
            self._produced.append(chunk)
            yield chunk

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_audio_speech_chunk_generator_forwards_an_unread_upstream_stream():
    """A streaming upstream response reaches the client piece by piece as it arrives, with no
    wait for a full AUDIO_SPEECH_CHUNK_SIZE before the first audio goes out."""
    produced: list[bytes] = []
    first, second = b"a" * (AUDIO_SPEECH_CHUNK_SIZE // 8), b"b" * AUDIO_SPEECH_CHUNK_SIZE
    stream = _RecordingAudioStream([first, second], produced)
    response = HttpxBinaryResponseContent(
        httpx.Response(status_code=200, headers={"content-type": "audio/mpeg"}, stream=stream)
    )

    chunks = proxy_server._audio_speech_chunk_generator(response)

    assert await anext(chunks) == first
    assert produced == [first]
    assert b"".join([chunk async for chunk in chunks]) == second
    assert stream.closed is True


def _speech_request(payload: dict) -> Request:
    body: Final = orjson.dumps(payload)

    async def _receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/audio/speech",
            "headers": [(b"content-type", b"application/json")],
            "query_string": b"",
        },
        receive=_receive,
    )


def _audio_response(stream: _RecordingAudioStream) -> HttpxBinaryResponseContent:
    return HttpxBinaryResponseContent(
        httpx.Response(status_code=200, headers={"content-type": "audio/mpeg"}, stream=stream)
    )


async def _serve_until_disconnect(response: StreamingResponse) -> None:
    """Drive the response through one ASGI cycle where the client drops at the first chunk."""

    async def _send(message):
        if message["type"] == "http.response.body":
            raise OSError("client gone")

    async def _receive():
        return {"type": "http.disconnect"}

    with contextlib.suppress(Exception):
        await response({"type": "http", "asgi": {"spec_version": "2.4"}}, _receive, _send)


@pytest.fixture
def patched_streaming_speech(monkeypatch):
    """Route /v1/audio/speech to an upstream whose audio body is left unread."""
    stream = _RecordingAudioStream([b"a" * AUDIO_SPEECH_CHUNK_SIZE], [])
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(return_value={}),
            update_request_status=AsyncMock(),
        ),
    )

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)

    async def _llm_call():
        return _audio_response(stream)

    async def _fake_route_request(*args, **kwargs):
        return _llm_call()

    monkeypatch.setattr(proxy_server, "route_request", _fake_route_request)
    yield stream


@pytest.mark.asyncio
async def test_audio_speech_releases_its_upstream_when_the_client_walks_away():
    """A client that drops mid-clip still has its upstream connection released. Starlette
    raises ClientDisconnect out of the response before any background task runs, so the
    body iterator's own cleanup is what has to cover this."""
    stream = _RecordingAudioStream([b"a" * AUDIO_SPEECH_CHUNK_SIZE] * 3, [])
    response = _audio_response(stream)

    await _serve_until_disconnect(
        StreamingResponse(
            proxy_server._audio_speech_chunk_generator(response),
            media_type="audio/mpeg",
            background=BackgroundTask(proxy_server._close_audio_speech_response, response),
        )
    )
    # The abandoned iterator is suspended mid-yield; it runs its cleanup once dropped.
    gc.collect()
    await asyncio.sleep(0.1)

    assert stream.closed is True


@pytest.mark.asyncio
async def test_audio_speech_route_releases_an_upstream_whose_body_is_never_read(
    monkeypatch, patched_streaming_speech
):
    """A response the server never iterates leaves the body iterator's cleanup unrun and
    out of reach of GC, so the route itself has to arm the release."""
    stream = patched_streaming_speech
    captured: list[StreamingResponse] = []

    original = proxy_server.StreamingResponse

    def _capture(*args, **kwargs):
        response = original(*args, **kwargs)
        captured.append(response)
        return response

    monkeypatch.setattr(proxy_server, "StreamingResponse", _capture)

    request = _speech_request({"model": "tts-1", "input": "Hi", "voice": "alloy", "stream": True})
    await proxy_server.audio_speech(
        request=request, fastapi_response=Response(), user_api_key_dict=UserAPIKeyAuth(api_key="sk-test")
    )

    assert len(captured) == 1
    assert stream.closed is False

    await captured[0].background()

    assert stream.closed is True
