"""Behavior pins for ``proxy_server.py`` audio routes.

Pins (PR2):
    - POST /v1/audio/speech
    - POST /audio/speech
    - POST /v1/audio/transcriptions
    - POST /audio/transcriptions
"""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.proxy import proxy_server


@pytest.fixture
def patched_speech(monkeypatch):
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

    class _FakeBinaryResp:
        async def aiter_bytes(self, chunk_size: int = 8192):
            async def _gen():
                yield b"\x00\x01\x02"

            return _gen()

    async def _llm_call():
        return _FakeBinaryResp()

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
    monkeypatch.setattr(proxy_server, "check_file_size_under_limit", lambda **kwargs: True)

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


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_audio_speech_error(client, auth_as, patched_speech_error, path):
    """Pins ``POST /v1/audio/speech`` and ``POST /audio/speech`` (error)."""
    payload = {"model": "tts-1", "input": "Hi", "voice": "alloy"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 500
    assert len(response.content) > 0


def _patch_speech_streaming(monkeypatch, frames, content_type):
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

    from litellm.types.llms.openai import SpeechStreamingResponse

    async def _frames():
        for frame in frames:
            yield frame

    headers = {"content-type": content_type} if content_type is not None else {}

    async def _llm_call():
        return SpeechStreamingResponse(stream_iterator=_frames(), headers=headers)

    async def _fake_route_request(*args, **kwargs):
        return _llm_call()

    monkeypatch.setattr(proxy_server, "route_request", _fake_route_request)


@pytest.fixture
def patched_speech_sse(monkeypatch):
    _patch_speech_streaming(
        monkeypatch,
        frames=[
            b'data: {"type":"speech.audio.delta","audio":"AAA"}\n\n',
            b'data: {"type":"speech.audio.done"}\n\n',
        ],
        content_type="text/event-stream; charset=utf-8",
    )
    yield


@pytest.fixture
def patched_speech_ignored_stream_format(monkeypatch):
    _patch_speech_streaming(
        monkeypatch,
        frames=[b"ID3\x04mp3-bytes"],
        content_type="audio/mpeg",
    )
    yield


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_audio_speech_sse_streaming(client, auth_as, patched_speech_sse, path):
    """Streaming TTS (stream_format='sse') forwards speech.audio.delta frames verbatim as
    text/event-stream, instead of re-chunking a buffered audio/mpeg blob (regression #33974)."""
    payload = {
        "model": "gpt-4o-mini-tts",
        "input": "Hi",
        "voice": "alloy",
        "stream_format": "sse",
    }
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("text/event-stream")
    body = response.content
    assert b"speech.audio.delta" in body
    assert b"speech.audio.done" in body


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_audio_speech_streaming_honors_provider_content_type(
    client, auth_as, patched_speech_ignored_stream_format, path
):
    """When the provider ignores stream_format and returns audio/mpeg (e.g. tts-1), the proxy
    must forward that content-type verbatim rather than mislabeling the bytes as SSE."""
    payload = {
        "model": "tts-1",
        "input": "Hi",
        "voice": "alloy",
        "stream_format": "sse",
    }
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert response.headers.get("content-type", "").startswith("audio/mpeg")
    assert response.content == b"ID3\x04mp3-bytes"


def test_audio_speech_requests_upstream_streaming(client, auth_as, monkeypatch):
    """The proxy must ask the handler to stream the upstream (stream_audio=True) even for a plain
    request with no stream_format, so it matches OpenAI's always-chunked /v1/audio/speech and does
    not buffer the full clip. Named stream_audio (not stream) so TTS cost tracking is not skipped."""
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

    from litellm.types.llms.openai import SpeechStreamingResponse

    captured: dict = {}

    async def _frames():
        yield b"audio-bytes"

    async def _llm_call():
        return SpeechStreamingResponse(stream_iterator=_frames(), headers={"content-type": "audio/mpeg"})

    async def _fake_route_request(*args, **kwargs):
        captured.update(kwargs.get("data", {}))
        return _llm_call()

    monkeypatch.setattr(proxy_server, "route_request", _fake_route_request)

    payload = {"model": "gpt-4o-mini-tts", "input": "Hi", "voice": "alloy"}
    with auth_as():
        response = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 200
    assert captured.get("stream_audio") is True
    assert "stream" not in captured  # must not be "stream" (would skip cost tracking)
    assert response.headers.get("content-type", "").startswith("audio/mpeg")
    assert response.content == b"audio-bytes"


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
