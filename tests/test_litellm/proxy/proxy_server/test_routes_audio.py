"""Behavior pins for ``proxy_server.py`` audio routes.

Pins (PR2):
    - POST /v1/audio/speech
    - POST /audio/speech
    - POST /v1/audio/transcriptions
    - POST /audio/transcriptions
"""

from __future__ import annotations

import asyncio
import io
import json
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.proxy import proxy_server
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


def test_audio_speech_failure_hook_receives_call_type(client, auth_as, patched_speech_error):
    """LIT-41521: failed /v1/audio/speech calls pass call_type='aspeech' and litellm_call_id to failure hook."""
    payload: Final = {"model": "tts-1", "input": "Hi", "voice": "alloy"}
    with auth_as():
        response: Final = client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 500
    call_id: Final = response.headers.get("x-litellm-call-id")
    assert call_id is not None
    proxy_server.proxy_logging_obj.post_call_failure_hook.assert_called_once()
    call_kwargs: Final = proxy_server.proxy_logging_obj.post_call_failure_hook.call_args.kwargs
    request_data: Final = call_kwargs["request_data"]
    assert request_data["call_type"] == "aspeech"
    assert request_data["litellm_call_id"] == call_id
    assert request_data["model"] == "tts-1"


def test_audio_transcription_failure_hook_receives_call_type(client, auth_as, patched_transcription_error):
    """LIT-41521: failed /v1/audio/transcriptions calls pass call_type='atranscription' to failure hook."""
    files: Final = {"file": ("audio.mp3", b"\x00\x01\x02", "audio/mpeg")}
    data: Final = {"model": "whisper-1"}
    with auth_as():
        response: Final = client.post("/v1/audio/transcriptions", files=files, data=data)
    assert response.status_code == 500
    call_id: Final = response.headers.get("x-litellm-call-id")
    assert call_id is not None
    proxy_server.proxy_logging_obj.post_call_failure_hook.assert_called_once()
    call_kwargs: Final = proxy_server.proxy_logging_obj.post_call_failure_hook.call_args.kwargs
    request_data: Final = call_kwargs["request_data"]
    assert request_data["call_type"] == "atranscription"
    assert request_data["litellm_call_id"] == call_id
    assert request_data["model"] == "whisper-1"


def test_audio_speech_failure_spend_log_recorded(client, auth_as, monkeypatch):
    """LIT-41521: failed audio/speech calls write a failure row with call_type='aspeech' to spend logs."""
    import litellm
    from litellm.proxy.hooks.proxy_track_cost_callback import _ProxyDBLogger
    from litellm.proxy.utils import ProxyLogging

    mock_prisma: Final = MagicMock()
    mock_prisma.spend_log_transactions = []  # mutable-ok: in-memory spend transactions list
    mock_prisma._spend_log_transactions_lock = asyncio.Lock()
    monkeypatch.setattr(proxy_server, "prisma_client", mock_prisma)
    monkeypatch.setattr(proxy_server, "disable_spend_logs", False)
    monkeypatch.setattr(proxy_server, "llm_router", MagicMock())

    async def _add_data(data, **kwargs):
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)

    async def _raise(*args, **kwargs):
        raise litellm.BadRequestError(
            message="AzureException - Voice not supported",
            model="tts-1",
            llm_provider="azure",
        )

    monkeypatch.setattr(proxy_server, "route_request", _raise)

    db_logger: Final = _ProxyDBLogger()
    monkeypatch.setattr(litellm, "callbacks", [db_logger])
    monkeypatch.setattr(proxy_server, "proxy_logging_obj", ProxyLogging(user_api_key_cache=MagicMock()))

    payload: Final = {"model": "tts-1", "input": "Hello", "voice": "invalid-voice"}
    with auth_as(api_key="sk-test-speech-key", user_id="u-speech-1"):
        response: Final = client.post("/v1/audio/speech", json=payload)

    assert response.status_code == 400
    call_id: Final = response.headers.get("x-litellm-call-id")
    assert call_id is not None
    assert len(mock_prisma.spend_log_transactions) == 1
    row: Final = mock_prisma.spend_log_transactions[0]
    assert row["request_id"] == call_id
    assert row["call_type"] == "aspeech"
    assert row["model"] == "tts-1"
    metadata: Final = json.loads(row["metadata"]) if isinstance(row["metadata"], str) else row["metadata"]
    assert metadata["status"] == "failure"
    assert metadata["error_information"]["error_class"] == "BadRequestError"
    assert "Voice not supported" in metadata["error_information"]["error_message"]
