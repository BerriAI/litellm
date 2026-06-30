"""Behavior pins for ``proxy_server.py`` audio routes.

Pins (PR2):
    - POST /v1/audio/speech
    - POST /audio/speech
    - POST /v1/audio/transcriptions
    - POST /audio/transcriptions
"""

from __future__ import annotations

import gzip
import io
import json
import wave
from array import array
from unittest.mock import AsyncMock, MagicMock

import litellm
import pytest

from litellm.llms.volcengine.audio_transcription.sauc_protocol import (
    COMP_GZIP,
    FLAG_NEG_SEQ,
    MSG_FULL_SERVER,
    SER_JSON,
    encode_sauc_frame,
)
from litellm.llms.volcengine.text_to_speech.protocol import (
    COMP_NONE,
    EV_CONNECTION_FINISHED,
    EV_CONNECTION_STARTED,
    EV_START_CONNECTION,
    EV_START_SESSION,
    EV_SESSION_FINISHED,
    EV_SESSION_STARTED,
    EV_TASK_REQUEST,
    EV_TTS_RESPONSE,
    FLAG_EVENT,
    MSG_AUDIO_SERVER,
    MSG_FULL_SERVER as TTS_MSG_FULL_SERVER,
    SER_JSON as TTS_SER_JSON,
    decode_event_frame,
    encode_event_frame,
)
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
def patched_speech_pcm(monkeypatch, patched_speech):
    class _FakeBinaryResp:
        _hidden_params = {"content_type": "audio/pcm"}

        async def aiter_bytes(self, chunk_size: int = 8192):
            async def _gen():
                yield b"\x00\x01"

            return _gen()

    async def _fake_route_request(*args, **kwargs):
        async def _llm_call():
            return _FakeBinaryResp()

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


@pytest.fixture
def patched_volcengine_proxy_runtime(monkeypatch):
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
    monkeypatch.setattr(
        proxy_server, "check_file_size_under_limit", lambda **kwargs: True
    )

    async def _add_data(data, **kwargs):
        logging_obj = MagicMock()
        logging_obj.model_call_details = {}
        data["litellm_logging_obj"] = logging_obj
        data["litellm_call_id"] = "test-call-id"
        data["metadata"] = data.get("metadata") or {}
        data["proxy_server_request"] = {"url": "http://testserver", "headers": {}}
        return data

    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", _add_data)


class _FakeConnect:
    def __init__(self, ws):
        self.ws = ws
        self.args = None
        self.kwargs = None

    def __call__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        return self

    async def __aenter__(self):
        return self.ws

    async def __aexit__(self, *args):
        return False


class _FakeWebSocket:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if not self.incoming:
            raise AssertionError("unexpected recv")
        return self.incoming.pop(0)


def _wav_bytes() -> bytes:
    samples = array("h", [0, 256, -256, 512, -512] * 80)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(samples.tobytes())
    return buf.getvalue()


def _sauc_transcript_frame(text: str) -> bytes:
    payload = gzip.compress(
        json.dumps(
            {"result": {"utterances": [{"text": text, "definite": True}]}}
        ).encode("utf-8")
    )
    return encode_sauc_frame(
        message_type=MSG_FULL_SERVER,
        flags=FLAG_NEG_SEQ,
        serialization=SER_JSON,
        compression=COMP_GZIP,
        sequence=-1,
        payload=payload,
    )


def _tts_event(event: int) -> bytes:
    return encode_event_frame(
        message_type=TTS_MSG_FULL_SERVER,
        flags=FLAG_EVENT,
        serialization=TTS_SER_JSON,
        compression=COMP_NONE,
        event=event,
        payload=b"{}",
    )


def _tts_audio(payload: bytes) -> bytes:
    return encode_event_frame(
        message_type=MSG_AUDIO_SERVER,
        flags=FLAG_EVENT,
        serialization=0,
        compression=COMP_NONE,
        event=EV_TTS_RESPONSE,
        payload=payload,
    )


def _set_volcengine_router(monkeypatch, model_list):
    router = litellm.Router(model_list=model_list)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    return router


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
def test_audio_speech_uses_provider_content_type(
    client, auth_as, patched_speech_pcm, path
):
    payload = {
        "model": "seed-tts-2.0",
        "input": "Hi",
        "voice": "zh_female_vv_uranus_bigtts",
        "response_format": "pcm",
    }
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 200
    assert response.headers.get("content-type") == "audio/pcm"
    assert response.content == b"\x00\x01"


@pytest.mark.parametrize("path", ["/v1/audio/speech", "/audio/speech"])
def test_audio_speech_error(client, auth_as, patched_speech_error, path):
    """Pins ``POST /v1/audio/speech`` and ``POST /audio/speech`` (error)."""
    payload = {"model": "tts-1", "input": "Hi", "voice": "alloy"}
    with auth_as():
        response = client.post(path, json=payload)
    assert response.status_code == 500
    assert len(response.content) > 0


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


def test_audio_transcription_routes_alias_to_volcengine_websocket(
    client, auth_as, monkeypatch, patched_volcengine_proxy_runtime
):
    fake_ws = _FakeWebSocket([_sauc_transcript_frame("volcengine transcript ok")])
    fake_connect = _FakeConnect(fake_ws)
    monkeypatch.setattr("websockets.connect", fake_connect)
    router = _set_volcengine_router(
        monkeypatch,
        [
            {
                "model_name": "volc.seedasr.sauc.duration",
                "litellm_params": {
                    "model": "volcengine/volc.seedasr.sauc.duration",
                    "api_key": "speech-api-key",
                    "api_base": "wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",
                },
            }
        ],
    )

    files = {"file": ("audio.wav", _wav_bytes(), "audio/wav")}
    data = {
        "model": "volc.seedasr.sauc.duration",
        "response_format": "json",
        "language": "zh",
        "api_base": "wss://attacker.test/asr",
    }
    try:
        with auth_as():
            response = client.post("/v1/audio/transcriptions", files=files, data=data)
    finally:
        router.discard()

    assert response.status_code == 200, response.text
    assert response.json()["text"] == "volcengine transcript ok"
    assert fake_connect.args == ("wss://openspeech.bytedance.com/api/v3/sauc/bigmodel",)
    assert fake_connect.kwargs["additional_headers"]["X-Api-Key"] == "speech-api-key"
    assert (
        fake_connect.kwargs["additional_headers"]["X-Api-Resource-Id"]
        == "volc.seedasr.sauc.duration"
    )
    assert len(fake_ws.sent) == 3


def test_audio_speech_routes_alias_to_volcengine_websocket(
    client, auth_as, monkeypatch, patched_volcengine_proxy_runtime
):
    pcm = b"\x01\x00\x02\x00"
    fake_ws = _FakeWebSocket(
        [
            _tts_event(EV_CONNECTION_STARTED),
            _tts_event(EV_SESSION_STARTED),
            _tts_audio(pcm),
            _tts_event(EV_SESSION_FINISHED),
            _tts_event(EV_CONNECTION_FINISHED),
        ]
    )
    fake_connect = _FakeConnect(fake_ws)
    monkeypatch.setattr("websockets.connect", fake_connect)
    router = _set_volcengine_router(
        monkeypatch,
        [
            {
                "model_name": "seed-tts-2.0",
                "litellm_params": {
                    "model": "volcengine/seed-tts-2.0",
                    "api_key": "speech-api-key",
                    "api_base": "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
                    "resource_id": "seed-tts-2.0",
                },
            }
        ],
    )

    payload = {
        "model": "seed-tts-2.0",
        "input": "hello",
        "voice": "alloy",
        "response_format": "pcm",
        "api_base": "wss://attacker.test/tts",
        "metadata": {"api_base": "wss://attacker.test/tts"},
    }
    try:
        with auth_as():
            response = client.post("/v1/audio/speech", json=payload)
    finally:
        router.discard()

    assert response.status_code == 200, response.text
    assert response.headers.get("content-type") == "audio/pcm"
    assert response.content == pcm
    assert fake_connect.args == (
        "wss://openspeech.bytedance.com/api/v3/tts/bidirection",
    )
    assert fake_connect.kwargs["additional_headers"]["X-Api-Key"] == "speech-api-key"
    assert (
        fake_connect.kwargs["additional_headers"]["X-Api-Resource-Id"] == "seed-tts-2.0"
    )
    assert [decode_event_frame(frame).event for frame in fake_ws.sent[:3]] == [
        EV_START_CONNECTION,
        EV_START_SESSION,
        EV_TASK_REQUEST,
    ]
