import json
import os
import sys
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.volcengine.text_to_speech.protocol import (
    COMP_NONE,
    EV_CONNECTION_FINISHED,
    EV_CONNECTION_STARTED,
    EV_SESSION_FINISHED,
    EV_SESSION_STARTED,
    EV_TTS_RESPONSE,
    FLAG_EVENT,
    MSG_AUDIO_SERVER,
    MSG_FULL_SERVER,
    SER_JSON,
    decode_event_frame,
    encode_event_frame,
    encode_json_event,
)
from litellm.llms.volcengine.text_to_speech.transformation import (
    VOLCENGINE_TTS_DEFAULT_VOICE,
    VolcEngineTextToSpeechConfig,
    pick_tts_request_model,
    pick_tts_resource_id,
)


class FakeConnect:
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


class FakeWebSocket:
    def __init__(self, incoming):
        self.incoming = list(incoming)
        self.sent = []

    async def send(self, data):
        self.sent.append(data)

    async def recv(self):
        if not self.incoming:
            raise AssertionError("unexpected recv")
        return self.incoming.pop(0)


def _server_event(event: int) -> bytes:
    return encode_event_frame(
        message_type=MSG_FULL_SERVER,
        flags=FLAG_EVENT,
        serialization=SER_JSON,
        compression=COMP_NONE,
        event=event,
        payload=b"{}",
    )


def _server_audio(payload: bytes) -> bytes:
    return encode_event_frame(
        message_type=MSG_AUDIO_SERVER,
        flags=FLAG_EVENT,
        serialization=0,
        compression=COMP_NONE,
        event=EV_TTS_RESPONSE,
        payload=payload,
    )


async def _read_binary_response(response) -> bytes:
    chunks = []
    generator = await response.aiter_bytes(chunk_size=8192)
    async for chunk in generator:
        chunks.append(chunk)
    return b"".join(chunks)


def test_tts_event_frame_round_trip():
    frame = decode_event_frame(
        encode_json_event(event=100, session_id="session-1", payload={"ok": True})
    )

    assert frame.message_type == 1
    assert frame.flags == FLAG_EVENT
    assert frame.event == 100
    assert frame.session_id == "session-1"
    assert frame.payload == b'{"ok": true}'


def test_tts_voice_fallback_and_resource_id_mapping():
    config = VolcEngineTextToSpeechConfig()

    voice, optional_params = config.map_openai_params(
        model="seed-tts-2.0",
        optional_params={},
        voice="alloy",
    )

    assert voice == VOLCENGINE_TTS_DEFAULT_VOICE
    assert optional_params["response_format"] == "pcm"
    for openai_voice in ("cedar", "marin", "verse"):
        voice, _ = config.map_openai_params(
            model="seed-tts-2.0",
            optional_params={},
            voice=openai_voice,
        )
        assert voice == VOLCENGINE_TTS_DEFAULT_VOICE
    assert pick_tts_resource_id("seed-tts-1.0-concurr") == "seed-tts-1.0-concurr"
    assert pick_tts_resource_id("seed-icl-2.0") == "seed-icl-2.0"
    assert pick_tts_resource_id("seed-tts-2.0") == "seed-tts-2.0"
    assert pick_tts_resource_id("seed-tts-2.0-expressive") == "seed-tts-2.0"
    assert (
        pick_tts_request_model("volcengine/seed-tts-2.0-expressive", {})
        == "seed-tts-2.0-expressive"
    )
    assert pick_tts_request_model("seed-tts-2.0", {}) is None


@pytest.mark.asyncio
async def test_tts_dispatch_uses_volcengine_ws_and_returns_pcm(monkeypatch):
    pcm = b"\x01\x00\x02\x00"
    fake_ws = FakeWebSocket(
        [
            _server_event(EV_CONNECTION_STARTED),
            _server_event(EV_SESSION_STARTED),
            _server_audio(pcm),
            _server_event(EV_SESSION_FINISHED),
            _server_event(EV_CONNECTION_FINISHED),
        ]
    )
    fake_connect = FakeConnect(fake_ws)
    monkeypatch.setattr("websockets.connect", fake_connect)

    response = await VolcEngineTextToSpeechConfig().dispatch_text_to_speech(
        model="seed-tts-2.0",
        input="你好",
        voice=VOLCENGINE_TTS_DEFAULT_VOICE,
        optional_params={
            "response_format": "pcm",
            "resource_id": "attacker-resource",
        },
        litellm_params_dict={
            "metadata": {
                "api_base": "wss://openspeech.bytedance.com/api/v3/tts/bidirection?deployment=test"
            },
            "resource_id": "seed-tts-2.0-configured",
        },
        logging_obj=MagicMock(),
        timeout=1,
        extra_headers=None,
        aspeech=True,
        api_base="wss://attacker.test/tts",
        api_key="speech-api-key",
    )

    assert await _read_binary_response(response) == pcm
    assert response._hidden_params["content_type"] == "audio/pcm"
    assert fake_connect.args == (
        "wss://openspeech.bytedance.com/api/v3/tts/bidirection?deployment=test",
    )
    assert fake_connect.kwargs["additional_headers"]["X-Api-Key"] == "speech-api-key"
    assert "X-Api-App-Id" not in fake_connect.kwargs["additional_headers"]
    assert "X-Api-Access-Key" not in fake_connect.kwargs["additional_headers"]
    assert (
        fake_connect.kwargs["additional_headers"]["X-Api-Resource-Id"]
        == "seed-tts-2.0-configured"
    )
    assert decode_event_frame(fake_ws.sent[0]).event == 1
    assert decode_event_frame(fake_ws.sent[1]).event == 100
    assert decode_event_frame(fake_ws.sent[2]).event == 200


@pytest.mark.asyncio
async def test_tts_dispatch_maps_official_2_0_model_name(monkeypatch):
    fake_ws = FakeWebSocket(
        [
            _server_event(EV_CONNECTION_STARTED),
            _server_event(EV_SESSION_STARTED),
            _server_audio(b"\x01\x00"),
            _server_event(EV_SESSION_FINISHED),
            _server_event(EV_CONNECTION_FINISHED),
        ]
    )
    fake_connect = FakeConnect(fake_ws)
    monkeypatch.setattr("websockets.connect", fake_connect)

    await VolcEngineTextToSpeechConfig().dispatch_text_to_speech(
        model="seed-tts-2.0-expressive",
        input="你好",
        voice=VOLCENGINE_TTS_DEFAULT_VOICE,
        optional_params={
            "response_format": "pcm",
            "resource_id": "attacker-resource",
        },
        litellm_params_dict={},
        logging_obj=MagicMock(),
        timeout=1,
        extra_headers=None,
        aspeech=True,
        api_base="wss://openspeech.bytedance.com/api/v3/tts/bidirection",
        api_key="speech-api-key",
    )

    assert (
        fake_connect.kwargs["additional_headers"]["X-Api-Resource-Id"] == "seed-tts-2.0"
    )
    start_session_payload = json.loads(decode_event_frame(fake_ws.sent[1]).payload)
    assert start_session_payload["req_params"]["model"] == "seed-tts-2.0-expressive"


def test_tts_get_complete_url_ignores_request_api_base():
    config = VolcEngineTextToSpeechConfig()

    assert (
        config.get_complete_url(
            model="seed-tts-2.0",
            api_base="wss://attacker.test/tts",
            litellm_params={
                "metadata": {
                    "api_base": "wss://openspeech.bytedance.com/api/v3/tts/bidirection?deployment=test"
                }
            },
        )
        == "wss://openspeech.bytedance.com/api/v3/tts/bidirection?deployment=test"
    )
    assert config.get_complete_url(
        model="seed-tts-2.0",
        api_base="wss://attacker.test/tts",
        litellm_params={},
    ).startswith("wss://openspeech.bytedance.com/")
    assert config.get_complete_url(
        model="seed-tts-2.0",
        api_base="wss://attacker.test/tts",
        litellm_params={
            "api_base": "wss://attacker.test/tts",
            "metadata": {"api_base": "wss://attacker.test/tts"},
        },
    ).startswith("wss://openspeech.bytedance.com/")
