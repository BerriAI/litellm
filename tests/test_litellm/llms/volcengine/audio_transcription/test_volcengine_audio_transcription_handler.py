import gzip
import json
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.volcengine import get_volcengine_speech_api_key
from litellm.llms.volcengine.audio_transcription import handler as handler_mod
from litellm.llms.volcengine.audio_transcription.handler import (
    VolcEngineAudioTranscription,
)
from litellm.llms.volcengine.audio_transcription.sauc_protocol import (
    COMP_GZIP,
    FLAG_NEG_SEQ,
    MSG_FULL_SERVER,
    SER_JSON,
    encode_sauc_frame,
)
from litellm.llms.volcengine.audio_transcription.transformation import (
    VolcEngineAudioTranscriptionConfig,
    pick_stt_resource_id,
)
from litellm.llms.volcengine.common_utils import VolcEngineError
from litellm.types.utils import TranscriptionResponse


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


def _server_transcript_frame(text: str) -> bytes:
    payload = gzip.compress(
        json.dumps(
            {
                "result": {
                    "utterances": [
                        {"text": text, "definite": True},
                    ]
                }
            }
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


def test_speech_key_accepts_single_api_key_without_leaking_secret():
    assert get_volcengine_speech_api_key("speech-api-key") == "speech-api-key"

    with pytest.raises(VolcEngineError) as exc_info:
        get_volcengine_speech_api_key("app-id:access-key")

    assert "app-id" not in exc_info.value.message
    assert "access-key" not in exc_info.value.message
    assert "single Speech API Key" in exc_info.value.message


def test_stt_resource_id_mapping_uses_official_model_names():
    assert (
        pick_stt_resource_id("volcengine/volc.seedasr.sauc.duration")
        == "volc.seedasr.sauc.duration"
    )
    assert (
        pick_stt_resource_id("volc.bigasr.sauc.concurrent")
        == "volc.bigasr.sauc.concurrent"
    )
    assert pick_stt_resource_id("unknown-model") == "volc.bigasr.sauc.duration"


@pytest.mark.asyncio
async def test_async_transcription_uses_volcengine_ws_and_returns_text(monkeypatch):
    fake_ws = FakeWebSocket([_server_transcript_frame("你好世界")])
    fake_connect = FakeConnect(fake_ws)
    monkeypatch.setattr("websockets.connect", fake_connect)
    monkeypatch.setattr(
        handler_mod,
        "resample_to_volcengine_stt_pcm",
        lambda _: SimpleNamespace(
            pcm_bytes=b"\x01\x00" * 3200,
            duration_seconds=0.2,
            sample_rate_hz=16000,
            num_channels=1,
        ),
    )

    response = await VolcEngineAudioTranscription().async_audio_transcriptions(
        model="volc.seedasr.sauc.duration",
        audio_file=b"fake-wav",
        optional_params={
            "response_format": "json",
            "language": "zh",
            "resource_id": "attacker-resource",
        },
        litellm_params={
            "metadata": {"api_base": "wss://example.test/asr"},
            "resource_id": "volc.seedasr.sauc.duration",
        },
        model_response=TranscriptionResponse(),
        timeout=None,  # type: ignore[arg-type]
        logging_obj=MagicMock(),
        api_key="speech-api-key",
        api_base="wss://attacker.test/asr",
        provider_config=VolcEngineAudioTranscriptionConfig(),
    )

    assert response.text == "你好世界"
    assert response._hidden_params["custom_llm_provider"] == "volcengine"
    assert response._hidden_params["audio_transcription_duration"] == 0.2
    assert fake_connect.args == ("wss://example.test/asr",)
    assert fake_connect.kwargs["additional_headers"]["X-Api-Key"] == "speech-api-key"
    assert "X-Api-App-Key" not in fake_connect.kwargs["additional_headers"]
    assert "X-Api-Access-Key" not in fake_connect.kwargs["additional_headers"]
    assert (
        fake_connect.kwargs["additional_headers"]["X-Api-Resource-Id"]
        == "volc.seedasr.sauc.duration"
    )
    assert len(fake_ws.sent) == 3


def test_stt_get_complete_url_ignores_request_api_base():
    config = VolcEngineAudioTranscriptionConfig()

    assert (
        config.get_complete_url(
            api_base="wss://attacker.test/asr",
            api_key="speech-api-key",
            model="volc.seedasr.sauc.duration",
            optional_params={},
            litellm_params={"metadata": {"api_base": "wss://example.test/asr"}},
        )
        == "wss://example.test/asr"
    )
    assert config.get_complete_url(
        api_base="wss://attacker.test/asr",
        api_key="speech-api-key",
        model="volc.seedasr.sauc.duration",
        optional_params={},
        litellm_params={},
    ).startswith("wss://openspeech.bytedance.com/")
