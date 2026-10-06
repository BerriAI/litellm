import base64
from datetime import datetime
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig
from litellm.llms.mistral.audio_speech.transformation import (
    MistralTextToSpeechConfig,
    MistralTextToSpeechException,
)
from litellm.utils import ProviderConfigManager

SPEECH_URL: Final = "https://api.mistral.ai/v1/audio/speech"
SPEECH_MODEL: Final = "voxtral-mini-tts-2603"
ENCODED_AUDIO: Final = "UklGRg=="


def test_mistral_text_to_speech_config_installed():
    config: Final = ProviderConfigManager.get_provider_text_to_speech_config(
        model="voxtral-mini-tts-2603",
        provider=litellm.LlmProviders.MISTRAL,
    )
    assert isinstance(config, BaseTextToSpeechConfig)
    assert isinstance(config, MistralTextToSpeechConfig)


def test_map_openai_params_drops_speed_and_instructions():
    config: Final = MistralTextToSpeechConfig()
    voice, params = config.map_openai_params(
        model="voxtral-mini-tts-2603",
        optional_params={"response_format": "wav", "speed": 1.5, "instructions": "sound cheerful"},
        voice="en_paul_neutral",
    )
    assert voice == "en_paul_neutral"
    assert params == {"response_format": "wav"}


def test_map_openai_params_accepts_voice_dict_and_ref_audio():
    config: Final = MistralTextToSpeechConfig()
    voice, params = config.map_openai_params(
        model="voxtral-mini-tts-2603",
        optional_params={},
        voice={"voice_id": "1f3a8b0c-voice-uuid"},
        kwargs={"ref_audio": "bXktdm9pY2Utc2FtcGxl"},
    )
    assert voice == "1f3a8b0c-voice-uuid"
    assert params == {"ref_audio": "bXktdm9pY2Utc2FtcGxl"}


def test_transform_request_builds_mistral_body():
    config: Final = MistralTextToSpeechConfig()
    data: Final = config.transform_text_to_speech_request(
        model="voxtral-mini-tts-2603",
        input="hello from litellm",
        voice="en_paul_neutral",
        optional_params={"response_format": "wav"},
        litellm_params={},
        headers={},
    )
    assert data["dict_body"] == {
        "model": "voxtral-mini-tts-2603",
        "input": "hello from litellm",
        "voice_id": "en_paul_neutral",
        "response_format": "wav",
    }
    assert data["headers"] == {"Content-Type": "application/json"}


def test_transform_request_omits_voice_for_ref_audio_cloning():
    config: Final = MistralTextToSpeechConfig()
    data: Final = config.transform_text_to_speech_request(
        model="voxtral-mini-tts-2603",
        input="clone me",
        voice=None,
        optional_params={"ref_audio": "bXktdm9pY2Utc2FtcGxl"},
        litellm_params={},
        headers={},
    )
    assert data["dict_body"] == {
        "model": "voxtral-mini-tts-2603",
        "input": "clone me",
        "ref_audio": "bXktdm9pY2Utc2FtcGxl",
    }


def test_get_complete_url_default_base():
    config: Final = MistralTextToSpeechConfig()
    url: Final = config.get_complete_url(model="voxtral-mini-tts-2603", api_base=None, litellm_params={})
    assert url == SPEECH_URL


@pytest.mark.parametrize(
    "api_base",
    ["https://custom.api.example.com/v1/", "https://custom.api.example.com/v1", "https://custom.api.example.com"],
)
def test_get_complete_url_custom_base_always_versioned(api_base: str):
    config: Final = MistralTextToSpeechConfig()
    url: Final = config.get_complete_url(model="voxtral-mini-tts-2603", api_base=api_base, litellm_params={})
    assert url == "https://custom.api.example.com/v1/audio/speech"


def test_validate_environment_sets_bearer_header():
    config: Final = MistralTextToSpeechConfig()
    headers: Final = config.validate_environment(
        headers={"x-custom": "1"},
        model="voxtral-mini-tts-2603",
        api_key="sk-mistral-test",
    )
    assert headers == {
        "x-custom": "1",
        "Authorization": "Bearer sk-mistral-test",
        "Content-Type": "application/json",
    }


def test_validate_environment_requires_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    config: Final = MistralTextToSpeechConfig()
    with pytest.raises(MistralTextToSpeechException, match="MISTRAL_API_KEY"):
        config.validate_environment(headers={}, model="voxtral-mini-tts-2603")


def test_transform_response_decodes_base64_audio():
    config: Final = MistralTextToSpeechConfig()
    audio_bytes: Final = b"RIFF-fake-wav-bytes"
    raw_response: Final = httpx.Response(
        200,
        json={"audio_data": base64.b64encode(audio_bytes).decode()},
        headers={"x-request-id": "req-123"},
        request=httpx.Request(
            "POST",
            SPEECH_URL,
            json={"model": "voxtral-mini-tts-2603", "input": "hi", "response_format": "wav"},
        ),
    )
    result: Final = config.transform_text_to_speech_response(
        model="voxtral-mini-tts-2603",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )
    assert result.content == audio_bytes
    assert result.response.headers["content-type"] == "audio/wav"
    assert result.response.headers["content-length"] == str(len(audio_bytes))
    assert result.response.headers["x-request-id"] == "req-123"


def test_transform_response_missing_audio_data_raises():
    config: Final = MistralTextToSpeechConfig()
    raw_response: Final = httpx.Response(
        200,
        json={"detail": "unexpected"},
        request=httpx.Request("POST", SPEECH_URL, json={"model": "voxtral-mini-tts-2603", "input": "hi"}),
    )
    with pytest.raises(MistralTextToSpeechException, match="audio_data"):
        config.transform_text_to_speech_response(
            model="voxtral-mini-tts-2603",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )


def test_map_openai_params_maps_openai_voice_aliases():
    config: Final = MistralTextToSpeechConfig()
    alloy_voice, _ = config.map_openai_params(
        model="voxtral-mini-tts-2603",
        optional_params={},
        voice="alloy",
    )
    nova_voice, _ = config.map_openai_params(
        model="voxtral-mini-tts-2603",
        optional_params={},
        voice="Nova",
    )
    passthrough_voice, _ = config.map_openai_params(
        model="voxtral-mini-tts-2603",
        optional_params={},
        voice="en_paul_happy",
    )
    assert alloy_voice == "en_paul_neutral"
    assert nova_voice == "gb_jane_sarcasm"
    assert passthrough_voice == "en_paul_happy"


def test_transform_response_invalid_base64_raises():
    config: Final = MistralTextToSpeechConfig()
    raw_response: Final = httpx.Response(
        status_code=200,
        json={"audio_data": "QUJD!QUJD"},
        request=httpx.Request("POST", SPEECH_URL),
    )
    with pytest.raises(MistralTextToSpeechException, match="base64"):
        config.transform_text_to_speech_response(
            model="voxtral-mini-tts-2603",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )


def _speech_logging() -> Logging:
    return Logging(
        model=SPEECH_MODEL,
        messages=[{"role": "user", "content": "hi"}],
        stream=False,
        call_type="speech",
        start_time=datetime(2026, 1, 1),
        litellm_call_id="mistral-speech-call",
        function_id="mistral-speech-function",
    )


@pytest.mark.parametrize(
    ("request_body", "expected_content_type"),
    [
        ({"model": SPEECH_MODEL, "input": "hi"}, "audio/mpeg"),
        ({"model": SPEECH_MODEL, "input": "hi", "response_format": "opus"}, "audio/ogg"),
        ({"model": SPEECH_MODEL, "input": "hi", "response_format": "flac"}, "audio/flac"),
        ({"model": SPEECH_MODEL, "input": "hi", "response_format": "aiff"}, "audio/mpeg"),
    ],
)
def test_transform_response_content_type_follows_the_requested_format(
    request_body: dict[str, str], expected_content_type: str
):
    raw_response: Final = httpx.Response(
        200,
        json={"audio_data": ENCODED_AUDIO},
        request=httpx.Request("POST", SPEECH_URL, json=request_body),
    )
    result: Final = MistralTextToSpeechConfig().transform_text_to_speech_response(
        model=SPEECH_MODEL,
        raw_response=raw_response,
        logging_obj=_speech_logging(),
    )
    assert result.content == b"RIFF"
    assert result.response.headers["content-type"] == expected_content_type


def test_transform_response_missing_audio_data_reports_the_response_keys():
    raw_response: Final = httpx.Response(
        200,
        json={"detail": "unexpected", "id": "req-9"},
        request=httpx.Request("POST", SPEECH_URL, json={"model": SPEECH_MODEL, "input": "hi"}),
    )
    with pytest.raises(MistralTextToSpeechException) as exc_info:
        MistralTextToSpeechConfig().transform_text_to_speech_response(
            model=SPEECH_MODEL,
            raw_response=raw_response,
            logging_obj=_speech_logging(),
        )
    assert exc_info.value.message == "No audio_data in Mistral speech response. Response keys: ('detail', 'id')"
    assert exc_info.value.status_code == 500


@pytest.mark.parametrize("payload", [ENCODED_AUDIO, [ENCODED_AUDIO], [{"audio_data": ENCODED_AUDIO}]])
def test_transform_response_rejects_non_object_bodies(payload: object):
    raw_response: Final = httpx.Response(
        200,
        json=payload,
        request=httpx.Request("POST", SPEECH_URL, json={"model": SPEECH_MODEL, "input": "hi"}),
    )
    with pytest.raises(ValidationError) as exc_info:
        MistralTextToSpeechConfig().transform_text_to_speech_response(
            model=SPEECH_MODEL,
            raw_response=raw_response,
            logging_obj=_speech_logging(),
        )
    assert ENCODED_AUDIO not in str(exc_info.value)
