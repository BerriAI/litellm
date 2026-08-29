import base64
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig
from litellm.llms.mistral.audio_speech.transformation import (
    MistralTextToSpeechConfig,
    MistralTextToSpeechException,
)
from litellm.utils import ProviderConfigManager

SPEECH_URL: Final = "https://api.mistral.ai/v1/audio/speech"


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


def test_get_complete_url_default_base(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MISTRAL_API_BASE", raising=False)
    config: Final = MistralTextToSpeechConfig()
    url: Final = config.get_complete_url(model="voxtral-mini-tts-2603", api_base=None, litellm_params={})
    assert url == SPEECH_URL


def test_get_complete_url_custom_base():
    config: Final = MistralTextToSpeechConfig()
    url: Final = config.get_complete_url(
        model="voxtral-mini-tts-2603",
        api_base="https://custom.api.example.com/v1/",
        litellm_params={},
    )
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
        json={"audio_data": "!!!not-base64!!!"},
        request=httpx.Request("POST", SPEECH_URL),
    )
    with pytest.raises(MistralTextToSpeechException, match="base64"):
        config.transform_text_to_speech_response(
            model="voxtral-mini-tts-2603",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )
