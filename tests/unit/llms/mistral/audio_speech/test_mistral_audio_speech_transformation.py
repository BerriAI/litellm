import base64
import json
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

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


_SPEECH_AUDIO: Final = b"ID3-fake-mp3-bytes"
_SPEECH_AUDIO_B64: Final = base64.b64encode(_SPEECH_AUDIO).decode()


@pytest.mark.parametrize(
    ("request_body", "expected_content_type"),
    [
        pytest.param(b'{"input": "hi", "response_format": "opus"}', "audio/ogg", id="known-format"),
        pytest.param(b'{"input": "hi", "response_format": "aac"}', "audio/mpeg", id="unknown-format"),
        pytest.param(b'{"input": "hi", "response_format": null}', "audio/mpeg", id="null-format"),
        pytest.param(b'{"input": "hi", "response_format": ["wav"]}', "audio/mpeg", id="non-string-format"),
        pytest.param(b'{"input": "hi", "ref_audio": "bXktdm9pY2Utc2FtcGxl"}', "audio/mpeg", id="format-omitted"),
        pytest.param(b"", "audio/mpeg", id="request-without-body"),
    ],
)
def test_transform_response_labels_audio_with_the_requested_format(request_body: bytes, expected_content_type: str):
    raw_response: Final = httpx.Response(
        200,
        json={"id": None, "audio_data": _SPEECH_AUDIO_B64, "usage": {"characters": 2}, "voices": ["en_paul_neutral"]},
        request=httpx.Request("POST", SPEECH_URL, content=request_body),
    )
    result: Final = MistralTextToSpeechConfig().transform_text_to_speech_response(
        model="voxtral-mini-tts-2603",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )
    assert result.content == _SPEECH_AUDIO
    assert result.response.headers["content-type"] == expected_content_type


@pytest.mark.parametrize(
    ("payload", "expected_keys"),
    [
        pytest.param({}, "()", id="empty-object"),
        pytest.param({"audio_data": None}, "('audio_data',)", id="null-audio"),
        pytest.param({"audio_data": ""}, "('audio_data',)", id="empty-audio"),
        pytest.param({"id": "tts-1", "audio_data": 7}, "('id', 'audio_data')", id="numeric-audio"),
        pytest.param({"audio_data": [_SPEECH_AUDIO_B64], "id": "tts-1"}, "('audio_data', 'id')", id="list-audio"),
    ],
)
def test_transform_response_without_usable_audio_reports_the_response_keys(
    payload: dict[str, object], expected_keys: str
):
    raw_response: Final = httpx.Response(200, json=payload, request=httpx.Request("POST", SPEECH_URL))
    with pytest.raises(MistralTextToSpeechException) as exc_info:
        MistralTextToSpeechConfig().transform_text_to_speech_response(
            model="voxtral-mini-tts-2603",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )
    assert exc_info.value.status_code == 500
    assert exc_info.value.message == f"No audio_data in Mistral speech response. Response keys: {expected_keys}"


def test_transform_response_non_json_body_raises_provider_error_with_upstream_status():
    raw_response: Final = httpx.Response(
        502,
        text="<html>bad gateway</html>",
        request=httpx.Request("POST", SPEECH_URL),
    )
    with pytest.raises(MistralTextToSpeechException) as exc_info:
        MistralTextToSpeechConfig().transform_text_to_speech_response(
            model="voxtral-mini-tts-2603",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )
    assert exc_info.value.status_code == 502
    assert exc_info.value.message == "Non-JSON response from Mistral speech API: <html>bad gateway</html>"


@pytest.mark.parametrize(
    "response_body",
    [b'["secret-audio-payload"]', b'"secret-audio-payload"', b"7", b"null", b"true"],
)
def test_transform_response_json_body_that_is_not_an_object_is_rejected_without_echoing_it(response_body: bytes):
    raw_response: Final = httpx.Response(200, content=response_body, request=httpx.Request("POST", SPEECH_URL))
    with pytest.raises(ValidationError) as exc_info:
        MistralTextToSpeechConfig().transform_text_to_speech_response(
            model="voxtral-mini-tts-2603",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )
    assert "secret-audio-payload" not in str(exc_info.value)
    assert "input_value" not in str(exc_info.value)


@pytest.mark.parametrize(
    ("request_body", "expected_error"),
    [
        pytest.param(b'["wav"]', ValidationError, id="json-array-body"),
        pytest.param(b'"wav"', ValidationError, id="json-string-body"),
        pytest.param(b"response_format=wav", json.JSONDecodeError, id="form-encoded-body"),
    ],
)
def test_transform_response_rejects_a_request_body_that_is_not_a_json_object(
    request_body: bytes, expected_error: type[Exception]
):
    raw_response: Final = httpx.Response(
        200,
        json={"audio_data": _SPEECH_AUDIO_B64},
        request=httpx.Request("POST", SPEECH_URL, content=request_body),
    )
    with pytest.raises(expected_error):
        MistralTextToSpeechConfig().transform_text_to_speech_response(
            model="voxtral-mini-tts-2603",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )
