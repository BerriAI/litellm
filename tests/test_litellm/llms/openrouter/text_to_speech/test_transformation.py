from typing import Final
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.openrouter.text_to_speech.transformation import OpenrouterTextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.utils import ProviderConfigManager


def test_openrouter_text_to_speech_registered():
    config: Final = ProviderConfigManager.get_provider_text_to_speech_config(
        model="google/gemini-3.1-flash-tts-preview",
        provider=litellm.LlmProviders.OPENROUTER,
    )
    assert isinstance(config, OpenrouterTextToSpeechConfig)


def test_openrouter_text_to_speech_supported_params():
    config: Final = OpenrouterTextToSpeechConfig()
    params: Final = config.get_supported_openai_params("any-model")
    assert "voice" in params
    assert "response_format" in params
    assert "speed" in params
    assert "instructions" in params


def test_openrouter_text_to_speech_map_params_voice():
    config: Final = OpenrouterTextToSpeechConfig()
    mapped_voice, _ = config.map_openai_params(
        model="any-model",
        optional_params={},
        voice="alloy",
    )
    assert mapped_voice == "alloy"

    structured_voice, _ = config.map_openai_params(
        model="any-model",
        optional_params={},
        voice={"voice_id": "alloy"},
    )
    assert structured_voice is None


def test_openrouter_text_to_speech_trusted_api_base():
    config: Final = OpenrouterTextToSpeechConfig()
    assert config.is_trusted_api_base(None) is True
    assert config.is_trusted_api_base("https://openrouter.ai/api/v1") is True
    assert config.is_trusted_api_base("https://subdomain.openrouter.ai/api/v1") is True
    assert config.is_trusted_api_base("http://openrouter.ai/api/v1") is False
    assert config.is_trusted_api_base("https://untrusted-domain.com/v1") is False


def test_openrouter_text_to_speech_resolve_credentials_isolation(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-server-key")
    config: Final = OpenrouterTextToSpeechConfig()

    base_trusted, key_trusted = config.resolve_api_base_and_key()
    assert base_trusted == "https://openrouter.ai/api/v1"
    assert key_trusted == "sk-or-server-key"

    base_untrusted, key_untrusted = config.resolve_api_base_and_key(api_base="https://untrusted-host.com/v1")
    assert base_untrusted == "https://untrusted-host.com/v1"
    assert key_untrusted is None

    base_http, key_http = config.resolve_api_base_and_key(api_base="http://openrouter.ai/api/v1")
    assert base_http == "http://openrouter.ai/api/v1"
    assert key_http is None


def test_openrouter_text_to_speech_url_and_transforms():
    config: Final = OpenrouterTextToSpeechConfig()
    url: Final = config.get_complete_url(
        model="google/gemini-3.1-flash-tts-preview",
        api_base=None,
        litellm_params={},
    )
    assert url == "https://openrouter.ai/api/v1/audio/speech"

    request_data: Final = config.transform_text_to_speech_request(
        model="google/gemini-3.1-flash-tts-preview",
        input="hello world",
        voice="alloy",
        optional_params={"speed": 1.2},
        litellm_params={},
        headers={"custom-header": "value"},
    )
    assert request_data["dict_body"]["model"] == "google/gemini-3.1-flash-tts-preview"
    assert request_data["dict_body"]["input"] == "hello world"
    assert request_data["dict_body"]["voice"] == "alloy"
    assert request_data["dict_body"]["speed"] == 1.2
    assert request_data["headers"]["custom-header"] == "value"

    raw_response: Final = httpx.Response(status_code=200, content=b"audio-bytes")
    wrapped: Final = config.transform_text_to_speech_response(
        model="google/gemini-3.1-flash-tts-preview",
        raw_response=raw_response,
        logging_obj=MagicMock(),
    )
    assert isinstance(wrapped, HttpxBinaryResponseContent)


def test_litellm_speech_openrouter_full_routing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    with patch("litellm.main.openai_chat_completions.audio_speech") as mock_audio:
        mock_audio.return_value = MagicMock(spec=HttpxBinaryResponseContent)

        litellm.speech(
            model="openrouter/google/gemini-3.1-flash-tts-preview",
            voice="alloy",
            input="testing speech",
        )

        mock_audio.assert_called_once()
        _, kwargs = mock_audio.call_args
        assert kwargs["model"] == "google/gemini-3.1-flash-tts-preview"
        assert kwargs["api_base"] == "https://openrouter.ai/api/v1"
        assert kwargs["api_key"] == "sk-or-test-key"
        assert kwargs["voice"] == "alloy"


@pytest.mark.asyncio
async def test_litellm_aspeech_openrouter_full_routing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OR_API_KEY", "sk-or-alt-key")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with patch("litellm.main.openai_chat_completions.audio_speech") as mock_audio:
        mock_response: Final = MagicMock(spec=HttpxBinaryResponseContent)

        async def _coro(*args, **kwargs):
            return mock_response

        mock_audio.side_effect = _coro

        await litellm.aspeech(
            model="openrouter/google/gemini-3.1-flash-tts-preview",
            voice="alloy",
            input="testing async speech",
        )

        mock_audio.assert_called_once()
        _, kwargs = mock_audio.call_args
        assert kwargs["model"] == "google/gemini-3.1-flash-tts-preview"
        assert kwargs["api_base"] == "https://openrouter.ai/api/v1"
        assert kwargs["api_key"] == "sk-or-alt-key"
        assert kwargs["aspeech"] is True


def test_litellm_speech_openrouter_rejects_dict_voice():
    with pytest.raises(litellm.BadRequestError):
        litellm.speech(
            model="openrouter/google/gemini-3.1-flash-tts-preview",
            voice={"voice_id": "alloy"},  # pyright: ignore[reportArgumentType]  # test invalid voice input
            input="testing structured voice rejection",
        )
