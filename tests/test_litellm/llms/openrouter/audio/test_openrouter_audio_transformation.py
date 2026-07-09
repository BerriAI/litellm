from unittest.mock import Mock

import httpx
import pytest

import litellm
import litellm.main as litellm_main
from litellm.litellm_core_utils.get_supported_openai_params import get_supported_openai_params
from litellm.llms.openrouter.audio_transcription.transformation import (
    OpenRouterAudioTranscriptionConfig,
)
from litellm.llms.openrouter.common_utils import OpenRouterException
from litellm.llms.openrouter.text_to_speech.transformation import (
    OpenRouterTextToSpeechConfig,
)
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


def _clear_openrouter_keys(monkeypatch):
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "openrouter_key", None)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OR_API_KEY", raising=False)


def test_provider_config_manager_registers_openrouter_audio_configs():
    assert isinstance(
        ProviderConfigManager.get_provider_text_to_speech_config(
            model="hexgrad/kokoro-82m",
            provider=LlmProviders.OPENROUTER,
        ),
        OpenRouterTextToSpeechConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_audio_transcription_config(
            model="openai/whisper-large-v3",
            provider=LlmProviders.OPENROUTER,
        ),
        OpenRouterAudioTranscriptionConfig,
    )


def test_get_supported_openai_params_returns_openrouter_transcription_params():
    assert get_supported_openai_params(
        model="openai/whisper-large-v3",
        custom_llm_provider="openrouter",
        request_type="transcription",
    ) == [
        "language",
        "prompt",
        "response_format",
        "temperature",
        "timestamp_granularities",
    ]


def test_litellm_speech_routes_openrouter_to_shared_tts_handler(monkeypatch):
    sentinel = object()
    captured_kwargs = {}

    def fake_text_to_speech_handler(**kwargs):
        captured_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        litellm_main.base_llm_http_handler,
        "text_to_speech_handler",
        fake_text_to_speech_handler,
    )

    response = litellm.speech(
        model="openrouter/hexgrad/kokoro-82m",
        input="hello",
        voice="af_alloy",
        api_key="test-key",
    )

    assert response is sentinel
    assert captured_kwargs["custom_llm_provider"] == "openrouter"
    assert captured_kwargs["model"] == "hexgrad/kokoro-82m"
    assert captured_kwargs["voice"] == "af_alloy"
    assert captured_kwargs["litellm_params"]["api_key"] == "test-key"
    assert isinstance(captured_kwargs["text_to_speech_provider_config"], OpenRouterTextToSpeechConfig)


def test_litellm_speech_requires_openrouter_voice():
    with pytest.raises(litellm.BadRequestError, match="'voice' is required"):
        litellm.speech(
            model="openrouter/hexgrad/kokoro-82m",
            input="hello",
            api_key="test-key",
        )


class TestOpenRouterTextToSpeechConfig:
    def setup_method(self):
        self.config = OpenRouterTextToSpeechConfig()
        self.logging_obj = Mock()

    def test_validate_environment_sets_openrouter_headers(self):
        headers = self.config.validate_environment(
            headers={"X-Custom": "value"},
            model="hexgrad/kokoro-82m",
            api_key="test-key",
        )

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["HTTP-Referer"] == "https://litellm.ai"
        assert headers["X-Title"] == "liteLLM"
        assert headers["Content-Type"] == "application/json"
        assert headers["X-Custom"] == "value"

    def test_validate_environment_raises_without_api_key(self, monkeypatch):
        _clear_openrouter_keys(monkeypatch)

        with pytest.raises(ValueError, match="OpenRouter API key is required"):
            self.config.validate_environment(
                headers={},
                model="hexgrad/kokoro-82m",
                api_key=None,
            )

    def test_get_complete_url(self):
        assert (
            self.config.get_complete_url(
                model="hexgrad/kokoro-82m",
                api_base="https://openrouter.ai/api/v1/",
                litellm_params={},
            )
            == "https://openrouter.ai/api/v1/audio/speech"
        )

    def test_map_openai_params_normalizes_voice_dict_and_extra_body(self):
        voice, params = self.config.map_openai_params(
            model="hexgrad/kokoro-82m",
            optional_params={"response_format": "mp3"},
            voice={"voice_id": "af_alloy"},
            kwargs={"extra_body": {"sample_rate": 24000}},
        )

        assert voice == "af_alloy"
        assert params == {"response_format": "mp3", "sample_rate": 24000}

    def test_transform_text_to_speech_request_preserves_openai_fields(self):
        request_data = self.config.transform_text_to_speech_request(
            model="hexgrad/kokoro-82m",
            input="Narrate this scene",
            voice="af_alloy",
            optional_params={
                "response_format": "mp3",
                "speed": 1,
                "instructions": "Warm narration",
            },
            litellm_params={},
            headers={},
        )

        body = request_data["dict_body"]
        assert body["model"] == "hexgrad/kokoro-82m"
        assert body["input"] == "Narrate this scene"
        assert body["voice"] == "af_alloy"
        assert body["response_format"] == "mp3"
        assert body["speed"] == 1
        assert body["instructions"] == "Warm narration"

    def test_transform_text_to_speech_response_returns_binary_content(self):
        raw_response = httpx.Response(
            200,
            content=b"audio-bytes",
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/speech"),
        )

        result = self.config.transform_text_to_speech_response(
            model="hexgrad/kokoro-82m",
            raw_response=raw_response,
            logging_obj=self.logging_obj,
        )

        assert isinstance(result, HttpxBinaryResponseContent)
        assert result.response.content == b"audio-bytes"


class TestOpenRouterAudioTranscriptionConfig:
    def setup_method(self):
        self.config = OpenRouterAudioTranscriptionConfig()

    def test_validate_environment_sets_multipart_safe_headers(self):
        headers = self.config.validate_environment(
            headers={"Content-Type": "application/json", "X-Custom": "value"},
            model="openai/whisper-large-v3",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )

        assert headers["Authorization"] == "Bearer test-key"
        assert headers["HTTP-Referer"] == "https://litellm.ai"
        assert headers["X-Title"] == "liteLLM"
        assert headers["X-Custom"] == "value"
        assert "Content-Type" not in headers
        assert "content-type" not in headers

    def test_validate_environment_raises_without_api_key(self, monkeypatch):
        _clear_openrouter_keys(monkeypatch)

        with pytest.raises(ValueError, match="OpenRouter API key is required"):
            self.config.validate_environment(
                headers={},
                model="openai/whisper-large-v3",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=None,
            )

    def test_get_complete_url(self):
        assert (
            self.config.get_complete_url(
                api_base="https://openrouter.ai/api/v1/",
                api_key="test-key",
                model="openai/whisper-large-v3",
                optional_params={},
                litellm_params={},
            )
            == "https://openrouter.ai/api/v1/audio/transcriptions"
        )

    def test_transform_audio_transcription_request_preserves_openai_fields(self):
        request_data = self.config.transform_audio_transcription_request(
            model="openai/whisper-large-v3",
            audio_file=("sample.wav", b"RIFF....", "audio/wav"),
            optional_params={
                "language": "en",
                "prompt": "Product narration",
                "response_format": "verbose_json",
                "temperature": 0,
                "timestamp_granularities": ["word"],
            },
            litellm_params={},
        )

        assert request_data.data["model"] == "openai/whisper-large-v3"
        assert request_data.data["language"] == "en"
        assert request_data.data["prompt"] == "Product narration"
        assert request_data.data["response_format"] == "verbose_json"
        assert request_data.data["temperature"] == 0
        assert request_data.data["timestamp_granularities[]"] == ["word"]
        assert request_data.files["file"][0] == "sample.wav"
        assert request_data.files["file"][1] == b"RIFF...."
        assert request_data.files["file"][2] == "audio/wav"

    def test_transform_audio_transcription_response_preserves_words(self):
        raw_response = httpx.Response(
            200,
            json={
                "text": "hello world",
                "language": "en",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.4},
                    {"word": "world", "start": 0.5, "end": 0.9},
                ],
                "usage": {"duration": 1.0},
            },
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
        )

        response = self.config.transform_audio_transcription_response(raw_response)

        assert response.text == "hello world"
        assert response["language"] == "en"
        assert response["words"][0] == {"word": "hello", "start": 0.0, "end": 0.4}
        assert response["usage"] == {"duration": 1.0}
        assert response._hidden_params["words"][1]["word"] == "world"

    def test_transform_audio_transcription_response_raises_openrouter_error(self):
        raw_response = httpx.Response(
            401,
            json={"error": {"message": "bad key"}},
            request=httpx.Request("POST", "https://openrouter.ai/api/v1/audio/transcriptions"),
        )

        with pytest.raises(OpenRouterException, match="bad key"):
            self.config.transform_audio_transcription_response(raw_response)
