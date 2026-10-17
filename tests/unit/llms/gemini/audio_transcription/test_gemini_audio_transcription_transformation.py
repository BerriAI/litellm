import base64
import json
from collections.abc import Mapping
from typing import Final

import httpx
import pytest

import litellm
from litellm.llms.gemini.audio_transcription.transformation import (
    GeminiAudioTranscriptionConfig,
)
from litellm.llms.gemini.common_utils import GeminiError
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_optional_params_transcription

AUDIO_BYTES = b"RIFF....WAVEfmt fake-wav-bytes"

COMPLETED_RESPONSE = {
    "id": "v1_abc123",
    "status": "completed",
    "usage": {
        "total_tokens": 200,
        "total_input_tokens": 200,
        "input_tokens_by_modality": [
            {"modality": "text", "tokens": 1},
            {"modality": "audio", "tokens": 199},
        ],
        "total_output_tokens": 0,
    },
    "steps": [
        {
            "type": "model_generation",
            "content": [
                {
                    "type": "text",
                    "text": "Hello world.",
                    "annotations": [
                        {
                            "type": "word_info",
                            "text": "Hello",
                            "speaker": "spk:0",
                            "start_offset": "0.100s",
                            "end_offset": "0.400s",
                        },
                        {
                            "type": "word_info",
                            "text": "world.",
                            "speaker": "spk:1",
                            "start_offset": "0.500s",
                            "end_offset": "0.900s",
                        },
                    ],
                }
            ],
        }
    ],
}


def make_response(payload):
    return httpx.Response(200, json=payload, request=httpx.Request("POST", "https://example.test"))


@pytest.fixture
def config():
    return GeminiAudioTranscriptionConfig()


def test_provider_config_manager_returns_gemini_config():
    provider_config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="gemini-3.5-transcribe", provider=LlmProviders.GEMINI
    )
    assert isinstance(provider_config, GeminiAudioTranscriptionConfig)


class TestValidateEnvironment:
    def test_sets_api_key_and_revision_headers(self, config):
        headers = config.validate_environment(
            headers={},
            model="gemini-3.5-transcribe",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="test-key",
        )
        assert headers["x-goog-api-key"] == "test-key"
        assert headers["Api-Revision"] == "2026-05-20"
        assert headers["Content-Type"] == "application/json"

    def test_missing_api_key_raises(self, config, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(GeminiError) as excinfo:
            config.validate_environment(
                headers={},
                model="gemini-3.5-transcribe",
                messages=[],
                optional_params={},
                litellm_params={},
            )
        assert excinfo.value.status_code == 401


class TestGetCompleteUrl:
    def test_defaults_to_interactions_endpoint(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-3.5-transcribe",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://generativelanguage.googleapis.com/v1beta/interactions"

    def test_api_base_override(self, config):
        url = config.get_complete_url(
            api_base="http://localhost:8080",
            api_key=None,
            model="gemini-3.5-transcribe",
            optional_params={},
            litellm_params={},
        )
        assert url == "http://localhost:8080/v1beta/interactions"


class TestTransformRequest:
    @pytest.mark.parametrize(
        ("params", "expected_codes"),
        [
            ({"language_codes": ["en", "es-ES"]}, ["en-US", "es-ES"]),
            ({"language_codes": '["en", "es-ES"]'}, ["en-US", "es-ES"]),
            ({"language_codes": [" en ", " es-ES "]}, ["en-US", "es-ES"]),
            ({"language_codes": ["zh-Hant-TW", "zh-Hans-CN"]}, ["zh-Hant-TW", "zh-Hans-CN"]),
            ({"language": "fr", "language_codes": ["en", "es"]}, ["en-US", "es-ES"]),
            ({"language": "fr", "language_codes": []}, []),
            ({"language": "fr", "language_codes": "[]"}, []),
            ({"language": "fr", "language_codes": None}, ["fr-FR"]),
        ],
    )
    def test_language_codes_reach_gemini_through_provider_params(
        self,
        config: GeminiAudioTranscriptionConfig,
        params: Mapping[str, object],
        expected_codes: list[str],
    ) -> None:
        optional_params: Final = get_optional_params_transcription(
            model="gemini-3.5-transcribe", custom_llm_provider="gemini", **params
        )
        request_data: Final = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe",
            audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
            optional_params=optional_params,
            litellm_params={},
        )
        assert json.loads(json.dumps(request_data.data)) == {
            "model": "gemini-3.5-transcribe",
            "input": [
                {
                    "type": "audio",
                    "data": base64.b64encode(AUDIO_BYTES).decode("utf-8"),
                    "mime_type": "audio/wav",
                }
            ],
            "generation_config": {"transcription_config": {"language_codes": expected_codes}},
        }

    @pytest.mark.parametrize(
        "language_codes",
        ["en", '"en"', "[", "null", 1, {}, ["en", 1], ["en", ""], [" "], '["en", null]'],
    )
    def test_invalid_language_codes_raise_instead_of_silently_dropping_hints(
        self, config: GeminiAudioTranscriptionConfig, language_codes: object
    ) -> None:
        with pytest.raises(GeminiError, match="language_codes must be a list of non-empty language strings") as excinfo:
            config.transform_audio_transcription_request(
                model="gemini-3.5-transcribe",
                audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
                optional_params={"language_codes": language_codes},
                litellm_params={},
            )
        assert excinfo.value.status_code == 400

    def test_multiple_language_codes_preserve_word_timestamps(self, config: GeminiAudioTranscriptionConfig) -> None:
        request_data: Final = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe",
            audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
            optional_params={"language_codes": ["en-US", "es-ES"], "timestamp_granularities": ["word"]},
            litellm_params={},
        )
        assert json.loads(json.dumps(request_data.data["generation_config"])) == {
            "transcription_config": {
                "language_codes": ["en-US", "es-ES"],
                "mode": {"type": "verbatim", "timestamp_granularities": ["word"], "diarization_mode": "speaker"},
            }
        }

    def test_builds_json_interaction_request(self, config):
        request_data = config.transform_audio_transcription_request(
            model="gemini/gemini-3.5-transcribe",
            audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
            optional_params={},
            litellm_params={},
        )
        assert request_data.files is None
        assert json.loads(json.dumps(request_data.data)) == {
            "model": "gemini-3.5-transcribe",
            "input": [
                {
                    "type": "audio",
                    "data": base64.b64encode(AUDIO_BYTES).decode("utf-8"),
                    "mime_type": "audio/wav",
                }
            ],
        }

    def test_language_maps_to_bcp47_language_codes(self, config):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe",
            audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
            optional_params={"language": "en"},
            litellm_params={},
        )
        transcription_config = request_data.data["generation_config"]["transcription_config"]
        assert json.loads(json.dumps(transcription_config)) == {"language_codes": ["en-US"]}

    def test_word_timestamp_granularity_maps_to_verbatim_diarization_mode(self, config):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe",
            audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
            optional_params={"timestamp_granularities": ["word"]},
            litellm_params={},
        )
        transcription_config = request_data.data["generation_config"]["transcription_config"]
        assert json.loads(json.dumps(transcription_config)) == {
            "mode": {
                "type": "verbatim",
                "timestamp_granularities": ["word"],
                "diarization_mode": "speaker",
            }
        }

    @pytest.mark.parametrize("response_format", ["srt", "vtt"])
    def test_subtitle_response_format_requests_word_timestamps(self, config, response_format):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe",
            audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
            optional_params={"response_format": response_format},
            litellm_params={},
        )
        transcription_config = request_data.data["generation_config"]["transcription_config"]
        assert json.loads(json.dumps(transcription_config)) == {
            "mode": {
                "type": "verbatim",
                "timestamp_granularities": ["word"],
                "diarization_mode": "speaker",
            }
        }

    @pytest.mark.parametrize("response_format", ["json", "text", "verbose_json"])
    def test_non_subtitle_response_format_sends_no_mode(self, config, response_format):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe",
            audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
            optional_params={"response_format": response_format},
            litellm_params={},
        )
        assert "generation_config" not in request_data.data

    def test_non_string_response_format_sends_no_mode(self, config):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe",
            audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
            optional_params={"response_format": {"type": "json_object"}},
            litellm_params={},
        )
        assert "generation_config" not in request_data.data

    def test_segment_granularity_sends_no_mode(self, config):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe",
            audio_file=("sample.wav", AUDIO_BYTES, "audio/wav"),
            optional_params={"timestamp_granularities": ["segment"]},
            litellm_params={},
        )
        assert "generation_config" not in request_data.data


class TestTransformResponse:
    def test_completed_interaction_maps_to_transcription_response(self, config):
        response = config.transform_audio_transcription_response(make_response(COMPLETED_RESPONSE))
        assert response.text == "Hello world."
        assert response["task"] == "transcribe"
        assert response["words"] == [
            {"word": "Hello", "start": 0.1, "end": 0.4, "speaker": "spk:0"},
            {"word": "world.", "start": 0.5, "end": 0.9, "speaker": "spk:1"},
        ]
        assert response["duration"] == 0.9
        assert response.usage.input_tokens == 200
        assert response.usage.output_tokens == 0
        assert response.usage.total_tokens == 200
        assert response.usage.input_token_details.audio_tokens == 199
        assert response.usage.input_token_details.text_tokens == 1

    def test_non_completed_status_raises(self, config):
        with pytest.raises(GeminiError, match="did not complete"):
            config.transform_audio_transcription_response(
                make_response({**COMPLETED_RESPONSE, "status": "in_progress"})
            )

    def test_non_json_response_raises(self, config):
        raw = httpx.Response(200, text="<html>oops</html>", request=httpx.Request("POST", "https://example.test"))
        with pytest.raises(GeminiError, match="non-JSON"):
            config.transform_audio_transcription_response(raw)

    def test_word_without_offsets_survives(self, config):
        payload = json.loads(json.dumps(COMPLETED_RESPONSE))
        payload["steps"][0]["content"][0]["annotations"] = [{"type": "word_info", "text": "Hello"}]
        response = config.transform_audio_transcription_response(make_response(payload))
        assert response["words"] == [{"word": "Hello"}]
        assert response.get("duration") is None


class TestSubtitleSynthesisThroughHandler:
    def _transform(self, config, response_format):
        from unittest.mock import Mock

        from litellm.llms.custom_httpx.llm_http_handler import BaseLLMHTTPHandler
        from litellm.types.utils import TranscriptionResponse

        return BaseLLMHTTPHandler()._transform_audio_transcription_response(
            provider_config=config,
            model="gemini-3.5-transcribe",
            response=make_response(COMPLETED_RESPONSE),
            model_response=TranscriptionResponse(),
            logging_obj=Mock(),
            optional_params={"response_format": response_format},
            api_key=None,
        )

    def test_supports_subtitle_synthesis(self, config):
        assert config.supports_subtitle_synthesis is True

    def test_srt_synthesizes_subtitle_document_and_drops_words(self, config):
        response = self._transform(config, "srt")
        assert response.text == (
            "1\n00:00:00,100 --> 00:00:00,400\nHello\n\n2\n00:00:00,500 --> 00:00:00,900\nworld.\n"
        )
        assert "words" not in response
        assert response["task"] == "transcribe"
        assert response["duration"] == 0.9
        assert response.usage.total_tokens == 200

    def test_vtt_synthesizes_subtitle_document_and_drops_words(self, config):
        response = self._transform(config, "vtt")
        assert response.text == (
            "WEBVTT\n\n00:00:00.100 --> 00:00:00.400\nHello\n\n00:00:00.500 --> 00:00:00.900\nworld.\n"
        )
        assert "words" not in response
        assert response.usage.total_tokens == 200

    @pytest.mark.parametrize("response_format", ["json", "verbose_json"])
    def test_non_subtitle_formats_keep_plain_text_and_words(self, config, response_format):
        response = self._transform(config, response_format)
        assert response.text == "Hello world."
        assert response["words"] == [
            {"word": "Hello", "start": 0.1, "end": 0.4, "speaker": "spk:0"},
            {"word": "world.", "start": 0.5, "end": 0.9, "speaker": "spk:1"},
        ]


class TestCostRegression:
    @pytest.fixture
    def local_cost_map(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))
