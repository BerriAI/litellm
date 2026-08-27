import base64
import json

import httpx
import pytest


import litellm
from litellm.llms.gemini.audio_transcription.transformation import (
    GeminiAudioTranscriptionConfig,
)
from litellm.llms.gemini.common_utils import GeminiError
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager

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


class TestCostRegression:
    @pytest.fixture
    def local_cost_map(self, monkeypatch):
        monkeypatch.setenv("LITELLM_LOCAL_MODEL_COST_MAP", "True")
        monkeypatch.setattr(litellm, "model_cost", litellm.get_model_cost_map(url=""))

    def test_registry_entries(self, local_cost_map):
        batch_entry = litellm.model_cost["gemini/gemini-3.5-transcribe"]
        assert batch_entry["mode"] == "audio_transcription"
        assert batch_entry["input_cost_per_audio_token"] == 2e-06
        assert batch_entry["input_cost_per_token"] == 2e-06
        assert batch_entry["output_cost_per_token"] == 1.2e-05
        assert batch_entry["supported_endpoints"] == ["/v1/audio/transcriptions"]

        live_entry = litellm.model_cost["gemini/gemini-3.5-transcribe-live"]
        assert live_entry["mode"] == "audio_transcription"
        assert live_entry["input_cost_per_audio_token"] == 3.5e-06
        assert live_entry["input_cost_per_token"] == 3.5e-06
        assert live_entry["output_cost_per_token"] == 2.1e-05
        assert live_entry["supported_endpoints"] == ["/v1/realtime"]

    def test_completion_cost_bills_provider_reported_tokens(self, config, local_cost_map):
        payload = json.loads(json.dumps(COMPLETED_RESPONSE))
        payload["usage"]["total_output_tokens"] = 10
        payload["usage"]["total_tokens"] = 210
        response = config.transform_audio_transcription_response(make_response(payload))
        cost = litellm.completion_cost(
            completion_response=response,
            model="gemini/gemini-3.5-transcribe",
            call_type="transcription",
        )
        assert cost == pytest.approx(199 * 2e-06 + 1 * 2e-06 + 10 * 1.2e-05)
