import base64
import json
import os
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.vertex_ai.audio_transcription.gemini_transcribe_transformation import (
    VertexGeminiAudioTranscriptionConfig,
)
from litellm.llms.vertex_ai.audio_transcription.transformation import (
    VertexAIAudioTranscriptionConfig,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.types.utils import LlmProviders, TranscriptionUsageTokensObject
from litellm.utils import ProviderConfigManager, get_optional_params_transcription

AUDIO_BYTES = b"fake-audio-bytes"
TRANSCRIPT_TEXT = (
    "Four score and seven years ago our fathers brought forth on this continent, a new nation, "
    "conceived in Liberty, and dedicated to the proposition that all men are created equal. "
    "Now we are engaged in a great civil war, testing whether that nation, or any nation so "
    "conceived and so dedicated, can long endure."
)
GENERATE_CONTENT_RESPONSE = {
    "candidates": [
        {
            "content": {
                "role": "model",
                "parts": [
                    {
                        "text": TRANSCRIPT_TEXT,
                        "audioTranscription": {"text": TRANSCRIPT_TEXT},
                    }
                ],
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 440,
        "candidatesTokenCount": 62,
        "totalTokenCount": 502,
        "trafficType": "ON_DEMAND",
        "promptTokensDetails": [{"modality": "AUDIO", "tokenCount": 440}],
        "candidatesTokensDetails": [{"modality": "TEXT", "tokenCount": 62}],
    },
    "modelVersion": "gemini-3.5-transcribe-preview",
    "createTime": "2026-08-29T07:25:27.591648Z",
    "responseId": "Z4mSaqCOJL-O4_UP0aSh4Aw",
}


@pytest.fixture
def config():
    return VertexGeminiAudioTranscriptionConfig()


class TestProviderRouting:
    @pytest.mark.parametrize(
        "model",
        [
            "gemini-3.5-transcribe-preview",
            "gemini-3.5-transcribe-live-preview",
            "vertex_ai/gemini-3.5-transcribe-preview",
        ],
    )
    def test_gemini_transcribe_models_use_generate_content_config(self, model):
        provider_config = ProviderConfigManager.get_provider_audio_transcription_config(
            model=model,
            provider=LlmProviders.VERTEX_AI,
        )
        assert isinstance(provider_config, VertexGeminiAudioTranscriptionConfig)

    @pytest.mark.parametrize("model", ["chirp_2", "chirp_3", "long-form", "gemini-2.5-flash"])
    def test_other_vertex_models_keep_speech_to_text_config(self, model):
        provider_config = ProviderConfigManager.get_provider_audio_transcription_config(
            model=model,
            provider=LlmProviders.VERTEX_AI,
        )
        assert isinstance(provider_config, VertexAIAudioTranscriptionConfig)
        assert not isinstance(provider_config, VertexGeminiAudioTranscriptionConfig)


class TestGetCompleteUrl:
    @pytest.fixture(autouse=True)
    def _clear_ambient_vertex_location(self, monkeypatch):
        monkeypatch.delenv("VERTEXAI_LOCATION", raising=False)
        monkeypatch.delenv("VERTEX_LOCATION", raising=False)

    def test_defaults_to_global_location(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-3.5-transcribe-preview",
            optional_params={},
            litellm_params={"vertex_project": "test-project"},
        )
        assert url == (
            "https://aiplatform.googleapis.com/v1/projects/test-project/locations/global"
            "/publishers/google/models/gemini-3.5-transcribe-preview:generateContent"
        )

    def test_explicit_location_is_honored(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="gemini-3.5-transcribe-preview",
            optional_params={},
            litellm_params={"vertex_project": "test-project", "vertex_location": "us-central1"},
        )
        assert url == (
            "https://us-central1-aiplatform.googleapis.com/v1/projects/test-project/locations/us-central1"
            "/publishers/google/models/gemini-3.5-transcribe-preview:generateContent"
        )

    def test_model_prefix_is_stripped(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="vertex_ai/gemini-3.5-transcribe-preview",
            optional_params={},
            litellm_params={"vertex_project": "test-project"},
        )
        assert "/models/gemini-3.5-transcribe-preview:generateContent" in url
        assert "vertex_ai/" not in url

    def test_api_base_override(self, config):
        url = config.get_complete_url(
            api_base="http://localhost:8080/",
            api_key=None,
            model="gemini-3.5-transcribe-preview",
            optional_params={},
            litellm_params={"vertex_project": "test-project"},
        )
        assert url == (
            "http://localhost:8080/v1/projects/test-project/locations/global"
            "/publishers/google/models/gemini-3.5-transcribe-preview:generateContent"
        )

    @pytest.mark.parametrize("malicious_location", ["attacker.example/", "evil.com#", "US", "us/../.."])
    def test_malicious_location_is_rejected(self, config, malicious_location):
        with pytest.raises(VertexAIError):
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model="gemini-3.5-transcribe-preview",
                optional_params={},
                litellm_params={"vertex_project": "test-project", "vertex_location": malicious_location},
            )

    @pytest.mark.parametrize("malicious_project", ["proj/../../locations", "proj#frag", "proj?a=b", "proj space"])
    def test_malicious_project_is_rejected(self, config, malicious_project):
        with pytest.raises(VertexAIError):
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model="gemini-3.5-transcribe-preview",
                optional_params={},
                litellm_params={"vertex_project": malicious_project},
            )


class TestTransformRequest:
    def test_request_body_shape(self, config):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe-preview",
            audio_file=AUDIO_BYTES,
            optional_params={},
            litellm_params={},
        )
        assert request_data.files is None
        assert request_data.data == {
            "contents": (
                {
                    "role": "user",
                    "parts": (
                        {
                            "inlineData": {
                                "mimeType": "audio/wav",
                                "data": base64.b64encode(AUDIO_BYTES).decode("utf-8"),
                            }
                        },
                    ),
                },
            ),
            "generationConfig": {"audioTranscriptionConfig": {}},
        }

    @pytest.mark.parametrize(
        "language,expected_language_codes",
        [
            ("en", ("en-US",)),
            ("en-US", ("en-US",)),
            ("fr", ("fr-FR",)),
        ],
    )
    def test_language_param_maps_to_language_codes(self, config, language, expected_language_codes):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe-preview",
            audio_file=AUDIO_BYTES,
            optional_params={"language": language},
            litellm_params={},
        )
        audio_config = request_data.data["generationConfig"]["audioTranscriptionConfig"]
        assert audio_config["languageCodes"] == expected_language_codes

    def test_body_round_trips_through_json(self, config):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe-preview",
            audio_file=AUDIO_BYTES,
            optional_params={"language": "en"},
            litellm_params={},
        )
        round_tripped = json.loads(json.dumps(request_data.data))
        assert round_tripped["generationConfig"] == {"audioTranscriptionConfig": {"languageCodes": ["en-US"]}}
        assert round_tripped["contents"][0]["role"] == "user"

    def test_webm_audio_uses_audio_mime_type(self, config):
        audio_file = Mock(spec=["name", "read", "seek"])
        audio_file.name = "speech.webm"
        audio_file.read.return_value = AUDIO_BYTES

        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe-preview",
            audio_file=audio_file,
            optional_params={},
            litellm_params={},
        )

        assert request_data.data["contents"][0]["parts"][0]["inlineData"]["mimeType"] == "audio/webm"

    def test_incoming_audio_mime_type_is_preserved(self, config):
        request_data = config.transform_audio_transcription_request(
            model="gemini-3.5-transcribe-preview",
            audio_file=("speech.webm", AUDIO_BYTES, "audio/webm; codecs=opus"),
            optional_params={},
            litellm_params={},
        )

        assert request_data.data["contents"][0]["parts"][0]["inlineData"]["mimeType"] == "audio/webm; codecs=opus"


class TestTransformResponse:
    def test_generate_content_response(self, config):
        raw_response = httpx.Response(status_code=200, json=GENERATE_CONTENT_RESPONSE)
        response = config.transform_audio_transcription_response(raw_response)
        assert response.text == TRANSCRIPT_TEXT
        assert response["task"] == "transcribe"
        assert isinstance(response.usage, TranscriptionUsageTokensObject)
        assert response.usage.input_tokens == 440
        assert response.usage.output_tokens == 62
        assert response.usage.total_tokens == 502
        assert response.usage.input_token_details.audio_tokens == 440
        assert response.usage.input_token_details.text_tokens == 0

    def test_multi_part_texts_are_joined(self, config):
        raw_response = httpx.Response(
            status_code=200,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "Hello world."}, {"text": "How are you?"}]}}
                ],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            },
        )
        response = config.transform_audio_transcription_response(raw_response)
        assert response.text == "Hello world. How are you?"

    def test_empty_candidates_returns_empty_text(self, config):
        raw_response = httpx.Response(status_code=200, json={})
        response = config.transform_audio_transcription_response(raw_response)
        assert response.text == ""
        assert response.usage is None

    def test_non_json_body_raises(self, config):
        raw_response = httpx.Response(status_code=200, text="<html>not json</html>")
        with pytest.raises(VertexAIError, match="non-JSON"):
            config.transform_audio_transcription_response(raw_response)


class TestValidateEnvironment:
    def test_sets_oauth_headers(self):
        class StubbedConfig(VertexGeminiAudioTranscriptionConfig):
            def _ensure_access_token(self, credentials, project_id, custom_llm_provider):
                return "fake-token", "resolved-project"

        headers = StubbedConfig().validate_environment(
            headers={},
            model="gemini-3.5-transcribe-preview",
            messages=[],
            optional_params={},
            litellm_params={"vertex_project": "resolved-project"},
        )
        assert headers["Authorization"] == "Bearer fake-token"
        assert headers["x-goog-user-project"] == "resolved-project"
        assert headers["Content-Type"] == "application/json"


class TestOptionalParams:
    def test_language_and_json_response_format_pass_through(self):
        optional_params = get_optional_params_transcription(
            model="gemini-3.5-transcribe-preview",
            custom_llm_provider="vertex_ai",
            language="fr-FR",
            response_format="json",
        )
        assert optional_params["language"] == "fr-FR"
        assert optional_params["response_format"] == "json"

    @pytest.mark.parametrize("response_format", ["verbose_json", "srt", "vtt"])
    def test_unsupported_response_format_raises(self, response_format):
        with pytest.raises(litellm.utils.UnsupportedParamsError, match="response_format"):
            get_optional_params_transcription(
                model="gemini-3.5-transcribe-preview",
                custom_llm_provider="vertex_ai",
                response_format=response_format,
            )

    @pytest.mark.parametrize("response_format", ["verbose_json", "srt", "vtt"])
    def test_unsupported_response_format_dropped_with_drop_params(self, response_format):
        optional_params = get_optional_params_transcription(
            model="gemini-3.5-transcribe-preview",
            custom_llm_provider="vertex_ai",
            language="fr-FR",
            response_format=response_format,
            drop_params=True,
        )
        assert "response_format" not in optional_params
        assert optional_params["language"] == "fr-FR"


class TestModelCostEntry:
    REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))

    @pytest.mark.parametrize(
        "cost_map_path",
        [
            "model_prices_and_context_window.json",
            "litellm/model_prices_and_context_window_backup.json",
        ],
    )
    def test_transcribe_preview_pricing(self, cost_map_path):
        with open(os.path.join(self.REPO_ROOT, cost_map_path)) as f:
            entry = json.load(f)["vertex_ai/gemini-3.5-transcribe-preview"]
        assert entry["mode"] == "audio_transcription"
        assert entry["litellm_provider"] == "vertex_ai"
        assert entry["input_cost_per_audio_token"] == pytest.approx(2.5e-06)
        assert entry["input_cost_per_token"] == pytest.approx(2.5e-06)
        assert entry["output_cost_per_token"] == pytest.approx(1.2e-05)
        assert entry["supported_endpoints"] == ["/v1/audio/transcriptions"]

    @pytest.mark.parametrize(
        "cost_map_path",
        [
            "model_prices_and_context_window.json",
            "litellm/model_prices_and_context_window_backup.json",
        ],
    )
    def test_transcribe_live_preview_pricing(self, cost_map_path):
        with open(os.path.join(self.REPO_ROOT, cost_map_path)) as f:
            entry = json.load(f)["vertex_ai/gemini-3.5-transcribe-live-preview"]
        assert entry["mode"] == "audio_transcription"
        assert entry["litellm_provider"] == "vertex_ai"
        assert entry["input_cost_per_audio_token"] == pytest.approx(3.5e-06)
        assert entry["input_cost_per_token"] == pytest.approx(3.5e-06)
        assert entry["output_cost_per_token"] == pytest.approx(2.1e-05)
        assert entry["supported_endpoints"] == ["/v1/realtime"]
