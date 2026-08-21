import base64
import json
import os
import sys
from urllib.parse import urlparse

import httpx
import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.llms.vertex_ai.audio_transcription.transformation import (
    VertexAIAudioTranscriptionConfig,
)
from litellm.llms.vertex_ai.common_utils import VertexAIError
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager, get_optional_params_transcription


@pytest.fixture
def config():
    return VertexAIAudioTranscriptionConfig()


class TestGetCompleteUrl:
    def test_defaults_to_us_regional_host(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="chirp_3",
            optional_params={},
            litellm_params={"vertex_project": "test-project"},
        )
        assert url == "https://us-speech.googleapis.com/v2/projects/test-project/locations/us/recognizers/_:recognize"

    def test_uses_vertex_location_for_regional_host(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="chirp_3",
            optional_params={},
            litellm_params={"vertex_project": "test-project", "vertex_location": "eu"},
        )
        assert url == "https://eu-speech.googleapis.com/v2/projects/test-project/locations/eu/recognizers/_:recognize"

    def test_global_location_uses_unprefixed_host(self, config):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="chirp_3",
            optional_params={},
            litellm_params={"vertex_project": "test-project", "vertex_location": "global"},
        )
        assert url == "https://speech.googleapis.com/v2/projects/test-project/locations/global/recognizers/_:recognize"

    def test_api_base_override(self, config):
        url = config.get_complete_url(
            api_base="http://localhost:8080/",
            api_key=None,
            model="chirp_3",
            optional_params={},
            litellm_params={"vertex_project": "test-project"},
        )
        assert url == "http://localhost:8080/v2/projects/test-project/locations/us/recognizers/_:recognize"

    @pytest.mark.parametrize(
        "location,expected_netloc",
        [
            ("us", "us-speech.googleapis.com"),
            ("us-central1", "us-central1-speech.googleapis.com"),
            ("eu", "eu-speech.googleapis.com"),
            ("global", "speech.googleapis.com"),
        ],
    )
    def test_valid_location_netloc_always_google(self, config, location, expected_netloc):
        url = config.get_complete_url(
            api_base=None,
            api_key=None,
            model="chirp_3",
            optional_params={},
            litellm_params={"vertex_project": "test-project", "vertex_location": location},
        )
        netloc = urlparse(url).netloc
        assert netloc == expected_netloc
        assert netloc.endswith("speech.googleapis.com")

    @pytest.mark.parametrize(
        "malicious_location",
        [
            "attacker.example/",
            "evil.com#",
            "us.attacker.example",
            "us/../..",
            "US",
            "us_central1",
            "us central1",
            "attacker.example:443",
            "-us",
        ],
    )
    def test_malicious_location_is_rejected(self, config, malicious_location):
        """SSRF/credential-exfil guard: vertex_location is client-controllable on
        the proxy, so a host-injecting value must raise rather than steer the
        request (and its admin-minted Google bearer token) at another host."""
        with pytest.raises(VertexAIError):
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model="chirp_3",
                optional_params={},
                litellm_params={"vertex_project": "test-project", "vertex_location": malicious_location},
            )

    @pytest.mark.parametrize(
        "malicious_project",
        [
            "proj/../../locations",
            "proj/evil",
            "proj#frag",
            "proj?a=b",
            "proj:evil",
            "proj space",
        ],
    )
    def test_malicious_project_is_rejected(self, config, malicious_project):
        with pytest.raises(VertexAIError):
            config.get_complete_url(
                api_base=None,
                api_key=None,
                model="chirp_3",
                optional_params={},
                litellm_params={"vertex_project": malicious_project, "vertex_location": "us"},
            )


class TestTransformRequest:
    def test_request_body_shape(self, config):
        audio_bytes = b"fake-audio-bytes"
        request_data = config.transform_audio_transcription_request(
            model="chirp_3",
            audio_file=audio_bytes,
            optional_params={},
            litellm_params={},
        )
        assert request_data.files is None
        assert request_data.data == {
            "config": {
                "model": "chirp_3",
                "languageCodes": ["auto"],
                "features": {"enableAutomaticPunctuation": True},
                "autoDecodingConfig": {},
            },
            "content": base64.b64encode(audio_bytes).decode("utf-8"),
        }

    @pytest.mark.parametrize(
        "language,expected_language_codes",
        [
            ("en", ["en-US"]),
            ("en-US", ["en-US"]),
            ("es-ES", ["es-ES"]),
            ("fr", ["fr-FR"]),
            (None, ["auto"]),
        ],
    )
    def test_language_param_maps_to_language_codes(self, config, language, expected_language_codes):
        request_data = config.transform_audio_transcription_request(
            model="chirp_3",
            audio_file=b"fake-audio-bytes",
            optional_params={"language": language} if language is not None else {},
            litellm_params={},
        )
        assert request_data.data["config"]["languageCodes"] == expected_language_codes

    def test_model_prefix_is_stripped(self, config):
        request_data = config.transform_audio_transcription_request(
            model="vertex_ai/chirp_3",
            audio_file=b"fake-audio-bytes",
            optional_params={},
            litellm_params={},
        )
        assert request_data.data["config"]["model"] == "chirp_3"

    def test_body_is_json_serializable(self, config):
        request_data = config.transform_audio_transcription_request(
            model="chirp_3",
            audio_file=b"fake-audio-bytes",
            optional_params={},
            litellm_params={},
        )
        json.dumps(request_data.data)


class TestTransformResponse:
    def test_multi_result_transcripts_are_joined(self, config):
        raw_response = httpx.Response(
            status_code=200,
            json={
                "results": [
                    {"alternatives": [{"transcript": "Hello world.", "confidence": 0.98}], "languageCode": "en-US"},
                    {"alternatives": [{"transcript": "How are you?", "confidence": 0.97}], "languageCode": "en-US"},
                ],
                "metadata": {"totalBilledDuration": "15s"},
            },
        )
        response = config.transform_audio_transcription_response(raw_response)
        assert response.text == "Hello world. How are you?"
        assert response["task"] == "transcribe"
        assert response["language"] == "en-US"
        assert response["duration"] == 15.0

    def test_results_without_alternatives_are_skipped(self, config):
        raw_response = httpx.Response(
            status_code=200,
            json={
                "results": [
                    {"alternatives": [{"transcript": "First."}]},
                    {"alternatives": []},
                    {},
                    {"alternatives": [{"transcript": "Last."}]},
                ]
            },
        )
        response = config.transform_audio_transcription_response(raw_response)
        assert response.text == "First. Last."

    def test_empty_results_returns_empty_text(self, config):
        raw_response = httpx.Response(status_code=200, json={})
        response = config.transform_audio_transcription_response(raw_response)
        assert response.text == ""

    def test_fractional_billed_duration(self, config):
        raw_response = httpx.Response(
            status_code=200,
            json={
                "results": [{"alternatives": [{"transcript": "Hi."}]}],
                "metadata": {"totalBilledDuration": "3.5s"},
            },
        )
        response = config.transform_audio_transcription_response(raw_response)
        assert response["duration"] == 3.5


class TestValidateEnvironment:
    def test_sets_oauth_headers(self):
        class StubbedConfig(VertexAIAudioTranscriptionConfig):
            def _ensure_access_token(self, credentials, project_id, custom_llm_provider):
                return "fake-token", "resolved-project"

        headers = StubbedConfig().validate_environment(
            headers={},
            model="chirp_3",
            messages=[],
            optional_params={},
            litellm_params={"vertex_project": "resolved-project"},
        )
        assert headers["Authorization"] == "Bearer fake-token"
        assert headers["x-goog-user-project"] == "resolved-project"
        assert headers["Content-Type"] == "application/json"


class TestProviderRouting:
    def test_provider_config_manager_returns_vertex_config(self):
        provider_config = ProviderConfigManager.get_provider_audio_transcription_config(
            model="chirp_3",
            provider=LlmProviders.VERTEX_AI,
        )
        assert isinstance(provider_config, VertexAIAudioTranscriptionConfig)

    def test_get_optional_params_transcription_maps_language(self):
        optional_params = get_optional_params_transcription(
            model="chirp_3",
            custom_llm_provider="vertex_ai",
            language="fr-FR",
            response_format="json",
        )
        assert optional_params["language"] == "fr-FR"
        assert optional_params["response_format"] == "json"

    def test_get_optional_params_transcription_rejects_unsupported_param(self):
        with pytest.raises(litellm.utils.UnsupportedParamsError):
            get_optional_params_transcription(
                model="chirp_3",
                custom_llm_provider="vertex_ai",
                temperature=1,
            )

    @pytest.mark.parametrize("response_format", ["json", "text"])
    def test_supported_response_formats_pass_through(self, response_format):
        optional_params = get_optional_params_transcription(
            model="chirp_3",
            custom_llm_provider="vertex_ai",
            response_format=response_format,
        )
        assert optional_params["response_format"] == response_format

    @pytest.mark.parametrize("response_format", ["verbose_json", "srt", "vtt"])
    def test_unsupported_response_format_raises(self, response_format):
        with pytest.raises(litellm.utils.UnsupportedParamsError, match="response_format"):
            get_optional_params_transcription(
                model="chirp_3",
                custom_llm_provider="vertex_ai",
                response_format=response_format,
            )

    @pytest.mark.parametrize("response_format", ["verbose_json", "srt", "vtt"])
    def test_unsupported_response_format_dropped_with_drop_params(self, response_format):
        optional_params = get_optional_params_transcription(
            model="chirp_3",
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
    def test_chirp_3_registered_as_audio_transcription(self, cost_map_path):
        with open(os.path.join(self.REPO_ROOT, cost_map_path)) as f:
            entry = json.load(f)["vertex_ai/chirp_3"]
        assert entry["mode"] == "audio_transcription"
        assert entry["litellm_provider"] == "vertex_ai"
        assert entry["input_cost_per_second"] == pytest.approx(0.016 / 60, rel=1e-3)
        assert entry["supported_endpoints"] == ["/v1/audio/transcriptions"]
