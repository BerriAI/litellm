import json

import pytest

import litellm
from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.llms.elevenlabs.audio_transcription.transformation import (
    ElevenLabsAudioTranscriptionConfig,
)
from litellm.utils import ProviderConfigManager


@pytest.fixture()
def config():
    return ElevenLabsAudioTranscriptionConfig()


def test_elevenlabs_config_registered():
    """Ensure ElevenLabs audio transcription config is registered."""
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="elevenlabs/scribe_v2",
        provider=litellm.LlmProviders.ELEVENLABS,
    )
    assert config is not None
    assert isinstance(config, BaseAudioTranscriptionConfig)
    assert isinstance(config, ElevenLabsAudioTranscriptionConfig)


def test_supported_openai_params_includes_prompt(config):
    """Fixes #25065 — prompt must be in supported params."""
    params = config.get_supported_openai_params("scribe_v2")
    assert "prompt" in params
    assert "language" in params
    assert "temperature" in params


def test_map_openai_params_prompt(config):
    """prompt should be mapped through to optional_params."""
    result = config.map_openai_params(
        non_default_params={"prompt": "technical terms: LiteLLM, Scribe"},
        optional_params={},
        model="scribe_v2",
        drop_params=False,
    )
    assert result["prompt"] == "technical terms: LiteLLM, Scribe"


def test_map_openai_params_language_mapped_to_language_code(config):
    """language should be mapped to language_code for ElevenLabs."""
    result = config.map_openai_params(
        non_default_params={"language": "en"},
        optional_params={},
        model="scribe_v2",
        drop_params=False,
    )
    assert result["language_code"] == "en"
    assert "language" not in result


def test_serialize_form_value_bool(config):
    """Booleans should be lowercase strings."""
    assert config._serialize_form_value(True) == "true"
    assert config._serialize_form_value(False) == "false"


def test_serialize_form_value_list(config):
    """Fixes #25066 — lists must be JSON-serialized, not Python repr."""
    value = ["term1", "term2"]
    result = config._serialize_form_value(value)
    assert result == '["term1", "term2"]'
    # Verify it's valid JSON
    assert json.loads(result) == ["term1", "term2"]


def test_serialize_form_value_dict(config):
    """Dicts must be JSON-serialized."""
    value = {"key": "val"}
    result = config._serialize_form_value(value)
    assert json.loads(result) == {"key": "val"}


def test_serialize_form_value_scalar(config):
    """Scalars should use str()."""
    assert config._serialize_form_value(0.5) == "0.5"
    assert config._serialize_form_value(42) == "42"
    assert config._serialize_form_value("hello") == "hello"


def test_transform_request_with_prompt(config):
    """prompt param should appear in form data after transform."""
    result = config.transform_audio_transcription_request(
        model="scribe_v2",
        audio_file=b"fake audio bytes",
        optional_params={"prompt": "vocabulary hint", "temperature": 0.5},
        litellm_params={},
    )
    assert result.data["prompt"] == "vocabulary hint"
    assert result.data["temperature"] == "0.5"
    assert result.data["model_id"] == "scribe_v2"


def test_transform_request_array_provider_param(config):
    """Provider-specific array params should be JSON-encoded, not Python repr."""
    result = config.transform_audio_transcription_request(
        model="scribe_v2",
        audio_file=b"fake audio bytes",
        optional_params={"keyterms": ["Atlas", "KIT"]},
        litellm_params={},
    )
    raw = result.data["keyterms"]
    assert raw == '["Atlas", "KIT"]'
    assert json.loads(raw) == ["Atlas", "KIT"]


def test_transform_request_bool_provider_param(config):
    """Provider-specific bool params should be lowercase strings."""
    result = config.transform_audio_transcription_request(
        model="scribe_v2",
        audio_file=b"fake audio bytes",
        optional_params={"diarize": True},
        litellm_params={},
    )
    assert result.data["diarize"] == "true"


def test_get_complete_url(config):
    url = config.get_complete_url(
        api_base=None,
        api_key="fake-key",
        model="scribe_v2",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.elevenlabs.io/v1/speech-to-text"


def test_get_complete_url_custom_base(config):
    url = config.get_complete_url(
        api_base="https://custom.api.example.com/",
        api_key="fake-key",
        model="scribe_v2",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://custom.api.example.com/v1/speech-to-text"
