import pytest

from litellm.llms.openai.transcriptions.gpt_transformation import (
    OpenAIGPTAudioTranscriptionConfig,
)
from litellm.llms.openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)
from litellm.types.utils import LlmProviders
from litellm.utils import ProviderConfigManager


@pytest.mark.parametrize(
    "model,expected_cls",
    [
        ("whisper-1", OpenAIWhisperAudioTranscriptionConfig),
        ("my-transcribe-whisper", OpenAIWhisperAudioTranscriptionConfig),
        ("gpt-4o-transcribe", OpenAIGPTAudioTranscriptionConfig),
        ("gpt-4o-mini-transcribe", OpenAIGPTAudioTranscriptionConfig),
        ("gpt-transcribe", OpenAIGPTAudioTranscriptionConfig),
    ],
)
def test_openai_transcription_config_dispatch(model, expected_cls):
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model=model, provider=LlmProviders.OPENAI
    )
    assert type(config) is expected_cls


@pytest.mark.parametrize(
    "model,optional_params,expected",
    [
        ("whisper-1", {}, "verbose_json"),
        ("whisper-1", {"response_format": "srt"}, "srt"),
        ("gpt-transcribe", {}, None),  # no forced verbose_json
        ("gpt-transcribe", {"response_format": "text"}, "text"),
    ],
)
def test_default_response_format(model, optional_params, expected):
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model=model, provider=LlmProviders.OPENAI
    )
    result = config.transform_audio_transcription_request(
        model=model,
        audio_file=b"x",
        optional_params=optional_params,
        litellm_params={},
    )
    assert result.data.get("response_format") == expected
