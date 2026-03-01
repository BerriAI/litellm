from unittest.mock import patch

import httpx

import litellm
from litellm import speech, transcription
from litellm.llms.sarvam.audio_transcription.transformation import (
    SarvamAudioTranscriptionConfig,
)
from litellm.llms.sarvam.common_utils import get_sarvam_user_agent
from litellm.llms.sarvam.text_to_speech.transformation import SarvamTextToSpeechConfig
from litellm.types.llms.openai import HttpxBinaryResponseContent
from litellm.types.utils import TranscriptionResponse
from litellm.utils import ProviderConfigManager


def test_sarvam_audio_transcription_config_registered():
    config = ProviderConfigManager.get_provider_audio_transcription_config(
        model="saaras:v3",
        provider=litellm.LlmProviders.SARVAM,
    )

    assert config is not None
    assert isinstance(config, SarvamAudioTranscriptionConfig)


def test_sarvam_text_to_speech_config_registered():
    config = ProviderConfigManager.get_provider_text_to_speech_config(
        model="bulbul:v3",
        provider=litellm.LlmProviders.SARVAM,
    )

    assert config is not None
    assert isinstance(config, SarvamTextToSpeechConfig)


def test_sarvam_tts_complete_url_drops_v1_suffix():
    config = SarvamTextToSpeechConfig()

    assert (
        config.get_complete_url(
            model="bulbul:v3",
            api_base="https://api.sarvam.ai/v1",
            litellm_params={},
        )
        == "https://api.sarvam.ai/text-to-speech"
    )


def test_sarvam_stt_complete_url_drops_v1_suffix():
    config = SarvamAudioTranscriptionConfig()

    assert (
        config.get_complete_url(
            api_base="https://api.sarvam.ai/v1",
            api_key=None,
            model="saaras:v3",
            optional_params={},
            litellm_params={},
        )
        == "https://api.sarvam.ai/speech-to-text"
    )


def test_sarvam_voice_headers_include_user_agent():
    stt_config = SarvamAudioTranscriptionConfig()
    tts_config = SarvamTextToSpeechConfig()
    expected_user_agent = get_sarvam_user_agent()

    stt_headers = stt_config.validate_environment(
        headers={},
        model="saaras:v3",
        messages=[],
        optional_params={},
        litellm_params={},
        api_key="test-key",
    )
    tts_headers = tts_config.validate_environment(
        headers={},
        model="bulbul:v3",
        api_key="test-key",
    )

    assert stt_headers["User-Agent"] == expected_user_agent
    assert tts_headers["User-Agent"] == expected_user_agent


@patch("litellm.main.openai_chat_completions.audio_speech")
@patch("litellm.main.base_llm_http_handler.text_to_speech_handler")
def test_speech_routes_sarvam_to_base_http_handler(
    mock_base_tts_handler, mock_openai_tts_handler
):
    mock_base_tts_handler.return_value = HttpxBinaryResponseContent(
        httpx.Response(
            status_code=200,
            content=b"audio-bytes",
            request=httpx.Request("POST", "https://api.sarvam.ai/text-to-speech"),
        )
    )

    response = speech(
        model="sarvam/bulbul:v2",
        input="Hello from Sarvam",
        voice="anushka",
        api_key="test-key",
    )

    assert isinstance(response, HttpxBinaryResponseContent)
    assert mock_base_tts_handler.called
    mock_openai_tts_handler.assert_not_called()


@patch("litellm.main.openai_audio_transcriptions.audio_transcriptions")
@patch("litellm.main.base_llm_http_handler.audio_transcriptions")
def test_transcription_routes_sarvam_to_base_http_handler(
    mock_base_transcription_handler, mock_openai_transcription_handler
):
    mock_base_transcription_handler.return_value = TranscriptionResponse(text="hello")

    response = transcription(
        model="sarvam/saaras:v3",
        file=("audio.wav", b"test-audio-bytes"),
        api_key="test-key",
    )

    assert isinstance(response, TranscriptionResponse)
    assert response.text == "hello"
    assert mock_base_transcription_handler.called
    mock_openai_transcription_handler.assert_not_called()
