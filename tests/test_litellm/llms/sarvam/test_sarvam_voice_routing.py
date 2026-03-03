import base64
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm import speech, transcription
from litellm.llms.sarvam.audio_transcription.transformation import (
    SarvamAudioTranscriptionConfig,
)
from litellm.llms.sarvam.common_utils import SarvamException, get_sarvam_user_agent
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


# ---------------------------------------------------------------------------
# TTS: transform_text_to_speech_response error / edge-case tests
# ---------------------------------------------------------------------------


class TestTTSTransformResponse:
    """Tests for SarvamTextToSpeechConfig.transform_text_to_speech_response"""

    def _make_response(self, *, json_body=None, text="", status_code=200):
        content = text.encode() if isinstance(text, str) else text
        if json_body is not None:
            import json

            content = json.dumps(json_body).encode()
        return httpx.Response(
            status_code=status_code,
            content=content,
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "https://api.sarvam.ai/text-to-speech"),
        )

    def test_should_decode_valid_base64_audio(self):
        config = SarvamTextToSpeechConfig()
        audio_bytes = b"fake-wav-data"
        body = {"audios": [base64.b64encode(audio_bytes).decode()]}
        resp = config.transform_text_to_speech_response(
            model="bulbul:v3",
            raw_response=self._make_response(json_body=body),
            logging_obj=MagicMock(),
        )
        assert isinstance(resp, HttpxBinaryResponseContent)
        assert resp.read() == audio_bytes

    def test_should_raise_on_empty_audios_list(self):
        config = SarvamTextToSpeechConfig()
        body = {"audios": []}
        with pytest.raises(ValueError, match="no audio data"):
            config.transform_text_to_speech_response(
                model="bulbul:v3",
                raw_response=self._make_response(json_body=body),
                logging_obj=MagicMock(),
            )

    def test_should_raise_on_missing_audios_key(self):
        config = SarvamTextToSpeechConfig()
        body = {"something_else": "value"}
        with pytest.raises(ValueError, match="no audio data"):
            config.transform_text_to_speech_response(
                model="bulbul:v3",
                raw_response=self._make_response(json_body=body),
                logging_obj=MagicMock(),
            )

    def test_should_raise_on_non_json_response(self):
        config = SarvamTextToSpeechConfig()
        with pytest.raises(SarvamException, match="Error parsing"):
            config.transform_text_to_speech_response(
                model="bulbul:v3",
                raw_response=self._make_response(text="not json"),
                logging_obj=MagicMock(),
            )


# ---------------------------------------------------------------------------
# STT: transform_audio_transcription_response error / edge-case tests
# ---------------------------------------------------------------------------


class TestSTTTransformResponse:
    """Tests for SarvamAudioTranscriptionConfig.transform_audio_transcription_response"""

    def _make_response(self, *, json_body=None, text="", status_code=200):
        content = text.encode() if isinstance(text, str) else text
        if json_body is not None:
            import json

            content = json.dumps(json_body).encode()
        return httpx.Response(
            status_code=status_code,
            content=content,
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", "https://api.sarvam.ai/speech-to-text"),
        )

    def test_should_parse_valid_transcript(self):
        config = SarvamAudioTranscriptionConfig()
        body = {"transcript": "hello world", "language_code": "en-IN"}
        resp = config.transform_audio_transcription_response(
            raw_response=self._make_response(json_body=body),
        )
        assert resp.text == "hello world"
        assert resp["language"] == "en-IN"

    def test_should_handle_missing_transcript_key(self):
        config = SarvamAudioTranscriptionConfig()
        body = {"language_code": "hi-IN"}
        resp = config.transform_audio_transcription_response(
            raw_response=self._make_response(json_body=body),
        )
        assert resp.text == ""

    def test_should_raise_on_non_json_response(self):
        config = SarvamAudioTranscriptionConfig()
        with pytest.raises(SarvamException, match="Error parsing"):
            config.transform_audio_transcription_response(
                raw_response=self._make_response(text="server error"),
            )


# ---------------------------------------------------------------------------
# STT: mode validation edge cases
# ---------------------------------------------------------------------------


class TestSTTModeValidation:
    """Tests for mode parameter validation in transform_audio_transcription_request"""

    def test_should_reject_mode_on_non_saaras_v3_model(self):
        config = SarvamAudioTranscriptionConfig()
        with pytest.raises(ValueError, match="only supported for model 'saaras:v3'"):
            config.transform_audio_transcription_request(
                model="saarika:v2.5",
                audio_file=("audio.wav", b"fake-audio", "audio/wav"),
                optional_params={"mode": "translate"},
                litellm_params={},
            )

    def test_should_reject_invalid_mode_value(self):
        config = SarvamAudioTranscriptionConfig()
        with pytest.raises(ValueError, match="Invalid mode"):
            config.transform_audio_transcription_request(
                model="saaras:v3",
                audio_file=("audio.wav", b"fake-audio", "audio/wav"),
                optional_params={"mode": "invalid_mode"},
                litellm_params={},
            )

    def test_should_reject_empty_mode_string(self):
        config = SarvamAudioTranscriptionConfig()
        with pytest.raises(ValueError, match="Invalid mode"):
            config.transform_audio_transcription_request(
                model="saaras:v3",
                audio_file=("audio.wav", b"fake-audio", "audio/wav"),
                optional_params={"mode": "  "},
                litellm_params={},
            )

    def test_should_accept_valid_mode_on_saaras_v3(self):
        config = SarvamAudioTranscriptionConfig()
        result = config.transform_audio_transcription_request(
            model="saaras:v3",
            audio_file=("audio.wav", b"fake-audio", "audio/wav"),
            optional_params={"mode": "translate"},
            litellm_params={},
        )
        assert result.data["mode"] == "translate"


# ---------------------------------------------------------------------------
# TTS: map_openai_params voice edge cases
# ---------------------------------------------------------------------------


class TestTTSMapOpenAIParams:
    """Tests for SarvamTextToSpeechConfig.map_openai_params voice handling"""

    def test_should_extract_voice_from_dict_with_name_key(self):
        config = SarvamTextToSpeechConfig()
        speaker, _ = config.map_openai_params(
            model="bulbul:v3",
            optional_params={},
            voice={"name": "Shubh"},
        )
        assert speaker == "shubh"

    def test_should_extract_voice_from_dict_with_voice_id_key(self):
        config = SarvamTextToSpeechConfig()
        speaker, _ = config.map_openai_params(
            model="bulbul:v3",
            optional_params={},
            voice={"voice_id": "Ritu"},
        )
        assert speaker == "ritu"

    def test_should_use_default_speaker_when_voice_is_none(self):
        config = SarvamTextToSpeechConfig()
        speaker, _ = config.map_openai_params(
            model="bulbul:v3",
            optional_params={},
            voice=None,
        )
        assert speaker == "shubh"

    def test_should_use_v2_default_speaker_for_v2_model(self):
        config = SarvamTextToSpeechConfig()
        speaker, _ = config.map_openai_params(
            model="bulbul:v2",
            optional_params={},
            voice=None,
        )
        assert speaker == "anushka"

    def test_should_map_speed_to_pace(self):
        config = SarvamTextToSpeechConfig()
        _, params = config.map_openai_params(
            model="bulbul:v3",
            optional_params={"speed": 1.5},
            voice="shubh",
        )
        assert params["pace"] == 1.5

    def test_should_coerce_whitespace_voice_to_empty_string(self):
        """Whitespace-only voice is coerced to '' via the non-None fallback branch."""
        config = SarvamTextToSpeechConfig()
        speaker, _ = config.map_openai_params(
            model="bulbul:v3",
            optional_params={},
            voice="   ",
        )
        assert speaker == ""


# ---------------------------------------------------------------------------
# URL construction edge cases
# ---------------------------------------------------------------------------


class TestURLConstruction:
    """Tests for get_complete_url with various api_base inputs"""

    def test_should_use_custom_api_base_without_v1(self):
        config = SarvamTextToSpeechConfig()
        url = config.get_complete_url(
            model="bulbul:v3",
            api_base="https://my-proxy.example.com",
            litellm_params={},
        )
        assert url == "https://my-proxy.example.com/text-to-speech"

    def test_should_strip_v1_from_custom_api_base(self):
        config = SarvamTextToSpeechConfig()
        url = config.get_complete_url(
            model="bulbul:v3",
            api_base="https://my-proxy.example.com/v1",
            litellm_params={},
        )
        assert url == "https://my-proxy.example.com/text-to-speech"

    def test_should_strip_v1_with_trailing_slash(self):
        config = SarvamTextToSpeechConfig()
        url = config.get_complete_url(
            model="bulbul:v3",
            api_base="https://my-proxy.example.com/v1/",
            litellm_params={},
        )
        assert url == "https://my-proxy.example.com/text-to-speech"

    def test_should_use_default_when_api_base_is_none(self):
        config = SarvamTextToSpeechConfig()
        url = config.get_complete_url(
            model="bulbul:v3",
            api_base=None,
            litellm_params={},
        )
        assert url == "https://api.sarvam.ai/text-to-speech"

    def test_should_preserve_stt_custom_api_base(self):
        config = SarvamAudioTranscriptionConfig()
        url = config.get_complete_url(
            api_base="https://my-proxy.example.com/v1",
            api_key=None,
            model="saaras:v3",
            optional_params={},
            litellm_params={},
        )
        assert url == "https://my-proxy.example.com/speech-to-text"

    def test_should_validate_environment_missing_api_key(self):
        config = SarvamTextToSpeechConfig()
        with pytest.raises(ValueError, match="API key is required"):
            config.validate_environment(
                headers={},
                model="bulbul:v3",
                api_key=None,
                api_base=None,
            )
