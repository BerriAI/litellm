"""
Tests for OpenAIWhisperAudioTranscriptionConfig.transform_audio_transcription_request
and transform_audio_transcription_response.
"""

import io
import json
from unittest.mock import MagicMock

from litellm.llms.openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)


class TestWhisperTransformRequestResponseFormat:
    def _transform(self, optional_params: dict) -> dict:
        config = OpenAIWhisperAudioTranscriptionConfig()
        audio_file = io.BytesIO(b"fake audio")
        audio_file.name = "test.wav"
        result = config.transform_audio_transcription_request(
            model="whisper-1",
            audio_file=audio_file,
            optional_params=optional_params,
            litellm_params={},
        )
        return result.data

    def test_defaults_to_verbose_json_when_unset(self):
        """When response_format is not specified, default to verbose_json for cost calculation."""
        data = self._transform({})
        assert data["response_format"] == "verbose_json"

    def test_respects_explicit_json(self):
        """When response_format='json' is set, do not override to verbose_json."""
        data = self._transform({"response_format": "json"})
        assert data["response_format"] == "json"

    def test_respects_explicit_text(self):
        """When response_format='text' is set, do not override to verbose_json."""
        data = self._transform({"response_format": "text"})
        assert data["response_format"] == "text"

    def test_preserves_verbose_json_when_set(self):
        """verbose_json explicitly set by the caller stays as-is."""
        data = self._transform({"response_format": "verbose_json"})
        assert data["response_format"] == "verbose_json"


class TestWhisperTransformResponse:
    def _make_response(self, *, text: str, is_json: bool):
        mock = MagicMock()
        if is_json:
            mock.json.return_value = {"text": text}
        else:
            mock.json.side_effect = json.JSONDecodeError("", "", 0)
            mock.text = text
        return mock

    def test_parses_json_response(self):
        """JSON body (verbose_json or json format) is parsed into TranscriptionResponse."""
        config = OpenAIWhisperAudioTranscriptionConfig()
        result = config.transform_audio_transcription_response(
            self._make_response(text="Hello world", is_json=True)
        )
        assert result.text == "Hello world"

    def test_parses_plain_text_response(self):
        """Plain-text body (response_format=text) is returned as TranscriptionResponse without error."""
        config = OpenAIWhisperAudioTranscriptionConfig()
        result = config.transform_audio_transcription_response(
            self._make_response(text="Four score and seven years ago", is_json=False)
        )
        assert result.text == "Four score and seven years ago"
