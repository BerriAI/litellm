"""
Tests for OpenAIWhisperAudioTranscriptionConfig.transform_audio_transcription_request.
"""

import io

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
