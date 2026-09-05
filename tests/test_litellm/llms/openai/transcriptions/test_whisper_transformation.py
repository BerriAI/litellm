"""
Tests for OpenAIWhisperAudioTranscriptionConfig.transform_audio_transcription_request
and transform_audio_transcription_response.
"""

import io
import json
from unittest.mock import MagicMock

import pytest

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


class TestWhisperTransformTimestampGranularities:
    """
    OpenAI's multipart form API expects array params as repeated bracketed
    fields (``timestamp_granularities[]``). If the list is sent under the bare
    ``timestamp_granularities`` key, OpenAI keeps only the last value, so a
    combined ``["segment", "word"]`` request silently returns one granularity.
    """

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

    def test_multi_value_list_uses_bracketed_key(self):
        """A multi-value list is sent under the bracketed key so both are honored."""
        data = self._transform(
            {
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment", "word"],
            }
        )
        assert data["timestamp_granularities[]"] == ["segment", "word"]
        # The bare key must not be present, otherwise OpenAI drops all but the last.
        assert "timestamp_granularities" not in data

    def test_single_value_list_uses_bracketed_key(self):
        """A single-value list is also sent under the bracketed key for consistency."""
        data = self._transform(
            {
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"],
            }
        )
        assert data["timestamp_granularities[]"] == ["word"]
        assert "timestamp_granularities" not in data

    def test_absent_granularities_not_added(self):
        """When timestamp_granularities is not provided, no bracketed key is added."""
        data = self._transform({"response_format": "verbose_json"})
        assert "timestamp_granularities[]" not in data
        assert "timestamp_granularities" not in data


class TestWhisperTransformResponse:
    def _make_response(self, *, text: str, content_type: str, is_json: bool):
        mock = MagicMock()
        mock.headers = {"content-type": content_type}
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
            self._make_response(
                text="Hello world", content_type="application/json", is_json=True
            )
        )
        assert result.text == "Hello world"

    def test_parses_plain_text_response(self):
        """Plain-text body (response_format=text) is returned as TranscriptionResponse without error."""
        config = OpenAIWhisperAudioTranscriptionConfig()
        result = config.transform_audio_transcription_response(
            self._make_response(
                text="Four score and seven years ago",
                content_type="text/plain",
                is_json=False,
            )
        )
        assert result.text == "Four score and seven years ago"

    def test_malformed_json_body_with_json_content_type_raises(self):
        """A non-JSON body labelled application/json is a genuine upstream error, not a transcription."""
        config = OpenAIWhisperAudioTranscriptionConfig()
        with pytest.raises(json.JSONDecodeError):
            config.transform_audio_transcription_response(
                self._make_response(
                    text="<html>502 Bad Gateway</html>",
                    content_type="application/json",
                    is_json=False,
                )
            )

    def test_json_content_type_match_is_case_insensitive(self):
        """Media types are case-insensitive (RFC 7231), so a mixed-case application/json still re-raises."""
        config = OpenAIWhisperAudioTranscriptionConfig()
        with pytest.raises(json.JSONDecodeError):
            config.transform_audio_transcription_response(
                self._make_response(
                    text="<html>502 Bad Gateway</html>",
                    content_type="Application/JSON; charset=utf-8",
                    is_json=False,
                )
            )
