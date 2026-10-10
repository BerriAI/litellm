"""
Tests for IBM WatsonX Audio Transcription.

Validates the WatsonX transcription response transformation.
"""

from unittest.mock import MagicMock

import httpx
import pytest
from pydantic import ValidationError

from litellm.llms.watsonx.audio_transcription.transformation import (
    IBMWatsonXAudioTranscriptionConfig,
)
from litellm.types.utils import TranscriptionResponse


class TestWatsonXAudioTranscription:
    def test_transform_audio_transcription_response_removes_model_field(self):
        """
        Test that transform_audio_transcription_response removes the 'model' field
        from WatsonX response before creating TranscriptionResponse.

        This test ensures that when WatsonX returns a response with a 'model' field,
        it is removed before creating the TranscriptionResponse object, since
        TranscriptionResponse doesn't accept a 'model' parameter.
        """
        handler = IBMWatsonXAudioTranscriptionConfig()

        # Mock response with 'model' field (as WatsonX may return)
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Hello, this is a test transcription.",
            "model": "whisper-large-v3-turbo",  # This field should be removed
            "duration": 5.5,
        }
        mock_response.text = '{"text": "Hello, this is a test transcription.", "model": "whisper-large-v3-turbo", "duration": 5.5}'

        # This should not raise a TypeError - model field should be removed
        result = handler.transform_audio_transcription_response(mock_response)

        # Verify the result is a TranscriptionResponse
        assert isinstance(result, TranscriptionResponse)

        # Verify the text is correct
        assert result.text == "Hello, this is a test transcription."

        # Verify duration is set via dictionary assignment
        assert result["duration"] == 5.5

        # Verify the model field is NOT in the serialized result
        # Check via model_dump() or dict() to ensure it's not in the output
        try:
            result_dict = result.model_dump()
        except AttributeError:
            # Fallback for pydantic v1
            result_dict = result.dict()

        # The 'model' field should not be in the result
        assert "model" not in result_dict, "Model field should be removed from response"

    def test_transform_audio_transcription_response_without_model_field(self):
        """
        Test that transform_audio_transcription_response works correctly
        when WatsonX response doesn't include a 'model' field.
        """
        handler = IBMWatsonXAudioTranscriptionConfig()

        # Mock response without 'model' field
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "text": "Hello, this is a test transcription.",
            "duration": 5.5,
        }
        mock_response.text = (
            '{"text": "Hello, this is a test transcription.", "duration": 5.5}'
        )

        result = handler.transform_audio_transcription_response(mock_response)

        # Verify the result is a TranscriptionResponse
        assert isinstance(result, TranscriptionResponse)

        # Verify the text is correct
        assert result.text == "Hello, this is a test transcription."

        # Verify duration is set via dictionary assignment
        assert result["duration"] == 5.5


def _transform_transcription_response(payload: object) -> TranscriptionResponse:
    return IBMWatsonXAudioTranscriptionConfig().transform_audio_transcription_response(
        httpx.Response(200, json=payload)
    )


@pytest.mark.parametrize(
    ("payload", "expected_extras"),
    [
        ({"text": "hello"}, {}),
        ({"text": "hello", "model": "whisper-large-v3-turbo"}, {}),
        (
            {"text": "hello", "duration": 1.5, "language": "en", "task": "transcribe"},
            {"duration": 1.5, "language": "en", "task": "transcribe"},
        ),
        (
            {"text": "hello", "segments": [{"id": 0, "text": "hello"}], "words": None},
            {"segments": [{"id": 0, "text": "hello"}], "words": None},
        ),
    ],
)
def test_transform_audio_transcription_response_copies_every_field_except_model(
    payload: dict[str, object], expected_extras: dict[str, object]
) -> None:
    response = _transform_transcription_response(payload)

    assert response.text == "hello"
    assert not hasattr(response, "model")
    assert {key: response[key] for key in expected_extras} == expected_extras


@pytest.mark.parametrize("payload", [{}, {"model": "whisper"}, {"duration": 2.0}, {"text": None, "usage": None}])
def test_transform_audio_transcription_response_without_text_or_usage_reports_the_body(
    payload: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="Invalid response format") as exc_info:
        _transform_transcription_response(payload)

    assert exc_info.value.args == (
        "Invalid response format. Received response does not match the expected format. Got: ",
        payload,
    )


@pytest.mark.parametrize("payload", [["not", "an", "object"], "plain text", 7, True])
def test_transform_audio_transcription_response_rejects_non_object_bodies_without_echoing_them(
    payload: object,
) -> None:
    with pytest.raises(ValidationError) as exc_info:
        _transform_transcription_response(payload)

    assert "input_value" not in str(exc_info.value)
