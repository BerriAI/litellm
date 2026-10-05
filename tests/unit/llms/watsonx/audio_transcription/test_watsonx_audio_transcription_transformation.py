"""
Tests for IBM WatsonX Audio Transcription.

Validates the WatsonX transcription response transformation.
"""

from unittest.mock import MagicMock

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
