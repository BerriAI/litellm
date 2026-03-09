"""
Tests that audio transcription duration is stored in _hidden_params
instead of the response body.

Adding duration to the response body tricks the OpenAI SDK's "best match
deserialization" into thinking a plain Transcription is a
TranscriptionVerbose/Diarized type.
"""

from unittest.mock import patch

from litellm.cost_calculator import completion_cost
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.types.utils import TranscriptionResponse


class TestTranscriptionDurationNotInResponseBody:
    """Duration calculated internally should be in _hidden_params, not in the response body."""

    def test_convert_dict_stores_internal_duration_in_hidden_params(self):
        """
        When the response dict contains _audio_transcription_duration (set by
        the handler for internally-calculated durations), it should be stored
        in _hidden_params and NOT appear in the response body.
        """
        response_object = {
            "text": "Hello world",
            "_audio_transcription_duration": 12.5,
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=TranscriptionResponse(),
            response_type="audio_transcription",
        )

        assert result._hidden_params["audio_transcription_duration"] == 12.5
        assert not hasattr(result, "_audio_transcription_duration")

    def test_convert_dict_preserves_provider_duration(self):
        """
        When the provider returns duration naturally (e.g. verbose_json format),
        it should still appear in the response body as normal.
        """
        response_object = {
            "text": "Hello world",
            "language": "en",
            "duration": 42.7,
            "segments": [],
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=TranscriptionResponse(),
            response_type="audio_transcription",
        )

        assert result.duration == 42.7

    def test_plain_json_response_has_no_duration(self):
        """
        A plain json transcription response (no verbose_json) should not have
        a duration attribute in the response body.
        """
        response_object = {
            "text": "Four score and seven years ago",
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=TranscriptionResponse(),
            response_type="audio_transcription",
        )

        duration = getattr(result, "duration", None)
        assert duration is None


class TestCostCalculatorReadsDurationFromHiddenParams:
    """The cost calculator should read duration from _hidden_params via completion_cost()."""

    @patch("litellm.cost_calculator.openai_cost_per_second")
    def test_completion_cost_uses_hidden_params_duration(self, mock_cost_fn):
        """
        completion_cost() should pass the duration from _hidden_params to
        openai_cost_per_second when calculating transcription costs.
        """
        mock_cost_fn.return_value = (0.001, 0.0)

        response = TranscriptionResponse(text="test")
        response._hidden_params = {
            "audio_transcription_duration": 17.5,
            "model": "whisper-1",
            "custom_llm_provider": "openai",
        }

        completion_cost(
            completion_response=response,
            model="whisper-1",
            call_type="atranscription",
        )

        mock_cost_fn.assert_called_once()
        _, kwargs = mock_cost_fn.call_args
        assert kwargs["duration"] == 17.5

    @patch("litellm.cost_calculator.openai_cost_per_second")
    def test_completion_cost_falls_back_to_response_duration(self, mock_cost_fn):
        """
        When _hidden_params doesn't have duration (e.g. verbose_json response
        where the provider returned it), fall back to response.duration.
        """
        mock_cost_fn.return_value = (0.001, 0.0)

        response = TranscriptionResponse(text="test")
        response._hidden_params = {
            "model": "whisper-1",
            "custom_llm_provider": "openai",
        }
        response.duration = 42.7  # type: ignore

        completion_cost(
            completion_response=response,
            model="whisper-1",
            call_type="atranscription",
        )

        mock_cost_fn.assert_called_once()
        _, kwargs = mock_cost_fn.call_args
        assert kwargs["duration"] == 42.7

    @patch("litellm.cost_calculator.openai_cost_per_second")
    def test_completion_cost_defaults_to_zero_duration(self, mock_cost_fn):
        """When neither hidden params nor response has duration, use 0.0."""
        mock_cost_fn.return_value = (0.0, 0.0)

        response = TranscriptionResponse(text="test")
        response._hidden_params = {
            "model": "whisper-1",
            "custom_llm_provider": "openai",
        }

        completion_cost(
            completion_response=response,
            model="whisper-1",
            call_type="atranscription",
        )

        mock_cost_fn.assert_called_once()
        _, kwargs = mock_cost_fn.call_args
        assert kwargs["duration"] == 0.0
