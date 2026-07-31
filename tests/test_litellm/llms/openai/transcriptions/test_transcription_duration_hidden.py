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
from litellm.types.utils import (
    TranscriptionResponse,
    TranscriptionUsageDurationObject,
    TranscriptionUsageTokensObject,
)


class TestDiarizedJsonUsageParsing:
    """gpt-4o-transcribe / diarized_json returns a fractional `usage.seconds`."""

    def test_fractional_duration_seconds_does_not_raise(self):
        """
        A diarized_json response carries usage={"type": "duration", "seconds": <float>}.
        OpenAI specs `seconds` as a float, so a fractional value must parse cleanly
        instead of raising and getting retried until the upstream rate-limits.
        """
        response_object = {
            "text": "speaker_1: Olá",
            "task": "transcribe",
            "duration": 295.8,
            "segments": [
                {
                    "id": "seg_001",
                    "speaker": "speaker_1",
                    "start": 0.0,
                    "end": 1.0,
                    "text": "Olá",
                    "type": "transcript.text.segment",
                }
            ],
            "usage": {"type": "duration", "seconds": 295.8},
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=TranscriptionResponse(),
            response_type="audio_transcription",
        )

        assert isinstance(result.usage, TranscriptionUsageDurationObject)
        assert result.usage.seconds == 295.8

    def test_usage_duration_object_accepts_float_seconds(self):
        assert (
            TranscriptionUsageDurationObject(type="duration", seconds=295.8).seconds
            == 295.8
        )


class TestTokensUsageParsingIsResilient:
    """
    Non-OpenAI OpenAI-compatible servers (e.g. llama.cpp) can return a
    `usage.type == "tokens"` object that omits or nulls fields OpenAI always
    sends. A non-conforming usage must never sink a successful transcription
    (regression for the input_token_details=None ValidationError in #33764).
    """

    def _convert(self, usage):
        return convert_to_model_response_object(
            response_object={"text": "hello world", "usage": usage},
            model_response_object=TranscriptionResponse(),
            response_type="audio_transcription",
        )

    def test_null_input_token_details_does_not_raise(self):
        result = self._convert(
            {
                "type": "tokens",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_token_details": None,
            }
        )
        assert result.text == "hello world"
        assert isinstance(result.usage, TranscriptionUsageTokensObject)
        assert result.usage.input_tokens == 10
        assert result.usage.input_token_details is None

    def test_missing_input_token_details_does_not_raise(self):
        result = self._convert(
            {
                "type": "tokens",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
            }
        )
        assert isinstance(result.usage, TranscriptionUsageTokensObject)
        assert result.usage.total_tokens == 15
        assert result.usage.input_token_details is None

    def test_full_token_usage_still_parses(self):
        result = self._convert(
            {
                "type": "tokens",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_token_details": {"audio_tokens": 3, "text_tokens": 7},
            }
        )
        assert isinstance(result.usage, TranscriptionUsageTokensObject)
        assert result.usage.input_token_details is not None
        assert result.usage.input_token_details.audio_tokens == 3

    def test_unparseable_usage_is_dropped_not_raised(self):
        """A tokens usage missing required counts should drop usage, not crash."""
        result = self._convert({"type": "tokens", "foo": "bar"})
        assert result.text == "hello world"
        assert getattr(result, "usage", None) is None

    def test_unknown_usage_type_is_dropped(self):
        result = self._convert({"type": "something_new", "value": 1})
        assert result.text == "hello world"
        assert getattr(result, "usage", None) is None


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
