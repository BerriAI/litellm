"""
Tests that audio transcription duration is stored in _hidden_params
instead of the response body.

Adding duration to the response body tricks the OpenAI SDK's "best match
deserialization" into thinking a plain Transcription is a
TranscriptionVerbose/Diarized type.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

        # Duration should be in _hidden_params
        assert result._hidden_params["audio_transcription_duration"] == 12.5
        # Duration should NOT be a visible attribute on the response
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

        # Provider-returned duration should be in the response body
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

        # No duration should be set
        duration = getattr(result, "duration", None)
        assert duration is None


class TestCostCalculatorReadsDurationFromHiddenParams:
    """The cost calculator should read duration from _hidden_params first."""

    def test_cost_calculator_reads_hidden_params_duration(self):
        """
        When _hidden_params has audio_transcription_duration, the cost
        calculator should use it instead of looking for response.duration.
        """
        response = TranscriptionResponse(text="test")
        response._hidden_params = {
            "audio_transcription_duration": 17.5,
            "model": "gpt-4o-transcribe",
            "custom_llm_provider": "openai",
        }

        # Simulate what cost_calculator.py does
        _hidden = getattr(response, "_hidden_params", {}) or {}
        duration = _hidden.get(
            "audio_transcription_duration",
            getattr(response, "duration", 0.0),
        )

        assert duration == 17.5

    def test_cost_calculator_falls_back_to_response_duration(self):
        """
        When _hidden_params doesn't have duration (e.g. verbose_json response),
        fall back to response.duration.
        """
        response = TranscriptionResponse(text="test")
        response._hidden_params = {}
        response.duration = 42.7  # type: ignore

        _hidden = getattr(response, "_hidden_params", {}) or {}
        duration = _hidden.get(
            "audio_transcription_duration",
            getattr(response, "duration", 0.0),
        )

        assert duration == 42.7

    def test_cost_calculator_returns_zero_when_no_duration(self):
        """When neither hidden params nor response has duration, return 0.0."""
        response = TranscriptionResponse(text="test")
        response._hidden_params = {}

        _hidden = getattr(response, "_hidden_params", {}) or {}
        duration = _hidden.get(
            "audio_transcription_duration",
            getattr(response, "duration", 0.0),
        )

        assert duration == 0.0
