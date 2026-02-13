"""
Test supports_native_streaming() behavior for custom/unknown models.

Related issue: https://github.com/BerriAI/litellm/issues/21090
"""

import pytest

from litellm.utils import supports_native_streaming


class TestSupportsNativeStreaming:
    """Test that supports_native_streaming defaults correctly for unknown models."""

    def test_unknown_model_defaults_to_true(self):
        """
        Custom models not in model_prices_and_context_window.json should default
        to True (supports native streaming), not False.

        When returning False, the Responses API falls back to
        MockResponsesAPIStreamingIterator which silently drops
        function_call_arguments.delta events.
        """
        result = supports_native_streaming(
            model="openai/my-custom-vllm-model",
            custom_llm_provider="openai",
        )
        assert result is True

    def test_known_model_returns_correct_value(self):
        """
        Known models that are registered in the DB should return their
        actual supports_native_streaming value.
        """
        # gpt-4o is a known model that supports streaming
        result = supports_native_streaming(
            model="gpt-4o",
            custom_llm_provider="openai",
        )
        assert result is True

    def test_completely_unknown_model_defaults_to_true(self):
        """
        Even completely unknown models (no provider match) should default
        to True so that real streaming is attempted rather than silently
        dropping events with fake streaming.
        """
        result = supports_native_streaming(
            model="some-random-model-xyz-123",
            custom_llm_provider="openai",
        )
        assert result is True
