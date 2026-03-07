"""
Unit tests for LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs,
specifically the max_tokens capping logic added to prevent HTTP 400 errors from providers
with strict output token limits (e.g. Amazon Nova Pro: 10,000 tokens).
"""
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


MESSAGES = [{"role": "user", "content": "hello"}]
MODEL = "converse/us.amazon.nova-pro-v1:0"
PROVIDER = "bedrock"
MODEL_MAX_OUTPUT = 10_000


def _call(max_tokens, extra_kwargs=None):
    """Helper: call _prepare_completion_kwargs and return the resolved max_tokens."""
    kwargs, _ = LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
        max_tokens=max_tokens,
        messages=MESSAGES,
        model=MODEL,
        extra_kwargs=extra_kwargs or {"custom_llm_provider": PROVIDER},
    )
    return kwargs["max_tokens"]


class TestMaxTokensCapping:
    def test_caps_when_exceeds_limit(self):
        """max_tokens above the model limit is silently capped to max_output_tokens."""
        with patch("litellm.get_model_info") as mock_info:
            mock_info.return_value = {"max_output_tokens": MODEL_MAX_OUTPUT}
            result = _call(max_tokens=16_000)
        assert result == MODEL_MAX_OUTPUT

    def test_unchanged_when_within_limit(self):
        """max_tokens at or below the model limit is left unchanged."""
        with patch("litellm.get_model_info") as mock_info:
            mock_info.return_value = {"max_output_tokens": MODEL_MAX_OUTPUT}
            result = _call(max_tokens=8_000)
        assert result == 8_000

    def test_unchanged_when_equal_to_limit(self):
        """max_tokens exactly equal to the model limit is left unchanged."""
        with patch("litellm.get_model_info") as mock_info:
            mock_info.return_value = {"max_output_tokens": MODEL_MAX_OUTPUT}
            result = _call(max_tokens=MODEL_MAX_OUTPUT)
        assert result == MODEL_MAX_OUTPUT

    def test_fallback_infers_provider_when_not_in_extra_kwargs(self):
        """When custom_llm_provider is absent from extra_kwargs, max_tokens is still
        capped correctly by inferring the provider from the model string."""
        with patch("litellm.utils.get_llm_provider") as mock_provider, \
             patch("litellm.get_model_info") as mock_info:
            mock_provider.return_value = (MODEL, PROVIDER, None, None)
            mock_info.return_value = {"max_output_tokens": MODEL_MAX_OUTPUT}

            result = _call(max_tokens=16_000, extra_kwargs={})

        assert result == MODEL_MAX_OUTPUT

    def test_resilient_when_get_model_info_raises(self):
        """If get_model_info raises, max_tokens is passed through unchanged."""
        with patch("litellm.get_model_info", side_effect=Exception("model not found")), \
             patch("litellm.utils.get_llm_provider", side_effect=Exception("no provider")):
            result = _call(max_tokens=16_000)
        assert result == 16_000

    def test_no_cap_when_max_output_tokens_missing(self):
        """If model_info has no max_output_tokens key, max_tokens is unchanged."""
        with patch("litellm.get_model_info") as mock_info:
            mock_info.return_value = {}
            result = _call(max_tokens=16_000)
        assert result == 16_000

    def test_no_cap_when_max_output_tokens_is_none(self):
        """Explicit None max_output_tokens does not trigger capping."""
        with patch("litellm.get_model_info") as mock_info:
            mock_info.return_value = {"max_output_tokens": None}
            result = _call(max_tokens=16_000)
        assert result == 16_000

    def test_explicit_provider_used_before_inference(self):
        """When custom_llm_provider is present, get_llm_provider is not called."""
        with patch("litellm.utils.get_llm_provider") as mock_provider, \
             patch("litellm.get_model_info") as mock_info:
            mock_info.return_value = {"max_output_tokens": MODEL_MAX_OUTPUT}
            _call(max_tokens=16_000, extra_kwargs={"custom_llm_provider": PROVIDER})
        mock_provider.assert_not_called()
