"""
Tests for OpenRouter model name routing in get_llm_provider.

OpenRouter-native models have IDs that start with "openrouter/" (e.g.
openrouter/auto, openrouter/free, openrouter/aurora-alpha).  When a user
configures such a model in LiteLLM they use the double-prefixed form
"openrouter/openrouter/aurora-alpha".  get_llm_provider must strip only
the outer "openrouter/" provider prefix and leave the inner one intact,
so the correct model ID is sent to the OpenRouter API.

See: https://github.com/BerriAI/litellm/issues/16353
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm


class TestOpenRouterNativeModelRouting:
    """get_llm_provider must not double-strip native OpenRouter model names."""

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            # Well-known native models
            ("openrouter/openrouter/auto", "openrouter/auto"),
            ("openrouter/openrouter/free", "openrouter/free"),
            ("openrouter/openrouter/bodybuilder", "openrouter/bodybuilder"),
            # Arbitrary native models — the fix must be pattern-based, not a hardcoded list
            ("openrouter/openrouter/aurora-alpha", "openrouter/aurora-alpha"),
            ("openrouter/openrouter/polaris-alpha", "openrouter/polaris-alpha"),
            ("openrouter/openrouter/some-future-model", "openrouter/some-future-model"),
        ],
    )
    def test_double_prefixed_strips_once(self, input_model, expected_model):
        """openrouter/openrouter/<model> should yield model=openrouter/<model>."""
        result_model, provider, _, _ = litellm.get_llm_provider(model=input_model)
        assert provider == "openrouter"
        assert result_model == expected_model

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            ("openrouter/openrouter/aurora-alpha", "openrouter/aurora-alpha"),
            ("openrouter/openrouter/auto", "openrouter/auto"),
        ],
    )
    def test_no_double_strip_on_second_call(self, input_model, expected_model):
        """Simulates two consecutive get_llm_provider calls (bridge → completion)."""
        model_first, provider, _, _ = litellm.get_llm_provider(model=input_model)
        assert model_first == expected_model

        model_second, provider2, _, _ = litellm.get_llm_provider(model=model_first)
        assert provider2 == "openrouter"
        assert model_second == expected_model

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            ("openrouter/anthropic/claude-3-haiku", "anthropic/claude-3-haiku"),
            ("openrouter/meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-70b-instruct"),
        ],
    )
    def test_regular_models_still_strip_normally(self, input_model, expected_model):
        """Non-native OpenRouter models should still have their prefix stripped."""
        result_model, provider, _, _ = litellm.get_llm_provider(model=input_model)
        assert provider == "openrouter"
        assert result_model == expected_model
