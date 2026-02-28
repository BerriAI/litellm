"""
Tests for native OpenRouter model name handling in get_llm_provider.

OpenRouter's native models (openrouter/auto, openrouter/free,
openrouter/bodybuilder) should not have their "openrouter/" prefix
stripped when passed to get_llm_provider(), since that prefix is part
of the actual model ID on OpenRouter's API.

"""

import pytest

import litellm


class TestNativeOpenRouterModelsNotStripped:
    """get_llm_provider must preserve native OpenRouter model names."""

    @pytest.mark.parametrize(
        "model",
        [
            "openrouter/auto",
            "openrouter/free",
            "openrouter/bodybuilder",
        ],
    )
    def test_native_model_not_stripped(self, model):
        """Native OpenRouter model IDs are returned as-is."""
        result_model, provider, _, _ = litellm.get_llm_provider(model=model)
        assert result_model == model
        assert provider == "openrouter"

    @pytest.mark.parametrize(
        "model,expected_model",
        [
            ("openrouter/openrouter/free", "openrouter/free"),
            ("openrouter/openrouter/auto", "openrouter/auto"),
            ("openrouter/openrouter/bodybuilder", "openrouter/bodybuilder"),
        ],
    )
    def test_double_prefixed_model_strips_once_to_native(self, model, expected_model):
        """openrouter/openrouter/free strips to openrouter/free (not further)."""
        result_model, provider, _, _ = litellm.get_llm_provider(model=model)
        assert result_model == expected_model
        assert provider == "openrouter"

    @pytest.mark.parametrize(
        "model,expected_model",
        [
            ("openrouter/openrouter/free", "openrouter/free"),
            ("openrouter/openrouter/auto", "openrouter/auto"),
        ],
    )
    def test_full_round_trip_no_double_strip(self, model, expected_model):
        """Simulates the bridge flow: two consecutive get_llm_provider calls."""
        # First call (in adapter/handler)
        model_after_first, provider, _, _ = litellm.get_llm_provider(model=model)
        assert model_after_first == expected_model

        # Second call (inside litellm.completion)
        model_after_second, provider2, _, _ = litellm.get_llm_provider(
            model=model_after_first
        )
        # Should stay as native model, not stripped further
        assert model_after_second == expected_model
        assert provider2 == "openrouter"

    def test_regular_openrouter_model_still_strips_normally(self):
        """Non-native models like openrouter/anthropic/claude-3-haiku still strip normally."""
        model, provider, _, _ = litellm.get_llm_provider(
            model="openrouter/anthropic/claude-3-haiku"
        )
        assert provider == "openrouter"
        # Should strip the openrouter/ prefix
        assert model == "anthropic/claude-3-haiku"
