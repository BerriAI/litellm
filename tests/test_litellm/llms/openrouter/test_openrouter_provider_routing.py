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
        "input_model",
        [
            "openrouter/openrouter/aurora-alpha",
            "openrouter/openrouter/auto",
            "openrouter/openrouter/free",
            "openrouter/openrouter/some-future-model",
        ],
    )
    def test_bridge_double_call_preserves_native_model(self, input_model):
        """Simulates two consecutive get_llm_provider calls (bridge → completion).

        The first call (bridge) strips the outer prefix:
            openrouter/openrouter/<model> → openrouter/<model>

        The second call (completion) receives custom_llm_provider="openrouter"
        from the bridge, detects the native model, and preserves it:
            openrouter/<model> → openrouter/<model>  (no further stripping)
        """
        # First call: bridge resolves provider
        model_first, provider, _, _ = litellm.get_llm_provider(model=input_model)
        assert provider == "openrouter"
        expected_model = input_model.split("/", 1)[1]  # openrouter/<model>
        assert model_first == expected_model

        # Second call: completion receives model + custom_llm_provider from bridge
        model_second, provider2, _, _ = litellm.get_llm_provider(
            model=model_first,
            custom_llm_provider="openrouter",
        )
        assert provider2 == "openrouter"
        assert model_second == expected_model  # preserved, not stripped further

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            ("openrouter/anthropic/claude-3-haiku", "anthropic/claude-3-haiku"),
            ("openrouter/meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-70b-instruct"),
            ("openrouter/google/gemini-3-flash-preview", "google/gemini-3-flash-preview"),
        ],
    )
    def test_regular_models_still_strip_normally(self, input_model, expected_model):
        """Non-native OpenRouter models should still have their prefix stripped."""
        result_model, provider, _, _ = litellm.get_llm_provider(model=input_model)
        assert provider == "openrouter"
        assert result_model == expected_model

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            ("openrouter/google/gemini-3-flash-preview", "google/gemini-3-flash-preview"),
            ("openrouter/anthropic/claude-3-haiku", "anthropic/claude-3-haiku"),
            ("openrouter/meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-70b-instruct"),
        ],
    )
    def test_regular_model_with_custom_llm_provider_strips_prefix(
        self, input_model, expected_model
    ):
        """Regression test for https://github.com/BerriAI/litellm/issues/22667.

        When the proxy/router calls get_llm_provider with both the full model
        name and custom_llm_provider="openrouter", the "openrouter/" prefix
        must still be stripped.  Previously, an over-broad early-return for
        native OpenRouter models caused the prefix to be kept, sending
        "openrouter/google/gemini-3-flash-preview" to the API instead of
        "google/gemini-3-flash-preview".
        """
        result_model, provider, _, _ = litellm.get_llm_provider(
            model=input_model,
            custom_llm_provider="openrouter",
        )
        assert provider == "openrouter"
        assert result_model == expected_model

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            ("google/gemini-3-flash-preview", "google/gemini-3-flash-preview"),
            ("anthropic/claude-3-haiku", "anthropic/claude-3-haiku"),
        ],
    )
    def test_router_second_call_strips_prefix(self, input_model, expected_model):
        """Simulate the router's second call with already-stripped model name.

        The router first calls get_llm_provider(model="openrouter/google/...")
        which returns model="google/..." and custom_llm_provider="openrouter".
        It then passes both to litellm.completion, which calls get_llm_provider
        again.  The model must not gain back and keep the "openrouter/" prefix.
        """
        result_model, provider, _, _ = litellm.get_llm_provider(
            model=input_model,
            custom_llm_provider="openrouter",
        )
        assert provider == "openrouter"
        assert result_model == expected_model
