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
from litellm.llms.openrouter.chat.transformation import OpenrouterConfig


class TestOpenRouterTransformRequest:
    """OpenrouterConfig.transform_request must strip doubled openrouter/ prefix."""

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            # Non-native: prefix must be stripped
            ("openrouter/anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
            ("openrouter/meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-70b-instruct"),
            # Native: prefix must be preserved
            ("openrouter/auto", "openrouter/auto"),
            ("openrouter/free", "openrouter/free"),
            # No prefix: unchanged
            ("anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
        ],
    )
    def test_transform_request_strips_non_native_prefix(self, input_model, expected_model):
        config = OpenrouterConfig()
        result = config.transform_request(
            model=input_model,
            messages=[{"role": "user", "content": "hi"}],
            optional_params={},
            litellm_params={},
            headers={},
        )
        assert result["model"] == expected_model


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
            ("anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
            ("anthropic/claude-3-haiku", "anthropic/claude-3-haiku"),
            ("meta-llama/llama-3-70b-instruct", "meta-llama/llama-3-70b-instruct"),
            # Pre-prefixed model passed with explicit custom_llm_provider
            ("openrouter/anthropic/claude-sonnet-4.5", "anthropic/claude-sonnet-4.5"),
        ],
    )
    def test_bridge_strips_prefix_for_non_native_models(
        self, input_model, expected_model
    ):
        """Simulates the /v1/messages adapter bridge path for non-native models.

        When the anthropic_messages adapter resolves an OpenRouter model, it
        calls get_llm_provider a second time with the already-resolved model
        and custom_llm_provider="openrouter".  The prepend logic adds the
        "openrouter/" prefix back, creating e.g.
        "openrouter/anthropic/claude-sonnet-4.5".  This prefix MUST be
        stripped before the early return, otherwise the OpenRouter API receives
        an invalid model ID and returns 400.

        This is the regression introduced by PR #20516 and reported in #16353.
        """
        # Second call in the bridge chain: model already stripped, provider known
        result_model, provider, _, _ = litellm.get_llm_provider(
            model=input_model,
            custom_llm_provider="openrouter",
        )
        assert provider == "openrouter"
        assert result_model == expected_model
