"""
Tests for OrcaRouter model name routing in get_llm_provider.

OrcaRouter follows the same routing convention as OpenRouter: nested model IDs
like `orcarouter/openai/gpt-5` should be sent to the upstream as `openai/gpt-5`,
while the router-level `orcarouter/auto` must stay intact so the OrcaRouter
backend recognizes it as the adaptive router.

Mirrors `test_openrouter_provider_routing.py` patterns. See:
https://docs.orcarouter.ai
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider


class TestOrcaRouterPrefixStripping:
    """When custom_llm_provider is already 'orcarouter', strip the outer prefix only."""

    @pytest.mark.parametrize(
        "input_model,expected_model",
        [
            ("orcarouter/openai/gpt-5", "openai/gpt-5"),
            ("orcarouter/anthropic/claude-opus-4.7", "anthropic/claude-opus-4.7"),
            (
                "orcarouter/google/gemini-3-flash-preview",
                "google/gemini-3-flash-preview",
            ),
            ("orcarouter/deepseek/deepseek-v4-pro", "deepseek/deepseek-v4-pro"),
            ("orcarouter/openai/dall-e-3", "openai/dall-e-3"),
        ],
    )
    def test_strip_outer_prefix_when_vendor_model_present(
        self, input_model, expected_model
    ):
        model, provider, _, _ = get_llm_provider(
            model=input_model, custom_llm_provider="orcarouter"
        )
        assert model == expected_model
        assert provider == "orcarouter"

    def test_orcarouter_auto_preserved(self):
        """`orcarouter/auto` has no inner slash after strip — must stay intact."""
        model, provider, _, _ = get_llm_provider(
            model="orcarouter/auto", custom_llm_provider="orcarouter"
        )
        assert model == "orcarouter/auto"
        assert provider == "orcarouter"

    def test_double_prefix_strips_once(self):
        """`orcarouter/orcarouter/auto` → strip outer only, keep inner."""
        model, provider, _, _ = get_llm_provider(
            model="orcarouter/orcarouter/auto",
            custom_llm_provider="orcarouter",
        )
        assert model == "orcarouter/auto"
        assert provider == "orcarouter"


class TestOrcaRouterBareModelFallback:
    """When a model is registered in orcarouter_models, infer the provider."""

    def setup_method(self):
        # Register a test model
        self._test_model = "test-orcarouter-bare-model-xyz"
        litellm.orcarouter_models.add(self._test_model)

    def teardown_method(self):
        litellm.orcarouter_models.discard(self._test_model)

    def test_bare_model_resolves_to_orcarouter(self):
        model, provider, _, _ = get_llm_provider(model=self._test_model)
        assert provider == "orcarouter"


class TestOrcaRouterRegistration:
    """Smoke test: provider is wired through enum + provider list."""

    def test_provider_in_chat_providers_list(self):
        from litellm.constants import LITELLM_CHAT_PROVIDERS

        assert "orcarouter" in LITELLM_CHAT_PROVIDERS

    def test_llm_providers_enum_has_orcarouter(self):
        from litellm.types.utils import LlmProviders

        assert LlmProviders.ORCAROUTER == "orcarouter"

    def test_models_by_provider_has_orcarouter_key(self):
        assert "orcarouter" in litellm.models_by_provider

    def test_orcarouter_key_attribute_exists(self):
        assert hasattr(litellm, "orcarouter_key")
        assert hasattr(litellm, "orcarouter_models")
