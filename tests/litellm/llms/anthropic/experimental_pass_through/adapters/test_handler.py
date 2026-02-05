"""
Tests for the Anthropic messages adapter handler.

Verifies that models whose ID on the provider contains a prefix matching
a known LiteLLM provider name (e.g. "openrouter/free" on OpenRouter) are
not double-stripped when the adapter delegates to litellm.completion.

See: https://github.com/BerriAI/litellm/issues/16353
"""

import pytest

import litellm
from litellm.llms.anthropic.experimental_pass_through.adapters.handler import (
    LiteLLMMessagesToCompletionTransformationHandler,
)


class TestPrepareCompletionKwargsModelReconstruction:
    """_prepare_completion_kwargs reconstructs the fully-qualified model name."""

    def test_openrouter_native_model_is_reconstructed(self):
        """openrouter/free (already stripped) becomes openrouter/openrouter/free."""
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                model="openrouter/free",
                extra_kwargs={"custom_llm_provider": "openrouter"},
            )
        )
        assert completion_kwargs["model"] == "openrouter/openrouter/free"

    def test_third_party_model_on_openrouter_is_reconstructed(self):
        """anthropic/claude-3-sonnet becomes openrouter/anthropic/claude-3-sonnet."""
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                model="anthropic/claude-3-sonnet",
                extra_kwargs={"custom_llm_provider": "openrouter"},
            )
        )
        assert completion_kwargs["model"] == "openrouter/anthropic/claude-3-sonnet"

    def test_model_without_provider_is_reconstructed(self):
        """gpt-4 with provider openai becomes openai/gpt-4."""
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4",
                extra_kwargs={"custom_llm_provider": "openai"},
            )
        )
        assert completion_kwargs["model"] == "openai/gpt-4"

    def test_no_provider_leaves_model_unchanged(self):
        """Without custom_llm_provider, model is not modified."""
        completion_kwargs, _ = (
            LiteLLMMessagesToCompletionTransformationHandler._prepare_completion_kwargs(
                max_tokens=10,
                messages=[{"role": "user", "content": "hi"}],
                model="gpt-4",
                extra_kwargs={},
            )
        )
        assert completion_kwargs["model"] == "gpt-4"


class TestGetLlmProviderDoubleStripPrevention:
    """Reconstructing the model name prevents double-stripping by get_llm_provider."""

    @pytest.mark.parametrize(
        "original_model,expected_after_strip",
        [
            ("openrouter/openrouter/free", "openrouter/free"),
            ("openrouter/openrouter/bodybuilder", "openrouter/bodybuilder"),
            ("openrouter/anthropic/claude-3-sonnet", "anthropic/claude-3-sonnet"),
        ],
    )
    def test_reconstructed_model_survives_second_strip(
        self, original_model, expected_after_strip
    ):
        """Simulates the full adapter flow: strip → reconstruct → strip."""
        # 1st strip (handler calls get_llm_provider)
        model, provider, _, _ = litellm.get_llm_provider(model=original_model)
        assert model == expected_after_strip

        # Reconstruct (the fix)
        reconstructed = f"{provider}/{model}"

        # 2nd strip (litellm.acompletion calls get_llm_provider)
        model2, provider2, _, _ = litellm.get_llm_provider(model=reconstructed)
        assert model2 == expected_after_strip
        assert provider2 == provider
