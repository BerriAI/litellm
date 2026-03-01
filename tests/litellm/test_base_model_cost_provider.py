"""
Tests for base_model cost calculation when base_model provider differs from
deployment provider.

Regression tests for https://github.com/BerriAI/litellm/issues/22257
"""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.cost_calculator import completion_cost
from litellm.types.utils import ModelResponse, Usage


@pytest.fixture(autouse=True)
def _load_local_model_cost_map():
    """Ensure model cost map is loaded from local file for all tests."""
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")


def _make_response(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    custom_llm_provider: str,
) -> ModelResponse:
    """Create a ModelResponse with hidden_params mimicking a real deployment."""
    response = ModelResponse(
        id="test-base-model-cost",
        model=model,
        choices=[],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )
    # Simulate hidden_params as set by litellm's main completion path.
    # This is the key part: hidden_params carries the *deployment* provider,
    # which would overwrite any earlier provider correction if the fix is not
    # placed after the hidden_params extraction.
    response._hidden_params = {
        "custom_llm_provider": custom_llm_provider,
    }
    return response


class TestBaseModelCostProviderOverride:
    """Verify that cost calculation uses the base_model's provider prefix
    instead of the deployment provider when they differ."""

    def test_base_model_with_different_provider_returns_nonzero_cost(self):
        """
        Scenario: A model is deployed via anthropic (model='anthropic/gpt-4o')
        but base_model is set to 'openai/gpt-4o' for cost lookup.

        Before the fix, custom_llm_provider stayed as 'anthropic' and the
        lookup for 'anthropic/gpt-4o' failed, returning cost 0.
        After the fix, custom_llm_provider is updated to 'openai' and the
        lookup for 'openai/gpt-4o' (which maps to 'gpt-4o') succeeds.
        """
        response = _make_response(
            model="anthropic/gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            custom_llm_provider="anthropic",
        )

        cost = completion_cost(
            completion_response=response,
            model="anthropic/gpt-4o",
            custom_llm_provider="anthropic",
            base_model="openai/gpt-4o",
        )

        assert cost > 0, (
            "Cost should be > 0 when base_model='openai/gpt-4o' is used for "
            "lookup, even though the deployment provider is 'anthropic'."
        )

    def test_base_model_cost_matches_direct_provider_cost(self):
        """
        The cost calculated via base_model='openai/gpt-4o' through a
        different deployment provider should match the cost of calling
        'gpt-4o' directly through openai.
        """
        prompt_tokens = 200
        completion_tokens = 100

        # Cost via base_model override
        response_via_base_model = _make_response(
            model="azure/my-gpt4o-deployment",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            custom_llm_provider="azure",
        )
        cost_via_base_model = completion_cost(
            completion_response=response_via_base_model,
            model="azure/my-gpt4o-deployment",
            custom_llm_provider="azure",
            base_model="openai/gpt-4o",
        )

        # Cost via direct openai call
        response_direct = _make_response(
            model="gpt-4o",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            custom_llm_provider="openai",
        )
        cost_direct = completion_cost(
            completion_response=response_direct,
            model="gpt-4o",
            custom_llm_provider="openai",
        )

        assert cost_via_base_model > 0, "base_model cost should be > 0"
        assert cost_direct > 0, "direct cost should be > 0"
        assert cost_via_base_model == pytest.approx(cost_direct, rel=1e-6), (
            f"Cost via base_model ({cost_via_base_model}) should match "
            f"direct openai cost ({cost_direct})"
        )

    def test_base_model_without_provider_prefix_does_not_change_provider(self):
        """
        When base_model has no provider prefix (e.g. 'gpt-4o'), the
        custom_llm_provider should not be changed.
        """
        response = _make_response(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            custom_llm_provider="openai",
        )

        cost = completion_cost(
            completion_response=response,
            model="gpt-4o",
            custom_llm_provider="openai",
            base_model="gpt-4o",
        )

        assert cost > 0, "Cost should be calculated normally"

    def test_base_model_same_provider_no_change(self):
        """
        When base_model has the same provider as custom_llm_provider,
        nothing should change and cost should still be correct.
        """
        response = _make_response(
            model="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            custom_llm_provider="openai",
        )

        cost = completion_cost(
            completion_response=response,
            model="gpt-4o",
            custom_llm_provider="openai",
            base_model="openai/gpt-4o",
        )

        assert cost > 0, "Cost should be calculated normally"

    def test_hidden_params_provider_is_overridden_by_base_model(self):
        """
        Explicitly verify that even when hidden_params sets
        custom_llm_provider to the wrong provider, the base_model
        provider takes precedence for cost lookup.

        This directly tests the bug described in issue #22257 where
        hidden_params['custom_llm_provider'] overwrites the provider
        after _select_model_name_for_cost_calc().
        """
        response = _make_response(
            model="anthropic/gemini-flash",
            prompt_tokens=50,
            completion_tokens=25,
            custom_llm_provider="anthropic",  # deployment provider in hidden_params
        )

        # Use a real model from the cost map as base_model
        # gpt-4o exists under openai provider
        cost = completion_cost(
            completion_response=response,
            model="anthropic/gemini-flash",
            custom_llm_provider="anthropic",
            base_model="openai/gpt-4o",
        )

        assert cost > 0, (
            "Cost should be > 0: base_model='openai/gpt-4o' provider "
            "should override hidden_params custom_llm_provider='anthropic'"
        )
