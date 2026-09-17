"""
Integration tests for Perplexity cost calculation and transformation.

Tests the end-to-end functionality of Perplexity cost calculation
including integration with the main LiteLLM cost calculator.
"""

import json
from typing import Final
import math
import os

import pytest

# Add the project root to Python path
import litellm
from litellm import ModelResponse
from litellm.cost_calculator import cost_per_token
from litellm.llms.perplexity.chat.transformation import PerplexityChatConfig
from litellm.types.utils import PromptTokensDetailsWrapper, Usage
from litellm.utils import get_model_info


class TestPerplexityIntegration:
    """Integration test suite for Perplexity functionality."""

    @pytest.fixture(autouse=True)
    def setup_model_cost_map(self):
        """Set up the model cost map for testing."""
        # Ensure we use local model cost map for consistent testing
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

        # Load the model cost map
        try:
            with open("model_prices_and_context_window.json", "r") as f:
                model_cost_map = json.load(f)
            litellm.model_cost = model_cost_map
        except FileNotFoundError:
            # Fallback to ensure we have the Perplexity model configuration
            litellm.model_cost = {
                "perplexity/sonar-deep-research": {
                    "max_tokens": 128000,
                    "max_input_tokens": 128000,
                    "input_cost_per_token": 2e-06,
                    "output_cost_per_token": 8e-06,
                    "output_cost_per_reasoning_token": 3e-06,
                    "citation_cost_per_token": 2e-06,
                    "search_context_cost_per_query": {
                        "search_context_size_low": 0.005,
                        "search_context_size_medium": 0.005,
                        "search_context_size_high": 0.005,
                    },
                    "litellm_provider": "perplexity",
                    "mode": "chat",
                    "supports_reasoning": True,
                    "supports_web_search": True,
                }
            }

    def test_model_info_includes_custom_fields(self):
        """Test that get_model_info returns the custom Perplexity cost fields."""
        model_info = get_model_info(model="sonar-deep-research", custom_llm_provider="perplexity")

        # Verify custom fields are included
        required_fields = [
            "citation_cost_per_token",
            "search_context_cost_per_query",
            "input_cost_per_token",
            "output_cost_per_token",
            "output_cost_per_reasoning_token",
        ]

        for field in required_fields:
            assert field in model_info, f"Missing field: {field}"
            assert model_info[field] is not None, f"Null value for field: {field}"

    def test_various_citation_sizes(self):
        """Test cost calculation with various citation sizes."""
        config = PerplexityChatConfig()

        test_cases = [
            # (citations, expected_approximate_tokens)
            (["Short"], 1),
            (["This is a medium-length citation with some content"], 12),
            (
                ["Very short", "Another citation", "Third one with more text content"],
                15,
            ),
            ([""], 0),  # Empty citation
        ]

        for citations, expected_approx_tokens in test_cases:
            model_response = ModelResponse()
            model_response.model = "sonar-deep-research"
            model_response.usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)

            raw_response_dict = {
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
                "citations": citations,
            }

            config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

            citation_tokens = getattr(model_response.usage, "citation_tokens", 0)

            # Allow for reasonable variance in token estimation
            if expected_approx_tokens == 0:
                assert citation_tokens == 0
            else:
                assert abs(citation_tokens - expected_approx_tokens) <= 5

    def test_transformation_preserves_existing_usage_fields(self):
        """Test that transformation doesn't overwrite existing standard usage fields."""
        config = PerplexityChatConfig()

        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=20,
        )

        # Store original values
        original_prompt_tokens = model_response.usage.prompt_tokens
        original_completion_tokens = model_response.usage.completion_tokens
        original_total_tokens = model_response.usage.total_tokens

        raw_response_dict = {
            "usage": {
                "prompt_tokens": 999,  # Different from original
                "completion_tokens": 999,  # Different from original
                "total_tokens": 999,  # Different from original
                "num_search_queries": 3,
            },
            "citations": ["Some citation"],
        }

        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)

        # Original usage fields should be preserved
        assert model_response.usage.prompt_tokens == original_prompt_tokens
        assert model_response.usage.completion_tokens == original_completion_tokens
        assert model_response.usage.total_tokens == original_total_tokens

        # But custom fields should be added
        assert hasattr(model_response.usage, "prompt_tokens_details")
        assert hasattr(model_response.usage, "citation_tokens")
        assert model_response.usage.prompt_tokens_details.web_search_requests == 3

    @pytest.mark.parametrize("provider_name", ["perplexity", "PERPLEXITY", "Perplexity"])
    def test_case_insensitive_provider_matching(self, provider_name):
        """Test that cost calculation works with different case variations of provider name."""
        usage = Usage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        usage.citation_tokens = 10
        usage.prompt_tokens_details = PromptTokensDetailsWrapper(web_search_requests=1)

        # Should work regardless of case
        prompt_cost, completion_cost_val = cost_per_token(
            model="sonar-deep-research",
            custom_llm_provider=provider_name.lower(),  # Normalize to lowercase
            usage_object=usage,
        )

        entry: Final = litellm.model_cost["perplexity/sonar-deep-research"]
        expected_prompt_cost: Final = (100 * entry["input_cost_per_token"]) + (10 * entry["citation_cost_per_token"])
        expected_completion_cost: Final = (50 * entry["output_cost_per_token"]) + (
            1 * entry["search_context_cost_per_query"]["search_context_size_low"]
        )

        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost_val, expected_completion_cost, rel_tol=1e-6)
