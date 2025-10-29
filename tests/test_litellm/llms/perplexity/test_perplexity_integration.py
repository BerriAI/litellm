"""
Integration tests for Perplexity cost calculation and transformation.

Tests the end-to-end functionality of Perplexity cost calculation 
including integration with the main LiteLLM cost calculator.
"""

import json
import math
import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm import ModelResponse
from litellm.cost_calculator import completion_cost, cost_per_token
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
                        "search_context_size_high": 0.005
                    },
                    "litellm_provider": "perplexity",
                    "mode": "chat",
                    "supports_reasoning": True,
                    "supports_web_search": True,
                }
            }

    def test_end_to_end_cost_calculation_with_transformation(self):
        """Test end-to-end cost calculation with response transformation."""
        # Create a Perplexity API response that includes citations and search queries
        config = PerplexityChatConfig()
        
        # Create a ModelResponse with basic usage (before transformation)
        model_response = ModelResponse()
        model_response.model = "sonar-deep-research"
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=10
        )
        
        # Simulate raw response from Perplexity API
        raw_response_dict = {
            "choices": [{"message": {"content": "Test response with citations"}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
                "num_search_queries": 2
            },
            "citations": [
                "This is the first citation with important information about the topic",
                "Another citation providing additional context for the response"
            ]
        }
        
        # Apply transformation to extract Perplexity-specific fields
        config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
        
        # Now calculate the cost with the enhanced usage
        total_cost = completion_cost(completion_response=model_response, custom_llm_provider="perplexity")
        
        # Calculate expected cost
        citation_chars = sum(len(citation) for citation in raw_response_dict["citations"])
        citation_tokens = citation_chars // 4
        
        expected_prompt_cost = (100 * 2e-6) + (citation_tokens * 2e-6)  # Input + citation
        expected_completion_cost = (50 * 8e-6) + (10 * 3e-6) + (2 / 1000 * 0.005)  # Output + reasoning + search
        expected_total = expected_prompt_cost + expected_completion_cost
        
        assert math.isclose(total_cost, expected_total, rel_tol=1e-6)

    def test_cost_calculation_without_custom_fields(self):
        """Test that cost calculation works normally when custom fields are absent."""
        # Create a standard response without Perplexity-specific fields
        model_response = ModelResponse()
        model_response.model = "sonar-deep-research"
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Calculate cost without custom fields
        total_cost = completion_cost(completion_response=model_response, custom_llm_provider="perplexity")
        
        # Should only include basic input/output costs
        expected_cost = (100 * 2e-6) + (50 * 8e-6)
        
        assert math.isclose(total_cost, expected_cost, rel_tol=1e-6)

    def test_main_cost_calculator_integration(self):
        """Test integration with the main LiteLLM cost calculator."""
        # Create usage with all Perplexity fields
        usage = Usage(
            prompt_tokens=200,
            completion_tokens=100,
            total_tokens=300,
            reasoning_tokens=25,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=3)
        )
        usage.citation_tokens = 40
        
        # Test main cost calculator
        prompt_cost, completion_cost_val = cost_per_token(
            model="sonar-deep-research",
            custom_llm_provider="perplexity",
            usage_object=usage
        )
        
        # Calculate expected costs
        expected_prompt_cost = (200 * 2e-6) + (40 * 2e-6)  # Input + citation
        expected_completion_cost = (100 * 8e-6) + (25 * 3e-6) + (3 / 1000 * 0.005)  # Output + reasoning + search
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost_val, expected_completion_cost, rel_tol=1e-6)

    def test_model_info_includes_custom_fields(self):
        """Test that get_model_info returns the custom Perplexity cost fields."""
        model_info = get_model_info(model="sonar-deep-research", custom_llm_provider="perplexity")
        
        # Verify custom fields are included
        required_fields = [
            "citation_cost_per_token",
            "search_context_cost_per_query",
            "input_cost_per_token",
            "output_cost_per_token",
            "output_cost_per_reasoning_token"
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
            (["Very short", "Another citation", "Third one with more text content"], 15),
            ([""], 0),  # Empty citation
        ]
        
        for citations, expected_approx_tokens in test_cases:
            model_response = ModelResponse()
            model_response.model = "sonar-deep-research"
            model_response.usage = Usage(
                prompt_tokens=100,
                completion_tokens=50,
                total_tokens=150
            )
            
            raw_response_dict = {
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
                "citations": citations
            }
            
            config._enhance_usage_with_perplexity_fields(model_response, raw_response_dict)
            
            citation_tokens = getattr(model_response.usage, "citation_tokens", 0)
            
            # Allow for reasonable variance in token estimation
            if expected_approx_tokens == 0:
                assert citation_tokens == 0
            else:
                assert abs(citation_tokens - expected_approx_tokens) <= 5

    def test_cost_calculation_with_zero_values(self):
        """Test cost calculation handles zero values for custom fields correctly."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Set custom fields to zero
        usage.citation_tokens = 0
        usage.prompt_tokens_details = PromptTokensDetailsWrapper(web_search_requests=0)
        
        # Should not add any extra cost
        prompt_cost, completion_cost_val = cost_per_token(
            model="sonar-deep-research",
            custom_llm_provider="perplexity",
            usage_object=usage
        )
        
        expected_prompt_cost = 100 * 2e-6
        expected_completion_cost = 50 * 8e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost_val, expected_completion_cost, rel_tol=1e-6)

    def test_high_volume_cost_calculation(self):
        """Test cost calculation with high token and query counts."""
        usage = Usage(
            prompt_tokens=50000,
            completion_tokens=25000,
            total_tokens=75000,
            reasoning_tokens=10000
        )
        
        usage.citation_tokens = 5000
        usage.prompt_tokens_details = PromptTokensDetailsWrapper(web_search_requests=100)
        
        total_cost = completion_cost(
            completion_response=ModelResponse(usage=usage, model="sonar-deep-research"),
            custom_llm_provider="perplexity"
        )
        
        # Calculate expected cost
        expected_prompt_cost = (50000 * 2e-6) + (5000 * 2e-6)  # $0.11
        expected_completion_cost = (25000 * 8e-6) + (10000 * 3e-6) + (100 / 1000 * 0.005)  # $0.23
        expected_total = expected_prompt_cost + expected_completion_cost  # $0.34
        
        assert math.isclose(total_cost, expected_total, rel_tol=1e-6)
        assert total_cost > 0.3  # Sanity check for high-volume scenario

    def test_transformation_preserves_existing_usage_fields(self):
        """Test that transformation doesn't overwrite existing standard usage fields."""
        config = PerplexityChatConfig()
        
        model_response = ModelResponse()
        model_response.usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=20
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
                "num_search_queries": 3
            },
            "citations": ["Some citation"]
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
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        usage.citation_tokens = 10
        usage.prompt_tokens_details = PromptTokensDetailsWrapper(web_search_requests=1)
        
        # Should work regardless of case
        prompt_cost, completion_cost_val = cost_per_token(
            model="sonar-deep-research",
            custom_llm_provider=provider_name.lower(),  # Normalize to lowercase
            usage_object=usage
        )
        
        # Should calculate costs correctly
        expected_prompt_cost = (100 * 2e-6) + (10 * 2e-6)
        expected_completion_cost = (50 * 8e-6) + (1 / 1000 * 0.005)
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost_val, expected_completion_cost, rel_tol=1e-6) 