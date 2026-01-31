"""
Test file for Perplexity cost calculator functionality.

Tests the cost calculation for Perplexity models including citation tokens, 
search queries, and reasoning tokens.
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
from litellm.cost_calculator import completion_cost, cost_per_token
from litellm.llms.perplexity.cost_calculator import cost_per_token as perplexity_cost_per_token
from litellm.types.utils import Usage, PromptTokensDetailsWrapper
from litellm.utils import get_model_info


class TestPerplexityCostCalculator:
    """Test suite for Perplexity cost calculation functionality."""

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

    def test_basic_cost_calculation(self):
        """Test basic cost calculation without additional fields."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-deep-research", 
            usage=usage
        )
        
        # Expected costs:
        # Input: 100 tokens * $2e-6 = $0.0002
        # Output: 50 tokens * $8e-6 = $0.0004
        expected_prompt_cost = 100 * 2e-6
        expected_completion_cost = 50 * 8e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)

    def test_citation_tokens_cost_calculation(self):
        """Test cost calculation with citation tokens."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Add citation tokens
        usage.citation_tokens = 25
        
        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-deep-research", 
            usage=usage
        )
        
        # Expected costs:
        # Input: 100 tokens * $2e-6 = $0.0002
        # Citation: 25 tokens * $2e-6 = $0.00005
        # Total prompt cost: $0.00025
        # Output: 50 tokens * $8e-6 = $0.0004
        expected_prompt_cost = (100 * 2e-6) + (25 * 2e-6)
        expected_completion_cost = 50 * 8e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)

    def test_search_queries_cost_calculation(self):
        """Test cost calculation with search queries."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=3)
        )
        
        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-deep-research", 
            usage=usage
        )
        
        # Expected costs:
        # Input: 100 tokens * $2e-6 = $0.0002
        # Output: 50 tokens * $8e-6 = $0.0004
        # Search: 3 queries * ($0.005 / 1000) = $0.000015
        # Total completion cost: $0.000415
        expected_prompt_cost = 100 * 2e-6
        expected_completion_cost = (50 * 8e-6) + (3 / 1000 * 0.005)
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)

    def test_reasoning_tokens_from_direct_attribute(self):
        """Test reasoning tokens cost calculation from direct attribute."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        
        # Set reasoning tokens directly
        usage.reasoning_tokens = 20
        
        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-deep-research", 
            usage=usage
        )
        
        # Expected costs:
        # Input: 100 tokens * $2e-6 = $0.0002
        # Output: 50 tokens * $8e-6 = $0.0004
        # Reasoning: 20 tokens * $3e-6 = $0.00006
        # Total completion cost: $0.00046
        expected_prompt_cost = 100 * 2e-6
        expected_completion_cost = (50 * 8e-6) + (20 * 3e-6)
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)

    def test_reasoning_tokens_from_completion_tokens_details(self):
        """Test reasoning tokens cost calculation from completion_tokens_details."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=20  # This should be stored in completion_tokens_details
        )
        
        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-deep-research", 
            usage=usage
        )
        
        # Expected costs:
        # Input: 100 tokens * $2e-6 = $0.0002
        # Output: 50 tokens * $8e-6 = $0.0004
        # Reasoning: 20 tokens * $3e-6 = $0.00006
        # Total completion cost: $0.00046
        expected_prompt_cost = 100 * 2e-6
        expected_completion_cost = (50 * 8e-6) + (20 * 3e-6)
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)

    def test_comprehensive_cost_calculation(self):
        """Test cost calculation with all fields combined."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=15,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=2)
        )
        
        # Add custom fields
        usage.citation_tokens = 30
        
        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-deep-research", 
            usage=usage
        )
        
        # Expected costs:
        # Input: 100 tokens * $2e-6 = $0.0002
        # Citation: 30 tokens * $2e-6 = $0.00006
        # Total prompt cost: $0.00026
        # Output: 50 tokens * $8e-6 = $0.0004
        # Reasoning: 15 tokens * $3e-6 = $0.000045
        # Search: 2 queries * ($0.005 / 1000) = $0.00001
        # Total completion cost: $0.000455
        expected_prompt_cost = (100 * 2e-6) + (30 * 2e-6)
        expected_completion_cost = (50 * 8e-6) + (15 * 3e-6) + (2 / 1000 * 0.005)
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)

    def test_zero_values_handling(self):
        """Test that zero or missing values are handled correctly."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=0)
        )
        
        # These should not raise errors and should not affect cost
        usage.citation_tokens = 0
        
        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-deep-research", 
            usage=usage
        )
        
        # Should be same as basic calculation
        expected_prompt_cost = 100 * 2e-6
        expected_completion_cost = 50 * 8e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)

    def test_missing_model_info_fields(self):
        """Test behavior when model info is missing some fields."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=2)
        )
        
        usage.citation_tokens = 25
        
        # Mock get_model_info to return incomplete model info
        with patch('litellm.llms.perplexity.cost_calculator.get_model_info') as mock_get_model_info:
            mock_get_model_info.return_value = {
                "input_cost_per_token": 2e-6,
                "output_cost_per_token": 8e-6,
                # Missing search_queries_cost_per_query
            }
            
            prompt_cost, completion_cost = perplexity_cost_per_token(
                model="sonar-deep-research", 
                usage=usage
            )
            
            # Should only calculate basic costs when fields are missing
            expected_prompt_cost = 100 * 2e-6
            expected_completion_cost = 50 * 8e-6
            
            assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
            assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)

    def test_integration_with_main_cost_calculator(self):
        """Test integration with the main LiteLLM cost calculator."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=10,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=1)
        )
        
        usage.citation_tokens = 20
        
        # Test main cost calculator
        prompt_cost, completion_cost_val = cost_per_token(
            model="sonar-deep-research",
            custom_llm_provider="perplexity",
            usage_object=usage
        )
        
        # Should match direct call to perplexity cost calculator
        expected_prompt, expected_completion = perplexity_cost_per_token(
            model="sonar-deep-research", 
            usage=usage
        )
        
        assert math.isclose(prompt_cost, expected_prompt, rel_tol=1e-6)
        assert math.isclose(completion_cost_val, expected_completion, rel_tol=1e-6)

    def test_integration_with_completion_cost_function(self):
        """Test integration with the completion_cost function."""
        from litellm import ModelResponse
        
        # Create a mock ModelResponse
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=10,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=1)
        )
        usage.citation_tokens = 15
        
        response = ModelResponse()
        response.usage = usage
        response.model = "sonar-deep-research"
        
        # Test completion_cost function
        total_cost = completion_cost(completion_response=response, custom_llm_provider="perplexity")
        
        # Calculate expected total cost
        expected_prompt_cost = (100 * 2e-6) + (15 * 2e-6)  # Input + citation
        expected_completion_cost = (50 * 8e-6) + (10 * 3e-6) + (1 / 1000 * 0.005)  # Output + reasoning + search
        expected_total = expected_prompt_cost + expected_completion_cost
        
        assert math.isclose(total_cost, expected_total, rel_tol=1e-6)

    def test_model_info_access(self):
        """Test that model info correctly returns the new cost fields."""
        model_info = get_model_info(model="sonar-deep-research", custom_llm_provider="perplexity")
        
        # Check that the new fields are accessible
        assert "citation_cost_per_token" in model_info
        assert model_info["citation_cost_per_token"] == 2e-6
        assert model_info["search_context_cost_per_query"] == {
            "search_context_size_low": 0.005,
            "search_context_size_medium": 0.005,
            "search_context_size_high": 0.005
        }

    @pytest.mark.parametrize("citation_tokens", [0, 10, 25, 100])
    @pytest.mark.parametrize("search_queries", [0, 1, 5, 10])
    @pytest.mark.parametrize("reasoning_tokens", [0, 15, 30])
    def test_cost_calculation_combinations(self, citation_tokens, search_queries, reasoning_tokens):
        """Test various combinations of citation tokens, search queries, and reasoning tokens."""
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            reasoning_tokens=reasoning_tokens,
            prompt_tokens_details=PromptTokensDetailsWrapper(web_search_requests=search_queries)
        )
        
        usage.citation_tokens = citation_tokens
        
        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-deep-research", 
            usage=usage
        )
        
        # Calculate expected costs
        expected_prompt_cost = (100 * 2e-6) + (citation_tokens * 2e-6)
        expected_completion_cost = (50 * 8e-6) + (reasoning_tokens * 3e-6) + (search_queries / 1000 * 0.005)
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-6)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-6)
        
        # Ensure costs are non-negative
        assert prompt_cost >= 0
        assert completion_cost >= 0

    def test_uses_perplexity_provided_cost_when_available(self):
        """
        Test that when Perplexity provides pre-calculated cost in usage.cost.total_cost,
        it is used directly instead of manual calculation.

        This is the fix for issue #15337 - Perplexity returns accurate costs including
        request_cost (fixed per-request fee) that LiteLLM cannot calculate.
        """
        # Create usage with Perplexity's cost object (as returned by the API)
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )

        # Add the cost object that Perplexity returns
        usage.cost = {
            "input_tokens_cost": 0.0,
            "output_tokens_cost": 0.002,
            "request_cost": 0.006,
            "total_cost": 0.008
        }

        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-pro",
            usage=usage
        )

        # When Perplexity provides total_cost, we use it directly
        # prompt_cost should be 0, completion_cost should be total_cost
        assert prompt_cost == 0.0
        assert completion_cost == 0.008
        assert prompt_cost + completion_cost == 0.008

    def test_falls_back_to_manual_calculation_when_no_cost_provided(self):
        """
        Test that manual cost calculation is used when Perplexity doesn't
        provide the cost object (fallback behavior).
        """
        usage = Usage(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150
        )
        # No cost object - should use manual calculation

        prompt_cost, completion_cost = perplexity_cost_per_token(
            model="sonar-deep-research",
            usage=usage
        )

        # Should calculate manually: 100 * 2e-6 + 50 * 8e-6
        expected_prompt = 100 * 2e-6
        expected_completion = 50 * 8e-6

        assert math.isclose(prompt_cost, expected_prompt, rel_tol=1e-6)
        assert math.isclose(completion_cost, expected_completion, rel_tol=1e-6)