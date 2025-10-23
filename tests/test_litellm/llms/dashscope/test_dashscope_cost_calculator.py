"""
Test suite for Dashscope cost calculation functionality.

Tests the cost calculation for Dashscope models including:
- Tiered pricing based on input token ranges
- Caching discounts
- Reasoning tokens
- Standard flat pricing fallback
"""

import json
import math
import os
import sys

import pytest

# Add the project root to Python path
sys.path.insert(0, os.path.abspath("../../../.."))

import litellm
from litellm.llms.dashscope.cost_calculator import (
    cost_per_token as dashscope_cost_per_token,
)
from litellm.types.utils import (
    CompletionTokensDetailsWrapper,
    PromptTokensDetailsWrapper,
    Usage,
)


class TestDashscopeCostCalculator:
    """Test suite for Dashscope cost calculation functionality."""

    @pytest.fixture(autouse=True)
    def setup_model_cost_map(self):
        """Set up the model cost map for testing."""
        # Ensure we use local model cost map for consistent testing
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        
        # Find the project root directory and load model cost map
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = current_dir
        while not os.path.exists(os.path.join(project_root, "model_prices_and_context_window.json")):
            parent = os.path.dirname(project_root)
            if parent == project_root:  # Reached filesystem root
                break
            project_root = parent
        
        model_cost_path = os.path.join(project_root, "model_prices_and_context_window.json")
        with open(model_cost_path, "r") as f:
            model_cost_map = json.load(f)
        litellm.model_cost = model_cost_map

    def test_flat_pricing_basic_cost_calculation(self):
        """Test basic cost calculation for flat pricing models (qwen-max)."""
        usage = Usage(
            prompt_tokens=1000,
            completion_tokens=500,
            total_tokens=1500
        )
        
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen-max", 
            usage=usage
        )
        
        # Expected costs for qwen-max:
        # Input: 1000 tokens * $1.6e-6 = $0.0016
        # Output: 500 tokens * $6.4e-6 = $0.0032
        expected_prompt_cost = 1000 * 1.6e-6
        expected_completion_cost = 500 * 6.4e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_single_tier(self):
        """Test tiered pricing when all tokens fall within first tier."""
        usage = Usage(
            prompt_tokens=20000,  # Within first tier (0-32K)
            completion_tokens=1000,
            total_tokens=21000
        )
        
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen3-coder-plus", 
            usage=usage
        )
        
        # Expected costs for qwen3-coder-plus (tier 1):
        # Input: 20,000 tokens * $1e-6 = $0.02
        # Output: 1,000 tokens * $5e-6 = $0.005
        expected_prompt_cost = 20000 * 1e-6
        expected_completion_cost = 1000 * 5e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_higher_tier(self):
        """Test tiered pricing when tokens fall in higher tier (tier 3)."""
        usage = Usage(
            prompt_tokens=150000,  # Falls in tier 3 (128K-256K)
            completion_tokens=2000,
            total_tokens=152000
        )
        
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen3-coder-plus", 
            usage=usage
        )
        
        # Expected input cost calculation:
        # 150,000 tokens falls in tier 3 (128K-256K), so all tokens are charged at tier 3 rate
        # Input: 150,000 tokens * $3e-6 = $0.45
        # Output: 2,000 tokens falls in tier 1 (0-32K), so charged at tier 1 rate
        # Output: 2,000 tokens * $5e-6 = $0.01
        
        expected_prompt_cost = 150000 * 3e-6  # All tokens at tier 3 rate
        expected_completion_cost = 2000 * 5e-6  # All tokens at tier 1 rate
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_with_caching(self):
        """Test tiered pricing with cached tokens."""
        prompt_tokens_details = PromptTokensDetailsWrapper(
            cached_tokens=10000  # 10K cached tokens
        )
        
        usage = Usage(
            prompt_tokens=50000,  # 40K regular + 10K cached = 50K total
            completion_tokens=1000,
            total_tokens=51000,
            prompt_tokens_details=prompt_tokens_details
        )
        
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen3-coder-plus", 
            usage=usage
        )
        
        # Expected cost calculation:
        # Regular tokens: 40,000 falls in tier 2 (32K-128K), so all charged at tier 2 rate
        # - Regular: 40,000 * $1.8e-6 = $0.072
        # Cached tokens: 10,000 falls in tier 1 (0-32K), so charged at tier 1 cached rate
        # - Cached: 10,000 * $1e-7 = $0.001
        # Total input cost = $0.072 + $0.001 = $0.073
        
        regular_tokens = 40000
        cached_tokens = 10000
        
        expected_regular_cost = regular_tokens * 1.8e-6  # Tier 2 rate
        expected_cached_cost = cached_tokens * 1e-7  # Tier 1 cached rate
        expected_prompt_cost = expected_regular_cost + expected_cached_cost
        expected_completion_cost = 1000 * 5e-6  # Tier 1 rate
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)

    def test_tiered_pricing_highest_tier(self):
        """Test tiered pricing when tokens exceed highest tier range."""
        usage = Usage(
            prompt_tokens=2000000,  # Exceeds tier 4 max (1M), should use tier 4 rate
            completion_tokens=5000,
            total_tokens=2005000
        )
        
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen3-coder-plus", 
            usage=usage
        )
        
        # Expected cost calculation:
        # 2,000,000 tokens exceeds tier 4 (256K-1M), so use tier 4 rate for all tokens
        # Input: 2,000,000 tokens * $6e-6 = $12.0
        # Output: 5,000 tokens falls in tier 1 (0-32K), so charged at tier 1 rate
        # Output: 5,000 tokens * $5e-6 = $0.025
        
        expected_prompt_cost = 2000000 * 6e-6  # Tier 4 rate (highest tier)
        expected_completion_cost = 5000 * 5e-6  # Tier 1 rate
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)