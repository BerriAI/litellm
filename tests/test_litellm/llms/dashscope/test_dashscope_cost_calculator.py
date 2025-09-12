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

    def test_tiered_pricing_multiple_tiers(self):
        """Test tiered pricing when tokens span multiple tiers."""
        usage = Usage(
            prompt_tokens=150000,  # Spans tiers 1 (0-32K), 2 (32K-128K), 3 (128K-256K)
            completion_tokens=2000,
            total_tokens=152000
        )
        
        prompt_cost, completion_cost = dashscope_cost_per_token(
            model="qwen3-coder-plus", 
            usage=usage
        )
        
        # Expected input cost calculation:
        # Tier 1 (0-32K): 32,000 tokens * $1e-6 = $0.032
        # Tier 2 (32K-128K): 96,000 tokens * $1.8e-6 = $0.1728
        # Tier 3 (128K-256K): 22,000 tokens * $3e-6 = $0.066
        # Total input cost = $0.032 + $0.1728 + $0.066 = $0.2708
        
        expected_prompt_cost = (32000 * 1e-6) + (96000 * 1.8e-6) + (22000 * 3e-6)
        expected_completion_cost = 2000 * 5e-6  # All in tier 1 for output
        
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
        # Regular tokens: 40,000 (32K in tier 1 + 8K in tier 2)
        # - Tier 1: 32,000 * $1e-6 = $0.032
        # - Tier 2: 8,000 * $1.8e-6 = $0.0144
        # Cached tokens: 10,000 in tier 1 at discounted rate
        # - Tier 1 cached: 10,000 * $1e-7 = $0.001
        # Total input cost = $0.032 + $0.0144 + $0.001 = $0.0474
        
        regular_tokens = 40000
        cached_tokens = 10000
        
        expected_regular_cost = (32000 * 1e-6) + (8000 * 1.8e-6)
        expected_cached_cost = cached_tokens * 1e-7  # Tier 1 cached rate
        expected_prompt_cost = expected_regular_cost + expected_cached_cost
        expected_completion_cost = 1000 * 5e-6
        
        assert math.isclose(prompt_cost, expected_prompt_cost, rel_tol=1e-10)
        assert math.isclose(completion_cost, expected_completion_cost, rel_tol=1e-10)