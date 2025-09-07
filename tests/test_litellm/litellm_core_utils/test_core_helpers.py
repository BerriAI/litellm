import json
import os
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs, safe_divide


def test_get_litellm_metadata_from_kwargs():
    kwargs = {
        "litellm_params": {
            "litellm_metadata": {},
            "metadata": {"user_api_key": "1234567890"},
        },
    }
    assert get_litellm_metadata_from_kwargs(kwargs) == {"user_api_key": "1234567890"}


def test_add_missing_spend_metadata_to_litellm_metadata():
    litellm_metadata = {"test_key": "test_value"}
    metadata = {"user_api_key_hash_value": "1234567890"}
    kwargs = {
        "litellm_params": {
            "litellm_metadata": litellm_metadata,
            "metadata": metadata,
        },
    }
    assert get_litellm_metadata_from_kwargs(kwargs) == {
        "test_key": "test_value",
        "user_api_key_hash_value": "1234567890",
    }


def test_preserve_upstream_non_openai_attributes():
    from litellm.litellm_core_utils.core_helpers import (
        preserve_upstream_non_openai_attributes,
    )
    from litellm.types.utils import ModelResponseStream

    model_response = ModelResponseStream(
        id="123",
        object="text_completion",
        created=1715811200,
        model="gpt-3.5-turbo",
    )

    setattr(model_response, "test_key", "test_value")
    preserve_upstream_non_openai_attributes(
        model_response=ModelResponseStream(),
        original_chunk=model_response,
    )

    assert model_response.test_key == "test_value"


def test_safe_divide_basic():
    """Test basic safe division functionality"""
    # Normal division
    result = safe_divide(10, 2)
    assert result == 5.0, f"Expected 5.0, got {result}"
    
    # Division with float
    result = safe_divide(7.5, 2.5)
    assert result == 3.0, f"Expected 3.0, got {result}"
    
    # Division by zero with default
    result = safe_divide(10, 0)
    assert result == 0, f"Expected 0, got {result}"
    
    # Division by zero with custom default
    result = safe_divide(10, 0, default=1)
    assert result == 1, f"Expected 1, got {result}"
    
    # Division by zero with custom default as float
    result = safe_divide(10, 0, default=0.5)
    assert result == 0.5, f"Expected 0.5, got {result}"


def test_safe_divide_edge_cases():
    """Test edge cases for safe division"""
    # Zero numerator
    result = safe_divide(0, 5)
    assert result == 0.0, f"Expected 0.0, got {result}"
    
    # Negative numbers
    result = safe_divide(-10, 2)
    assert result == -5.0, f"Expected -5.0, got {result}"
    
    # Negative denominator
    result = safe_divide(10, -2)
    assert result == -5.0, f"Expected -5.0, got {result}"
    
    # Both negative
    result = safe_divide(-10, -2)
    assert result == 5.0, f"Expected 5.0, got {result}"
    
    # Float division
    result = safe_divide(1, 3)
    assert abs(result - 0.3333333333333333) < 1e-10, f"Expected ~0.333..., got {result}"


def test_safe_divide_weight_scenario():
    """Test safe division in the context of weight calculations"""
    # Simulate weight calculation scenario
    weights = [3, 7, 0, 2]
    total_weight = sum(weights)  # 12
    
    # Normal case
    normalized_weights = [safe_divide(w, total_weight) for w in weights]
    expected = [0.25, 7/12, 0.0, 1/6]
    
    for i, (actual, exp) in enumerate(zip(normalized_weights, expected)):
        assert abs(actual - exp) < 1e-10, f"Weight {i}: Expected {exp}, got {actual}"
    
    # Zero total weight scenario (division by zero)
    zero_weights = [0, 0, 0]
    zero_total = sum(zero_weights)  # 0
    
    # Should return default values (0) for all weights
    normalized_zero_weights = [safe_divide(w, zero_total) for w in zero_weights]
    expected_zero = [0, 0, 0]
    
    assert normalized_zero_weights == expected_zero, f"Expected {expected_zero}, got {normalized_zero_weights}"
