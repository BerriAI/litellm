# What is this?
## Unit tests for router priority normalization functionality

import sys, os
import pytest
from typing import Any, Optional

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from litellm import Router


class TestRouterPriorityNormalization:
    """Test priority normalization in Router"""

    def setup_method(self):
        """Setup router instance for testing"""
        self.router = Router(
            model_list=[
                {
                    "model_name": "gpt-3.5-turbo",
                    "litellm_params": {
                        "model": "gpt-3.5-turbo",
                        "api_key": "fake_key"
                    }
                }
            ],
            default_priority=100  # Set a default priority for testing
        )

    @pytest.mark.parametrize(
        "input_priority, default_priority, expected_output",
        [
            # Valid integer priorities
            (5, None, 5),
            (0, None, 0),
            (255, None, 255),
            (300, None, 255),  # Should clamp to max
            (-10, None, 0),    # Should clamp to min

            # Float priorities
            (5.7, None, 5),
            (10.2, None, 10),
            (255.9, None, 255),

            # String priorities
            ("5", None, 5),
            ("10.5", None, 10),
            ("", 50, 50),      # Empty string should use default
            ("  7  ", None, 7), # String with whitespace
            ("abc", 50, 50),    # Invalid string should use default

            # List/tuple priorities (should use first valid element)
            ([5, 10], None, 5),
            ([5.5, 10], None, 5),
            (["7", 10], None, 7),
            ([None, 8], None, 8),
            ([], 50, 50),       # Empty list should use default
            (["abc", "def"], 50, 50),  # Invalid items should use default

            # Tuple priorities
            ((3, 7), None, 3),
            ((3.5,), None, 3),
            (("9",), None, 9),
            ((), 50, 50),       # Empty tuple should use default

            # None and other invalid types
            (None, 75, 75),
            ({}, 75, 75),
            (set(), 75, 75),
        ]
    )
    def test_normalize_priority(self, input_priority: Any, default_priority: Optional[int], expected_output: Optional[int]):
        """Test priority normalization with various input types"""
        result = self.router._normalize_priority(input_priority, default_priority)
        assert result == expected_output, f"Failed for input {input_priority}, default {default_priority}: expected {expected_output}, got {result}"

    def test_normalize_priority_range_validation(self):
        """Test that priority normalization enforces valid range (0-255)"""
        # Test values outside valid range
        test_cases = [
            (-100, 0),   # Negative should clamp to 0
            (1000, 255), # Too high should clamp to 255
            (-1, 0),     # Slightly negative should clamp to 0
            (256, 255),  # Just over max should clamp to 255
        ]

        for input_val, expected in test_cases:
            result = self.router._normalize_priority(input_val)
            assert result == expected, f"Range validation failed for {input_val}: expected {expected}, got {result}"

    def test_normalize_priority_type_preservation(self):
        """Test that normalized priorities are always integers"""
        test_inputs = [5, 5.7, "5", [5.9], (3.2,)]

        for input_val in test_inputs:
            result = self.router._normalize_priority(input_val)
            if result is not None:
                assert isinstance(result, int), f"Result should be int, got {type(result)} for input {input_val}"

    def test_normalize_priority_with_corrupted_redis_data(self):
        """Test normalization with data types that might come from corrupted Redis cache"""
        # These are examples of corrupted data that might come from ast.literal_eval
        corrupted_data = [
            ([1, 2, 3], 1),       # List with multiple elements
            (("nested", (5,)), 50),   # Nested tuple (should use default)
            ([{"key": 5}], 50),   # List with dict (should use default)
            ([[1, 2]], 50),      # Nested list (should use default)
            ([None, None, 7], 7), # List with None values
        ]

        for input_val, expected in corrupted_data:
            result = self.router._normalize_priority(input_val, 50)
            assert result == expected, f"Corrupted data handling failed for {input_val}: expected {expected}, got {result}"

    def test_normalize_priority_edge_cases(self):
        """Test edge cases for priority normalization"""
        # Test with very large numbers
        result = self.router._normalize_priority(10**10)
        assert result == 255

        # Test with very small float
        result = self.router._normalize_priority(0.1)
        assert result == 0

        # Test with string containing float
        result = self.router._normalize_priority("5.99")
        assert result == 5

        # Test with mixed type list
        result = self.router._normalize_priority([None, "abc", 3.7, "def"])
        assert result == 3


if __name__ == "__main__":
    pytest.main([__file__])