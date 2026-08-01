"""
Tests for litellm.llms.gigachat.utils
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath("../../../../../")
)  # Adds the project root to the system path

import pytest
from litellm.llms.gigachat.utils import convert_usage
from litellm.types.utils import PromptTokensDetailsWrapper, Usage


class TestConvertUsage:
    def test_basic_usage_without_precached(self):
        """Test convert_usage with standard tokens, no precached prompt tokens."""
        result = convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        )

        assert result == Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=None,
        )

    def test_usage_with_precached_prompt_tokens(self):
        """Test convert_usage adds precached_prompt_tokens to prompt_tokens and total_tokens."""
        result = convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "precached_prompt_tokens": 3,
                "total_tokens": 15,
            }
        )

        assert result == Usage(
            prompt_tokens=13,
            completion_tokens=5,
            total_tokens=18,
            prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=3),
        )

    def test_zero_precached_prompt_tokens(self):
        """Test convert_usage with zero precached_prompt_tokens does not create details wrapper."""
        result = convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "precached_prompt_tokens": 0,
                "total_tokens": 15,
            }
        )

        assert result == Usage(
            prompt_tokens=10,
            completion_tokens=5,
            total_tokens=15,
            prompt_tokens_details=None,
        )

    def test_missing_optional_fields(self):
        """Test convert_usage with missing optional fields defaults to zero."""
        result = convert_usage(
            {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            }
        )

        assert result.prompt_tokens == 10
        assert result.completion_tokens == 5
        assert result.total_tokens == 15
        assert result.prompt_tokens_details is None
