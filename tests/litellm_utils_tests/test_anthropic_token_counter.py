"""
Anthropic Token Counter Tests.

Tests for the Anthropic token counter implementation using the base test suite.
"""

import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.anthropic.count_tokens import AnthropicTokenCounter
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from tests.litellm_utils_tests.base_token_counter_test import BaseTokenCounterTest


class TestAnthropicTokenCounter(BaseTokenCounterTest):
    """Test suite for Anthropic token counter."""

    def get_token_counter(self) -> BaseTokenCounter:
        return AnthropicTokenCounter()

    def get_test_model(self) -> str:
        return "claude-sonnet-4-20250514"

    def get_test_messages(self) -> List[Dict[str, Any]]:
        return [
            {"role": "user", "content": "Hello, how are you today?"}
        ]

    def get_deployment_config(self) -> Dict[str, Any]:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            pytest.skip("ANTHROPIC_API_KEY not set")
        return {
            "litellm_params": {
                "api_key": api_key,
            }
        }

    def get_custom_llm_provider(self) -> str:
        return "anthropic"
