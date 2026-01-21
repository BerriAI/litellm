"""
Azure AI Anthropic Token Counter Tests.

Tests for the Azure AI Anthropic token counter implementation using the base test suite.
"""

import os
import sys
from typing import Any, Dict, List

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.azure_ai.anthropic.count_tokens import AzureAIAnthropicTokenCounter
from litellm.llms.base_llm.base_utils import BaseTokenCounter
from tests.litellm_utils_tests.base_token_counter_test import BaseTokenCounterTest


class TestAzureAIAnthropicTokenCounter(BaseTokenCounterTest):
    """Test suite for Azure AI Anthropic token counter."""

    def get_token_counter(self) -> BaseTokenCounter:
        return AzureAIAnthropicTokenCounter()

    def get_test_model(self) -> str:
        return "claude-3-5-sonnet"

    def get_test_messages(self) -> List[Dict[str, Any]]:
        return [
            {"role": "user", "content": "Hello, how are you today?"}
        ]

    def get_deployment_config(self) -> Dict[str, Any]:
        api_key = os.getenv("AZURE_AI_API_KEY")
        api_base = os.getenv("AZURE_AI_API_BASE")
        
        if not api_key:
            pytest.skip("AZURE_AI_API_KEY not set")
        if not api_base:
            pytest.skip("AZURE_AI_API_BASE not set")
            
        return {
            "litellm_params": {
                "api_key": api_key,
                "api_base": api_base,
            }
        }

    def get_custom_llm_provider(self) -> str:
        return "azure_ai"
