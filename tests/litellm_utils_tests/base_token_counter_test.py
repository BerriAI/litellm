"""
Base Token Counter Test Suite.

This module provides an abstract base test class that enforces common tests
across all token counter implementations. Similar to base_llm_unit_tests.py
for LLM chat tests.

Usage:
    Create a test class that inherits from BaseTokenCounterTest and implement
    the abstract methods to provide provider-specific configuration.
"""

import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

from litellm.llms.base_llm.base_utils import BaseTokenCounter
from litellm.types.utils import TokenCountResponse


class BaseTokenCounterTest(ABC):
    """
    Abstract base test class for token counter implementations.

    Subclasses must implement:
        - get_token_counter(): Returns the token counter instance
        - get_test_model(): Returns the model name to use for testing
        - get_test_messages(): Returns test messages for token counting
        - get_deployment_config(): Returns deployment configuration with credentials
        - get_custom_llm_provider(): Returns the provider name for should_use_token_counting_api
    """

    @abstractmethod
    def get_token_counter(self) -> BaseTokenCounter:
        """Must return the token counter instance to test."""
        pass

    @abstractmethod
    def get_test_model(self) -> str:
        """Must return the model name to use for testing."""
        pass

    @abstractmethod
    def get_test_messages(self) -> List[Dict[str, Any]]:
        """Must return test messages for token counting."""
        pass

    @abstractmethod
    def get_deployment_config(self) -> Dict[str, Any]:
        """Must return deployment configuration with credentials."""
        pass

    @abstractmethod
    def get_custom_llm_provider(self) -> str:
        """Must return the provider name for should_use_token_counting_api check."""
        pass

    @pytest.fixture(autouse=True)
    def _handle_missing_credentials(self):
        """Fixture to skip tests when credentials are missing."""
        try:
            yield
        except Exception as e:
            error_str = str(e).lower()
            if "api key" in error_str or "api_key" in error_str or "unauthorized" in error_str:
                pytest.skip(f"Missing or invalid credentials: {e}")
            raise

    @pytest.mark.asyncio
    async def test_count_tokens_basic(self):
        """
        Test basic token counting functionality.

        Verifies that:
        - Token counter returns a TokenCountResponse
        - total_tokens is greater than 0
        - tokenizer_type is set
        - No error occurred
        """
        token_counter = self.get_token_counter()
        model = self.get_test_model()
        messages = self.get_test_messages()
        deployment = self.get_deployment_config()

        result = await token_counter.count_tokens(
            model_to_use=model,
            messages=messages,
            contents=None,
            deployment=deployment,
            request_model=model,
        )

        print(f"Token count result: {result}")

        assert result is not None, "Token counter should return a result"
        assert isinstance(result, TokenCountResponse), "Result should be TokenCountResponse"
        assert result.total_tokens > 0, f"Token count should be > 0, got {result.total_tokens}"
        assert result.tokenizer_type is not None, "tokenizer_type should be set"
        assert result.error is not True, f"Token counting should not error: {result.error_message}"

    def test_should_use_token_counting_api(self):
        """
        Test that should_use_token_counting_api returns True for the correct provider.

        Verifies that the token counter correctly identifies when it should be used
        based on the custom_llm_provider.
        """
        token_counter = self.get_token_counter()
        provider = self.get_custom_llm_provider()

        result = token_counter.should_use_token_counting_api(
            custom_llm_provider=provider
        )

        assert result is True, f"should_use_token_counting_api should return True for {provider}"

        # Also verify it returns False for other providers
        other_provider = "some_other_provider_that_doesnt_exist"
        result_other = token_counter.should_use_token_counting_api(
            custom_llm_provider=other_provider
        )

        assert result_other is False, f"should_use_token_counting_api should return False for {other_provider}"
