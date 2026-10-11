"""
Base Token Counter Test Suite.

This module provides an abstract base test class that enforces common tests
across all token counter implementations. Similar to base_llm_unit_tests.py
for LLM chat tests.

Usage:
    Create a test class that inherits from BaseTokenCounterTest and implement
    the abstract methods to provide provider-specific configuration.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pytest


from litellm.llms.base_llm.base_utils import BaseTokenCounter


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
            if (
                "api key" in error_str
                or "api_key" in error_str
                or "unauthorized" in error_str
            ):
                pytest.skip(f"Missing or invalid credentials: {e}")
            raise
