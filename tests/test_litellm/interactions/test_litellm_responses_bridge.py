"""
Tests for LiteLLM Responses bridge provider.

Inherits from BaseInteractionsTest to run the same test suite against
the litellm_responses bridge provider, which calls litellm.responses() internally.
"""

import os

from tests.test_litellm.interactions.base_interactions_test import (
    BaseInteractionsTest,
)


class TestLiteLLMResponsesBridge(BaseInteractionsTest):
    """Test LiteLLM Responses bridge using the base test suite."""
    
    def get_model(self) -> str:
        """Return the model string for the bridge provider.
        
        The bridge provider uses litellm.responses() internally, so we can
        use any model that litellm.responses() supports (e.g., gpt-4o).
        """
        return "gpt-4o"
    
    def get_api_key(self) -> str:
        """Return the OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY", "")

