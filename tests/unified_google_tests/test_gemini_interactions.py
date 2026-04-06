"""
Tests for Gemini Interactions API.

Inherits from BaseInteractionsTest to run the same test suite against Gemini.
"""

import os

from tests.unified_google_tests.base_interactions_test import (
    BaseInteractionsTest,
)


class TestGeminiInteractions(BaseInteractionsTest):
    """Test Gemini Interactions API using the base test suite."""
    
    def get_model(self) -> str:
        """Return the Gemini model string."""
        return "gemini/gemini-2.5-flash"
    
    def get_api_key(self) -> str:
        """Return the Gemini API key from environment."""
        return os.getenv("GEMINI_API_KEY", "")

