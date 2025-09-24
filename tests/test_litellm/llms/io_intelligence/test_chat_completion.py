"""
Test IO Intelligence chat completion functionality with proper mocking
"""
import pytest
from litellm.constants import openai_compatible_providers


class TestIOIntelligenceChatCompletion:
    """Test IO Intelligence chat completion with mocking"""

    def test_io_intelligence_provider_registration(self):
        """Test that IO Intelligence is properly registered as an OpenAI-compatible provider"""
        assert "io_intelligence" in openai_compatible_providers