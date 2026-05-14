"""
Tests for LiteLLM Responses bridge provider.

Inherits from BaseInteractionsTest to run the same test suite against
the litellm_responses bridge provider, which calls litellm.responses() internally.
"""

import os
from unittest.mock import patch

import pytest

from litellm.types.interactions.generated import InteractionsAPIResponse
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

    @pytest.mark.asyncio
    async def test_acreate_simple(self):
        """Test async interaction creation with mocked API call."""
        mock_response = InteractionsAPIResponse(
            id="interaction-abc123",
            status="completed",
            model="gpt-4o",
            outputs=[{"type": "text", "text": "The speed of light is approximately 299,792,458 meters per second."}],
            usage={"input_tokens": 10, "output_tokens": 20},
        )

        import litellm.interactions as interactions

        with patch("litellm.interactions.main.create", return_value=mock_response):
            response = await interactions.acreate(
                model=self.get_model(),
                input="What is the speed of light?",
                api_key="sk-fake-key-for-unit-test",
            )
        assert response is not None
        assert response.id is not None or response.status is not None
