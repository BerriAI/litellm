"""
Base test class for Anthropic Messages API structured outputs E2E tests.

Tests that structured outputs work correctly via litellm.anthropic.messages interface
by making actual API calls and validating JSON response format.
"""

import json
import os
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath("../../.."))

import pytest
import litellm


class BaseAnthropicMessagesStructuredOutputTest(ABC):
    """
    Base test class for structured outputs E2E tests across different providers.

    Subclasses must implement:
    - get_model(): Returns the model string to use for tests
    """

    @abstractmethod
    def get_model(self) -> str:
        """
        Returns the model string to use for tests.
        """
        pass

    def get_output_format_schema(self) -> Dict[str, Any]:
        """
        Returns a simple JSON schema for testing structured outputs.
        """
        return {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "sentiment": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"]
                    }
                },
                "required": ["sentiment"],
                "additionalProperties": False
            }
        }

    def get_test_messages(self) -> List[Dict[str, Any]]:
        """
        Returns test messages for structured output testing.
        """
        return [
            {
                "role": "user",
                "content": "What is the sentiment of this text: 'This product is amazing!' Return only the sentiment."
            }
        ]

    @pytest.mark.asyncio
    async def test_structured_output_e2e(self):
        """
        E2E test: Make actual API call with structured output and validate JSON response.
        """
        messages = self.get_test_messages()
        output_format = self.get_output_format_schema()

        response = await litellm.anthropic.messages.acreate(
            model=self.get_model(),
            messages=messages,
            max_tokens=100,
            output_format=output_format,
        )

        print(f"Response: {response}")

        # Validate response structure
        assert "content" in response
        assert len(response["content"]) > 0

        content = response["content"][0]
        assert "text" in content

        response_text = content["text"]
        print(f"Response text: {response_text}")

        # The response should be valid JSON
        parsed_json = json.loads(response_text)
        print(f"Parsed JSON: {parsed_json}")

        # Validate the JSON structure
        assert "sentiment" in parsed_json
        assert parsed_json["sentiment"] in ["positive", "negative", "neutral"]