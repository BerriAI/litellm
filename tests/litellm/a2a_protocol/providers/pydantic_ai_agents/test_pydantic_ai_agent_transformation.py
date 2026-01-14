"""
Tests for Pydantic AI agents transformation.

Tests the helper functions and response transformation without making real API calls.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.a2a_protocol.providers.pydantic_ai_agents.transformation import (
    PydanticAITransformation,
)


class TestPydanticAITransformation:
    """Tests for PydanticAITransformation helper methods."""

    def test_remove_none_values(self):
        """
        Test that _remove_none_values recursively removes None values from dicts.
        FastA2A servers reject None values for optional fields.
        """
        input_data = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello"}],
                "contextId": None,
                "taskId": None,
                "metadata": None,
            },
            "configuration": None,
            "metadata": {"key": "value", "empty": None},
        }

        result = PydanticAITransformation._remove_none_values(input_data)

        # None values should be removed
        assert "contextId" not in result["message"]
        assert "taskId" not in result["message"]
        assert "metadata" not in result["message"]
        assert "configuration" not in result
        assert "empty" not in result["metadata"]

        # Non-None values should be preserved
        assert result["message"]["role"] == "user"
        assert result["message"]["parts"] == [{"kind": "text", "text": "Hello"}]
        assert result["metadata"]["key"] == "value"

    def test_transform_to_a2a_response(self):
        """
        Test that _transform_to_a2a_response converts Pydantic AI task format
        to standard A2A non-streaming response format.
        """
        # Pydantic AI returns tasks with history/artifacts
        pydantic_ai_response = {
            "jsonrpc": "2.0",
            "id": "req-123",
            "result": {
                "id": "task-456",
                "kind": "task",
                "status": {"state": "completed"},
                "history": [
                    {
                        "role": "user",
                        "parts": [{"kind": "text", "text": "What is 2+2?"}],
                        "messageId": "msg-user-1",
                    },
                    {
                        "role": "agent",
                        "parts": [{"kind": "text", "text": "The answer is 4."}],
                        "messageId": "msg-agent-1",
                    },
                ],
                "artifacts": [
                    {
                        "artifactId": "artifact-1",
                        "name": "response",
                        "parts": [{"kind": "text", "text": "The answer is 4."}],
                    }
                ],
            },
        }

        result = PydanticAITransformation._transform_to_a2a_response(
            response_data=pydantic_ai_response,
            request_id="req-123",
        )

        # Should return standard A2A format with message
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == "req-123"
        assert "message" in result["result"]
        assert result["result"]["message"]["role"] == "agent"
        assert result["result"]["message"]["parts"][0]["text"] == "The answer is 4."

