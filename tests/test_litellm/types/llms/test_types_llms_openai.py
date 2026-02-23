import asyncio
import os
import sys
from typing import Optional
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))
import json

import litellm


def test_generic_event():
    from litellm.types.llms.openai import GenericEvent

    event = {"type": "test", "test": "test"}
    event = GenericEvent(**event)
    assert event.type == "test"
    assert event.test == "test"


def test_output_item_added_event():
    from litellm.types.llms.openai import OutputItemAddedEvent

    event = {
        "type": "response.output_item.added",
        "sequence_number": 4,
        "output_index": 1,
        "item": None,
    }
    event = OutputItemAddedEvent(**event)
    assert event.type == "response.output_item.added"
    assert event.sequence_number == 4
    assert event.output_index == 1
    assert event.item is None


class TestResponsesAPIResponseOutputText:
    """Tests for the output_text property on ResponsesAPIResponse"""

    def test_output_text_with_single_message(self):
        """Test output_text with a single message containing text output"""
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "id": "msg_123",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello, world!",
                        }
                    ],
                }
            ],
        )

        assert response.output_text == "Hello, world!"

    def test_output_text_with_multiple_messages(self):
        """Test output_text with multiple messages aggregates all text"""
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "First part. ",
                        }
                    ],
                },
                {
                    "type": "message",
                    "id": "msg_2",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Second part.",
                        }
                    ],
                },
            ],
        )

        assert response.output_text == "First part. Second part."

    def test_output_text_with_no_text_content(self):
        """Test output_text returns empty string when no output_text content exists"""
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            output=[
                {
                    "type": "function_call",
                    "id": "call_123",
                    "status": "completed",
                    "name": "get_weather",
                    "arguments": "{}",
                }
            ],
        )

        assert response.output_text == ""

    def test_output_text_with_mixed_content(self):
        """Test output_text only aggregates output_text type content"""
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            output=[
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "The weather is sunny. ",
                        },
                        {
                            "type": "refusal",
                            "refusal": "I cannot do that.",
                        },
                    ],
                },
                {
                    "type": "function_call",
                    "id": "call_123",
                    "status": "completed",
                    "name": "get_weather",
                    "arguments": "{}",
                },
            ],
        )

        assert response.output_text == "The weather is sunny. "

    def test_output_text_with_empty_output(self):
        """Test output_text returns empty string with empty output list"""
        from litellm.types.llms.openai import ResponsesAPIResponse

        response = ResponsesAPIResponse(
            id="resp_123",
            created_at=1234567890,
            output=[],
        )

        assert response.output_text == ""
