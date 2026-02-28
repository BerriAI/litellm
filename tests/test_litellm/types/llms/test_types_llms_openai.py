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


class TestAssistantMessageImageUrlContent:
    """
    Regression tests for image_url blocks in assistant message content.

    Bug: ChatCompletionAssistantMessage.content did not include
    ChatCompletionImageObject in its union, so Pydantic v2 silently dropped
    image_url blocks (content → []) when serialising via AllMessageValues.
    This affects users who store conversation history as JSON (e.g. in a DB)
    and read it back typed as list[AllMessageValues].
    """

    ASSISTANT_MESSAGE_WITH_IMAGE = {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Here is the image you requested:"},
            {
                "type": "image_url",
                "image_url": {
                    "url": (
                        "data:image/png;base64,"
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAA"
                        "DUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    )
                },
            },
        ],
    }

    def test_assistant_message_image_url_preserved_single(self):
        """
        TypeAdapter(ChatCompletionAssistantMessage): image_url block must survive
        validate_python → dump_python without being dropped or raising an error.
        """
        from typing import List

        from pydantic import TypeAdapter

        from litellm.types.llms.openai import ChatCompletionAssistantMessage

        adapter = TypeAdapter(ChatCompletionAssistantMessage)
        validated = adapter.validate_python(self.ASSISTANT_MESSAGE_WITH_IMAGE)
        dumped = adapter.dump_python(validated)

        raw_content = dumped.get("content")
        # Pydantic may return a lazy SerializationIterator for Iterable fields;
        # convert to list to consume it — this must not raise ValidationError.
        content_blocks = list(raw_content) if raw_content is not None else []

        assert len(content_blocks) == 2, (
            f"Expected 2 content blocks (text + image_url), got {len(content_blocks)}: {content_blocks}"
        )
        types = [b.get("type") for b in content_blocks if isinstance(b, dict)]
        assert "image_url" in types, f"image_url block was silently dropped; blocks: {content_blocks}"

    def test_assistant_message_image_url_preserved_in_all_message_values(self):
        """
        TypeAdapter(List[AllMessageValues]) DB round-trip: image_url blocks in an
        assistant message must not be silently dropped during dump_python(mode='json').

        This is the primary failing path: conversation history stored as JSON in a
        database and read back typed as list[AllMessageValues].
        """
        from typing import List

        from pydantic import TypeAdapter

        from litellm.types.llms.openai import AllMessageValues

        conversation = [
            {
                "role": "user",
                "content": "Generate an image of a banana wearing a LiteLLM costume",
            },
            self.ASSISTANT_MESSAGE_WITH_IMAGE,
        ]

        adapter = TypeAdapter(List[AllMessageValues])
        validated = adapter.validate_python(conversation)
        dumped = adapter.dump_python(validated, mode="json")

        assistant = next((m for m in dumped if m.get("role") == "assistant"), None)
        assert assistant is not None, "Assistant message missing after serialisation"

        content = assistant.get("content", [])
        assert isinstance(content, list), f"content should be a list, got {type(content)}"
        assert len(content) == 2, (
            f"Expected 2 content blocks (text + image_url), got {len(content)}: {content}"
        )
        types = [b.get("type") for b in content if isinstance(b, dict)]
        assert "image_url" in types, (
            f"image_url block was silently dropped during AllMessageValues serialisation; blocks: {content}"
        )


class TestResponsesAPIReasoningNullFields:
    """
    Tests for issue #16824: reasoning output items should not include null
    status/content/encrypted_content fields.

    When a provider returns reasoning items without these fields, LiteLLM's
    Pydantic parsing adds them as Optional defaults (None). Serializing them
    as null breaks downstream SDKs (e.g., the OpenAI C# SDK crashes on
    status=null).

    The fix uses a field_serializer on ResponsesAPIResponse.output that
    mirrors the request-side filtering in
    OpenAIResponsesAPIConfig._handle_reasoning_item().
    """

    def _make_response(self, output):
        from litellm.types.llms.openai import ResponsesAPIResponse

        return ResponsesAPIResponse(
            id="resp_test",
            created_at=1741476542,
            model="gpt-5-mini",
            object="response",
            status="completed",
            output=output,
        )

    def test_reasoning_item_null_fields_removed_model_dump(self):
        """Null status/content/encrypted_content should be absent from model_dump."""
        response = self._make_response(
            output=[{"id": "rs_abc", "type": "reasoning", "summary": []}]
        )
        dumped = response.model_dump()
        reasoning = dumped["output"][0]
        assert "status" not in reasoning
        assert "content" not in reasoning
        assert "encrypted_content" not in reasoning

    def test_reasoning_item_null_fields_removed_model_dump_json(self):
        """Null fields should also be absent from model_dump_json."""
        response = self._make_response(
            output=[{"id": "rs_abc", "type": "reasoning", "summary": []}]
        )
        parsed = json.loads(response.model_dump_json())
        reasoning = parsed["output"][0]
        assert "status" not in reasoning
        assert "content" not in reasoning
        assert "encrypted_content" not in reasoning

    def test_reasoning_item_non_null_values_preserved(self):
        """Non-null values on reasoning items should be kept."""
        response = self._make_response(
            output=[
                {
                    "id": "rs_abc",
                    "type": "reasoning",
                    "summary": [],
                    "status": "completed",
                    "encrypted_content": "gAAAA...",
                }
            ]
        )
        dumped = response.model_dump()
        reasoning = dumped["output"][0]
        assert reasoning["status"] == "completed"
        assert reasoning["encrypted_content"] == "gAAAA..."

    def test_message_item_not_affected(self):
        """Non-reasoning output items should keep all their fields."""
        response = self._make_response(
            output=[
                {
                    "id": "msg_abc",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Hello!",
                            "annotations": [],
                        }
                    ],
                }
            ]
        )
        dumped = response.model_dump()
        message = dumped["output"][0]
        assert message["status"] == "completed"
        assert message["type"] == "message"
        assert len(message["content"]) == 1

    def test_mixed_output_reasoning_and_message(self):
        """Reasoning items cleaned, message items untouched in same response."""
        response = self._make_response(
            output=[
                {"id": "rs_abc", "type": "reasoning", "summary": []},
                {
                    "id": "msg_abc",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Answer",
                            "annotations": [],
                        }
                    ],
                },
            ]
        )
        dumped = response.model_dump()
        reasoning = [
            o for o in dumped["output"] if isinstance(o, dict) and o.get("type") == "reasoning"
        ][0]
        message = [
            o for o in dumped["output"] if isinstance(o, dict) and o.get("type") == "message"
        ][0]
        assert "status" not in reasoning
        assert "content" not in reasoning
        assert message["status"] == "completed"
        assert len(message["content"]) == 1

    def test_reasoning_core_fields_preserved(self):
        """id, type, summary should always be present on reasoning items."""
        response = self._make_response(
            output=[{"id": "rs_abc", "type": "reasoning", "summary": ["thinking..."]}]
        )
        dumped = response.model_dump()
        reasoning = dumped["output"][0]
        assert reasoning["id"] == "rs_abc"
        assert reasoning["type"] == "reasoning"
        assert reasoning["summary"] == ["thinking..."]

    def test_top_level_null_fields_unaffected(self):
        """Top-level response fields with None should not be affected."""
        response = self._make_response(
            output=[{"id": "rs_abc", "type": "reasoning", "summary": []}]
        )
        dumped = response.model_dump()
        assert "error" in dumped
        assert dumped["error"] is None
        assert "instructions" in dumped
        assert dumped["instructions"] is None
