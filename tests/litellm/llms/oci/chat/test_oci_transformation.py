"""
Unit tests for OCI chat transformation, specifically for streaming chunks.

Tests cover:
- Tool call ID generation when OCI doesn't return IDs (Gemini models)
- Missing field handling in streaming chunks
- Content item validation fixes
"""

import json
import os
import sys
import uuid
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.oci.chat.transformation import (
    OCIStreamWrapper,
    adapt_messages_to_generic_oci_standard,
    adapt_messages_to_generic_oci_standard_tool_response,
)
from litellm.types.llms.oci import OCIMessage


class TestOCIStreamWrapperToolCalls:
    """Tests for OCI streaming wrapper tool call handling."""

    def _create_stream_wrapper(self):
        """Create a minimal OCIStreamWrapper for testing."""
        return OCIStreamWrapper(
            completion_stream=iter([]),
            model="oracle/google.gemini-2.5-flash",
            custom_llm_provider="oci",
            logging_obj=MagicMock(),
        )

    def test_handle_generic_stream_chunk_generates_uuid_for_missing_tool_call_id(self):
        """
        Test that when OCI Gemini streaming returns tool calls without ID,
        we generate a UUID for each tool call.
        """
        wrapper = self._create_stream_wrapper()

        # Simulate OCI Gemini response with tool call missing 'id' field
        dict_chunk = {
            "index": 0,
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": ""}],
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "name": "get_weather",
                        "arguments": '{"city": "Madrid"}'
                    }
                ]
            },
            "finishReason": "TOOL_CALL"
        }

        # Call the handler
        result = wrapper._handle_generic_stream_chunk(dict_chunk)

        # Verify a UUID was generated for the tool call
        assert "id" in dict_chunk["message"]["toolCalls"][0]
        generated_id = dict_chunk["message"]["toolCalls"][0]["id"]
        assert len(generated_id) == 32  # UUID hex is 32 chars
        # Verify it's a valid hex string (UUID format)
        int(generated_id, 16)  # Will raise if not valid hex

    def test_handle_generic_stream_chunk_preserves_existing_tool_call_id(self):
        """
        Test that when OCI returns a tool call with an ID, we preserve it.
        """
        wrapper = self._create_stream_wrapper()

        existing_id = "existing_tool_call_id_123"
        dict_chunk = {
            "index": 0,
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": ""}],
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "id": existing_id,
                        "name": "get_weather",
                        "arguments": '{"city": "Madrid"}'
                    }
                ]
            },
            "finishReason": "TOOL_CALL"
        }

        result = wrapper._handle_generic_stream_chunk(dict_chunk)

        # Verify the existing ID was preserved
        assert dict_chunk["message"]["toolCalls"][0]["id"] == existing_id

    def test_handle_generic_stream_chunk_generates_unique_ids_for_multiple_tool_calls(self):
        """
        Test that multiple tool calls each get unique UUIDs.
        """
        wrapper = self._create_stream_wrapper()

        dict_chunk = {
            "index": 0,
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": ""}],
                "toolCalls": [
                    {"type": "FUNCTION", "name": "get_weather", "arguments": '{"city": "Madrid"}'},
                    {"type": "FUNCTION", "name": "get_weather", "arguments": '{"city": "Barcelona"}'},
                    {"type": "FUNCTION", "name": "get_time", "arguments": '{}'}
                ]
            },
            "finishReason": "TOOL_CALL"
        }

        result = wrapper._handle_generic_stream_chunk(dict_chunk)

        # Verify each tool call has a unique ID
        ids = [tc["id"] for tc in dict_chunk["message"]["toolCalls"]]
        assert len(ids) == 3
        assert len(set(ids)) == 3  # All unique

    def test_handle_generic_stream_chunk_adds_missing_text_field(self):
        """
        Test that when content has type TEXT but missing text field,
        we add an empty text field.
        """
        wrapper = self._create_stream_wrapper()

        # OCI sometimes returns content with type but no text
        dict_chunk = {
            "index": 0,
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT"}],  # Missing 'text' field
                "toolCalls": None
            },
            "finishReason": None
        }

        result = wrapper._handle_generic_stream_chunk(dict_chunk)

        # Verify text field was added
        assert "text" in dict_chunk["message"]["content"][0]
        assert dict_chunk["message"]["content"][0]["text"] == ""

    def test_handle_generic_stream_chunk_adds_missing_arguments_field(self):
        """
        Test that when tool call is missing arguments field,
        we add an empty arguments string.
        """
        wrapper = self._create_stream_wrapper()

        dict_chunk = {
            "index": 0,
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": ""}],
                "toolCalls": [
                    {"type": "FUNCTION", "name": "get_info"}  # Missing 'arguments'
                ]
            },
            "finishReason": "TOOL_CALL"
        }

        result = wrapper._handle_generic_stream_chunk(dict_chunk)

        # Verify arguments field was added
        assert "arguments" in dict_chunk["message"]["toolCalls"][0]
        assert dict_chunk["message"]["toolCalls"][0]["arguments"] == ""

    def test_handle_generic_stream_chunk_adds_missing_name_field(self):
        """
        Test that when tool call is missing name field,
        we add an empty name string.
        """
        wrapper = self._create_stream_wrapper()

        dict_chunk = {
            "index": 0,
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": ""}],
                "toolCalls": [
                    {"type": "FUNCTION", "arguments": "{}"}  # Missing 'name'
                ]
            },
            "finishReason": "TOOL_CALL"
        }

        result = wrapper._handle_generic_stream_chunk(dict_chunk)

        # Verify name field was added
        assert "name" in dict_chunk["message"]["toolCalls"][0]
        assert dict_chunk["message"]["toolCalls"][0]["name"] == ""


class TestOCIToolResponseTransformation:
    """Tests for transforming tool responses to OCI format."""

    def test_adapt_tool_response_includes_tool_call_id(self):
        """
        Test that tool responses include the tool_call_id.
        """
        tool_call_id = "abc123"
        content = "The weather in Madrid is 25C"

        result = adapt_messages_to_generic_oci_standard_tool_response(
            role="tool",
            tool_call_id=tool_call_id,
            content=content
        )

        assert isinstance(result, OCIMessage)
        assert result.role == "TOOL"
        assert result.toolCallId == tool_call_id
        assert result.content[0].text == content

    def test_adapt_tool_response_with_uuid_format_id(self):
        """
        Test that tool responses work with UUID-format IDs.
        """
        # UUID format like we generate
        tool_call_id = uuid.uuid4().hex
        content = "Result from tool"

        result = adapt_messages_to_generic_oci_standard_tool_response(
            role="tool",
            tool_call_id=tool_call_id,
            content=content
        )

        assert result.toolCallId == tool_call_id


class TestOCIMessageTransformation:
    """Tests for transforming messages to OCI format."""

    def test_adapt_messages_with_tool_response(self):
        """
        Test that a conversation with tool response is properly transformed.
        """
        messages = [
            {"role": "user", "content": "What's the weather in Madrid?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Madrid"}'
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_123",
                "content": "25C and sunny"
            }
        ]

        result = adapt_messages_to_generic_oci_standard(messages)

        assert len(result) == 3
        # Check user message
        assert result[0].role == "USER"
        # Check assistant message with tool call
        assert result[1].role == "ASSISTANT"
        assert result[1].toolCalls is not None
        assert len(result[1].toolCalls) == 1
        # Check tool response
        assert result[2].role == "TOOL"
        assert result[2].toolCallId == "call_123"
