"""
Tests for OCI streaming responses with tool calls.

This test file specifically addresses the issue where OCI API returns tool calls
without required fields like 'arguments', 'id', or 'name' during streaming,
causing Pydantic validation errors.

Issue: OCI API returns tool calls with incomplete structures during streaming
Error: ValidationError: 1 validation error for OCIStreamChunk message.toolCalls.0.arguments Field required
"""

import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.oci.chat.generic import handle_generic_stream_chunk
from litellm.types.utils import ModelResponseStream


class TestOCIStreamingToolCalls:
    """Test cases for OCI streaming responses with incomplete tool call data."""

    def test_stream_chunk_with_missing_arguments_field(self):
        """
        OCI API can return tool calls in early chunks without the 'arguments' field,
        which should be filled with an empty string to satisfy Pydantic validation.
        """
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "id": "call_abc123",
                        "name": "get_weather",
                        # 'arguments' field is missing
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert len(result.choices) == 1
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == ""

    def test_stream_chunk_with_missing_id_field(self):
        """Missing 'id' gets a generated call_* id."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}',
                        # 'id' field is missing
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0]["id"].startswith("call_")

    def test_stream_chunk_with_missing_name_field(self):
        """Missing 'name' defaults to empty string."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "id": "call_abc123",
                        "arguments": '{"location": "San Francisco"}',
                        # 'name' field is missing
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0]["function"]["name"] == ""

    def test_stream_chunk_with_all_missing_fields(self):
        """All optional fields missing — all default gracefully."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION"
                        # id, name, arguments all missing
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0]["id"].startswith("call_")
        assert result.choices[0].delta.tool_calls[0]["function"]["name"] == ""
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == ""

    def test_stream_chunk_with_complete_tool_call(self):
        """Fully-populated tool call passes through unchanged."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION",
                        "id": "call_abc123",
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco", "unit": "celsius"}',
                    }
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0]["id"] == "call_abc123"
        assert (
            result.choices[0].delta.tool_calls[0]["function"]["name"] == "get_weather"
        )
        assert (
            result.choices[0].delta.tool_calls[0]["function"]["arguments"]
            == '{"location": "San Francisco", "unit": "celsius"}'
        )

    def test_stream_chunk_with_multiple_tool_calls_missing_fields(self):
        """Multiple tool calls with a mix of complete and incomplete entries."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {"type": "FUNCTION", "id": "call_1", "name": "get_weather"},
                    {
                        "type": "FUNCTION",
                        "name": "get_time",
                        "arguments": '{"timezone": "UTC"}',
                    },
                    {
                        "type": "FUNCTION",
                        "id": "call_3",
                        "name": "calculate",
                        "arguments": '{"expression": "2+2"}',
                    },
                ],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 3

        assert result.choices[0].delta.tool_calls[0]["id"] == "call_1"
        assert (
            result.choices[0].delta.tool_calls[0]["function"]["name"] == "get_weather"
        )
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == ""

        assert result.choices[0].delta.tool_calls[1]["id"].startswith("call_")
        assert result.choices[0].delta.tool_calls[1]["function"]["name"] == "get_time"
        assert (
            result.choices[0].delta.tool_calls[1]["function"]["arguments"]
            == '{"timezone": "UTC"}'
        )

        assert result.choices[0].delta.tool_calls[2]["id"] == "call_3"
        assert result.choices[0].delta.tool_calls[2]["function"]["name"] == "calculate"
        assert (
            result.choices[0].delta.tool_calls[2]["function"]["arguments"]
            == '{"expression": "2+2"}'
        )

    def test_stream_chunk_without_tool_calls(self):
        """Plain text chunks (no tool calls) pass through correctly."""
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": [{"type": "TEXT", "text": "Hello, how can I help you?"}],
            },
        }

        result = handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.content == "Hello, how can I help you?"
        assert result.choices[0].delta.tool_calls is None
