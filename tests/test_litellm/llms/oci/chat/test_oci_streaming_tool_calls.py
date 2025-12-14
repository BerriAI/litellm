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
import pytest
from unittest.mock import MagicMock

# Adds the parent directory to the system path
sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.oci.chat.transformation import OCIStreamWrapper
from litellm.types.utils import ModelResponseStream


class TestOCIStreamingToolCalls:
    """Test cases for OCI streaming responses with incomplete tool call data."""

    def test_stream_chunk_with_missing_arguments_field(self):
        """
        Test that streaming chunks with tool calls missing 'arguments' field are handled.

        OCI API can return tool calls in early chunks without the 'arguments' field,
        which should be filled with an empty string to satisfy Pydantic validation.
        """
        # Mock streaming chunk with tool call missing 'arguments' field
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
                        "name": "get_weather"
                        # Note: 'arguments' field is missing
                    }
                ]
            }
        }

        wrapper = OCIStreamWrapper(
            completion_stream=iter([]),
            model="meta.llama-3.1-405b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock()
        )

        # This should not raise a ValidationError
        result = wrapper._handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert len(result.choices) == 1
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == ""

    def test_stream_chunk_with_missing_id_field(self):
        """
        Test that streaming chunks with tool calls missing 'id' field are handled.
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
                        "name": "get_weather",
                        "arguments": '{"location": "San Francisco"}'
                        # Note: 'id' field is missing
                    }
                ]
            }
        }

        wrapper = OCIStreamWrapper(
            completion_stream=iter([]),
            model="meta.llama-3.1-405b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock()
        )

        result = wrapper._handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0]["id"] == ""

    def test_stream_chunk_with_missing_name_field(self):
        """
        Test that streaming chunks with tool calls missing 'name' field are handled.
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
                        "arguments": '{"location": "San Francisco"}'
                        # Note: 'name' field is missing
                    }
                ]
            }
        }

        wrapper = OCIStreamWrapper(
            completion_stream=iter([]),
            model="meta.llama-3.1-405b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock()
        )

        result = wrapper._handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0]["function"]["name"] == ""

    def test_stream_chunk_with_all_missing_fields(self):
        """
        Test that streaming chunks with tool calls missing all optional fields are handled.
        """
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": None,
                "toolCalls": [
                    {
                        "type": "FUNCTION"
                        # All fields missing: id, name, arguments
                    }
                ]
            }
        }

        wrapper = OCIStreamWrapper(
            completion_stream=iter([]),
            model="meta.llama-3.1-405b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock()
        )

        result = wrapper._handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert result.choices[0].delta.tool_calls[0]["id"] == ""
        assert result.choices[0].delta.tool_calls[0]["function"]["name"] == ""
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == ""

    def test_stream_chunk_with_complete_tool_call(self):
        """
        Test that streaming chunks with complete tool calls still work correctly.
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
                        "arguments": '{"location": "San Francisco", "unit": "celsius"}'
                    }
                ]
            }
        }

        wrapper = OCIStreamWrapper(
            completion_stream=iter([]),
            model="meta.llama-3.1-405b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock()
        )

        result = wrapper._handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 1
        assert result.choices[0].delta.tool_calls[0]["id"] == "call_abc123"
        assert result.choices[0].delta.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == '{"location": "San Francisco", "unit": "celsius"}'

    def test_stream_chunk_with_multiple_tool_calls_missing_fields(self):
        """
        Test that streaming chunks with multiple tool calls, some with missing fields, are handled.
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
                        "id": "call_1",
                        "name": "get_weather"
                        # Missing arguments
                    },
                    {
                        "type": "FUNCTION",
                        "name": "get_time",
                        "arguments": '{"timezone": "UTC"}'
                        # Missing id
                    },
                    {
                        "type": "FUNCTION",
                        "id": "call_3",
                        "name": "calculate",
                        "arguments": '{"expression": "2+2"}'
                        # Complete
                    }
                ]
            }
        }

        wrapper = OCIStreamWrapper(
            completion_stream=iter([]),
            model="meta.llama-3.1-405b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock()
        )

        result = wrapper._handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.tool_calls is not None
        assert len(result.choices[0].delta.tool_calls) == 3

        # First tool call - missing arguments
        assert result.choices[0].delta.tool_calls[0]["id"] == "call_1"
        assert result.choices[0].delta.tool_calls[0]["function"]["name"] == "get_weather"
        assert result.choices[0].delta.tool_calls[0]["function"]["arguments"] == ""

        # Second tool call - missing id
        assert result.choices[0].delta.tool_calls[1]["id"] == ""
        assert result.choices[0].delta.tool_calls[1]["function"]["name"] == "get_time"
        assert result.choices[0].delta.tool_calls[1]["function"]["arguments"] == '{"timezone": "UTC"}'

        # Third tool call - complete
        assert result.choices[0].delta.tool_calls[2]["id"] == "call_3"
        assert result.choices[0].delta.tool_calls[2]["function"]["name"] == "calculate"
        assert result.choices[0].delta.tool_calls[2]["function"]["arguments"] == '{"expression": "2+2"}'

    def test_stream_chunk_without_tool_calls(self):
        """
        Test that streaming chunks without tool calls continue to work as before.
        """
        chunk_data = {
            "index": 0,
            "finishReason": None,
            "message": {
                "role": "ASSISTANT",
                "content": [
                    {
                        "type": "TEXT",
                        "text": "Hello, how can I help you?"
                    }
                ]
            }
        }

        wrapper = OCIStreamWrapper(
            completion_stream=iter([]),
            model="meta.llama-3.1-405b-instruct",
            custom_llm_provider="oci",
            logging_obj=MagicMock()
        )

        result = wrapper._handle_generic_stream_chunk(chunk_data)

        assert isinstance(result, ModelResponseStream)
        assert result.choices[0].delta.content == "Hello, how can I help you?"
        assert result.choices[0].delta.tool_calls is None
