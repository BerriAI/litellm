"""
Test cases for functionCall args serialization in Vertex AI Gemini.

This test file specifically tests the edge cases where Vertex AI might return
functionCall args in unexpected formats that could lead to invalid JSON strings
like: {"x":"x"}{"a":"a"}
"""
import json
from typing import List, Optional

import pytest

from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)
from litellm.types.llms.vertex_ai import HttpxPartType


class TestFunctionCallArgsSerialization:
    """Test cases for functionCall args serialization edge cases."""

    def test_normal_dict_args(self):
        """Test normal case: args is a dict."""
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Boston", "unit": "celsius"},
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "get_weather"
        
        # Verify arguments is a valid JSON string
        arguments = tools[0]["function"]["arguments"]
        assert isinstance(arguments, str)
        # Should be valid JSON
        parsed = json.loads(arguments)
        assert parsed == {"location": "Boston", "unit": "celsius"}

    def test_none_args(self):
        """Test case: args is None."""
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": None,
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        arguments = tools[0]["function"]["arguments"]
        # Should serialize None to "null" or empty dict
        assert isinstance(arguments, str)
        parsed = json.loads(arguments)
        # json.dumps(None) returns "null"
        assert parsed is None or parsed == {}

    def test_args_as_string_valid_json(self):
        """Test case: args is already a valid JSON string."""
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": '{"location": "Boston"}',  # String, not dict
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        arguments = tools[0]["function"]["arguments"]
        # If args is a string, json.dumps will double-encode it
        # This would result in: "{\"location\": \"Boston\"}"
        assert isinstance(arguments, str)
        # This is the problematic case - string gets double-encoded
        # The result would be a JSON string containing a JSON string
        parsed = json.loads(arguments)
        # If it's double-encoded, parsed would be a string, not a dict
        if isinstance(parsed, str):
            # Double-encoded case
            inner_parsed = json.loads(parsed)
            assert inner_parsed == {"location": "Boston"}
        else:
            # Normal case (shouldn't happen if args is string)
            assert parsed == {"location": "Boston"}

    def test_args_as_string_invalid_json_concatenated(self):
        """Test case: args is a string with concatenated JSON objects (the bug case).
        
        When args is a string like '{"x":"x"}{"a":"a"}', json.dumps() will serialize it
        as a JSON string, resulting in: "{\"x\":\"x\"}{\"a\":\"a\"}"
        This is a valid JSON string (the outer quotes), but the content inside is invalid JSON.
        When you try to parse the inner content, it fails.
        """
        # This simulates the case where Vertex might return something like:
        # args = '{"x":"x"}{"a":"a"}'  # Two JSON objects concatenated
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": '{"x":"x"}{"a":"a"}',  # Invalid concatenated JSON
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        arguments = tools[0]["function"]["arguments"]
        assert isinstance(arguments, str)
        
        # json.dumps() on a string will escape it, so we get:
        # arguments = '"{\\"x\\":\\"x\\"}{\\"a\\":\\"a\\"}"'
        # This is a valid JSON string (the outer quotes), but the inner content is invalid
        parsed_outer = json.loads(arguments)
        assert isinstance(parsed_outer, str)
        
        # The inner string is invalid JSON (two objects concatenated)
        # This is the bug: the inner content cannot be parsed as valid JSON
        with pytest.raises(json.JSONDecodeError):
            json.loads(parsed_outer)
        
        # The arguments string would be: "{\"x\":\"x\"}{\"a\":\"a\"}"
        # Which when parsed gives: '{"x":"x"}{"a":"a"}' (invalid JSON)

    def test_args_as_array(self):
        """Test case: args is an array (unexpected but possible)."""
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": [{"x": "x"}, {"a": "a"}],  # Array of objects
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        arguments = tools[0]["function"]["arguments"]
        assert isinstance(arguments, str)
        # Should serialize array correctly
        parsed = json.loads(arguments)
        assert parsed == [{"x": "x"}, {"a": "a"}]

    def test_args_missing_key(self):
        """Test case: args key is missing from functionCall.
        
        This will raise a KeyError because the code directly accesses part["functionCall"]["args"]
        without checking if the key exists. This is a bug that should be fixed.
        """
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    # args key missing
                }
            }
        ]

        # This should raise KeyError because args key is missing
        with pytest.raises(KeyError):
            VertexGeminiConfig._transform_parts(
                parts=parts, cumulative_tool_call_idx=0, is_function_call=False
            )

    def test_multiple_function_calls(self):
        """Test case: multiple function calls in parts."""
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {"location": "Boston"},
                }
            },
            {
                "functionCall": {
                    "name": "get_time",
                    "args": {"timezone": "EST"},
                }
            },
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 2
        assert tools[0]["function"]["name"] == "get_weather"
        assert tools[1]["function"]["name"] == "get_time"
        
        # Both should have valid JSON arguments
        args1 = json.loads(tools[0]["function"]["arguments"])
        args2 = json.loads(tools[1]["function"]["arguments"])
        assert args1 == {"location": "Boston"}
        assert args2 == {"timezone": "EST"}

    def test_args_with_vertex_protobuf_format(self):
        """Test case: args in Vertex protobuf format with string_value, etc."""
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {
                        "location": {"string_value": "Boston, MA"},
                        "unit": {"string_value": "celsius"},
                    },
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        arguments = tools[0]["function"]["arguments"]
        assert isinstance(arguments, str)
        # Should serialize the nested structure correctly
        parsed = json.loads(arguments)
        assert "location" in parsed
        assert "unit" in parsed

    def test_args_as_empty_dict(self):
        """Test case: args is an empty dict."""
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {},
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        arguments = tools[0]["function"]["arguments"]
        assert isinstance(arguments, str)
        parsed = json.loads(arguments)
        assert parsed == {}

    def test_args_with_special_characters(self):
        """Test case: args contains special characters that need escaping."""
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": {
                        "location": 'Boston, MA "downtown"',
                        "note": "Line 1\nLine 2",
                    },
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        arguments = tools[0]["function"]["arguments"]
        assert isinstance(arguments, str)
        # Should handle special characters correctly
        parsed = json.loads(arguments)
        assert parsed["location"] == 'Boston, MA "downtown"'
        assert parsed["note"] == "Line 1\nLine 2"

    def test_args_as_list_of_strings_that_look_like_json(self):
        """Test case: args is a list containing strings that look like JSON objects."""
        # This could potentially cause issues if not handled correctly
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "get_weather",
                    "args": ['{"x":"x"}', '{"a":"a"}'],  # List of JSON strings
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        arguments = tools[0]["function"]["arguments"]
        assert isinstance(arguments, str)
        # Should serialize list correctly
        parsed = json.loads(arguments)
        assert isinstance(parsed, list)
        assert parsed == ['{"x":"x"}', '{"a":"a"}']

    def test_args_as_dict_with_nested_structures(self):
        """Test case: args contains nested dicts and lists."""
        parts: List[HttpxPartType] = [
            {
                "functionCall": {
                    "name": "complex_function",
                    "args": {
                        "nested": {"key": "value"},
                        "list": [1, 2, 3],
                        "mixed": [{"a": 1}, {"b": 2}],
                    },
                }
            }
        ]

        function, tools, idx = VertexGeminiConfig._transform_parts(
            parts=parts, cumulative_tool_call_idx=0, is_function_call=False
        )

        assert tools is not None
        assert len(tools) == 1
        arguments = tools[0]["function"]["arguments"]
        assert isinstance(arguments, str)
        parsed = json.loads(arguments)
        assert parsed["nested"] == {"key": "value"}
        assert parsed["list"] == [1, 2, 3]
        assert parsed["mixed"] == [{"a": 1}, {"b": 2}]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

