"""
Tests for LiteLLMCompletionTransformationHandler argument validation.

Tests the _ensure_all_tool_calls_have_valid_json_arguments function that
validates and fixes tool_calls arguments in messages for domestic models.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.responses.litellm_completion_transformation.handler import (
    LiteLLMCompletionTransformationHandler,
)


class TestEnsureToolCallsValidJsonArguments(unittest.TestCase):
    """Test _ensure_all_tool_calls_have_valid_json_arguments static method."""

    def setUp(self):
        """Set up handler instance."""
        self.handler = LiteLLMCompletionTransformationHandler()

    def test_valid_json_arguments(self):
        """Test that valid JSON arguments are preserved."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": '{"key": "value"}',
                        },
                    }
                ],
            }
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        self.assertEqual(
            result[0]["tool_calls"][0]["function"]["arguments"], '{"key": "value"}'
        )

    def test_empty_arguments(self):
        """Test that empty arguments are converted to '{}'."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": "",
                        },
                    }
                ],
            }
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        self.assertEqual(result[0]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_none_arguments(self):
        """Test that None arguments are converted to '{}'."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": None,
                        },
                    }
                ],
            }
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        self.assertEqual(result[0]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_invalid_json_arguments(self):
        """Test that invalid JSON arguments are converted to '{}'."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": "not a valid json",
                        },
                    }
                ],
            }
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        self.assertEqual(result[0]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_dict_arguments_converted_to_json(self):
        """Test that dict arguments are converted to JSON string."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": {"key": "value", "number": 42},
                        },
                    }
                ],
            }
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        # dict should be converted to JSON string
        self.assertEqual(
            result[0]["tool_calls"][0]["function"]["arguments"],
            '{"key": "value", "number": 42}',
        )

    def test_whitespace_arguments(self):
        """Test that whitespace-only arguments are converted to '{}'."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "test_func",
                            "arguments": "   ",
                        },
                    }
                ],
            }
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        self.assertEqual(result[0]["tool_calls"][0]["function"]["arguments"], "{}")

    def test_non_assistant_messages_preserved(self):
        """Test that non-assistant messages are not modified."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["role"], "user")
        self.assertEqual(result[1]["role"], "assistant")

    def test_assistant_without_tool_calls_preserved(self):
        """Test that assistant messages without tool_calls are preserved."""
        messages = [
            {"role": "assistant", "content": "No tools here"},
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        self.assertEqual(result[0]["content"], "No tools here")

    def test_multiple_tool_calls_in_one_message(self):
        """Test handling of multiple tool calls in a single message."""
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "func1",
                            "arguments": '{"valid": true}',
                        },
                    },
                    {
                        "id": "call_2",
                        "type": "function",
                        "function": {
                            "name": "func2",
                            "arguments": "invalid",
                        },
                    },
                    {
                        "id": "call_3",
                        "type": "function",
                        "function": {
                            "name": "func3",
                            "arguments": None,
                        },
                    },
                ],
            }
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        # First should be preserved (valid)
        self.assertEqual(
            result[0]["tool_calls"][0]["function"]["arguments"], '{"valid": true}'
        )
        # Second should be fixed (invalid -> {})
        self.assertEqual(result[0]["tool_calls"][1]["function"]["arguments"], "{}")
        # Third should be fixed (None -> {})
        self.assertEqual(result[0]["tool_calls"][2]["function"]["arguments"], "{}")

    def test_empty_messages_list(self):
        """Test handling of empty messages list."""
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments([])
        self.assertEqual(result, [])

    def test_complex_valid_json(self):
        """Test that complex valid JSON is preserved."""
        complex_json = json.dumps(
            {
                "nested": {"deep": {"value": [1, 2, 3]}},
                "string": 'with "escaped" quotes',
                "number": 123.456,
            }
        )
        messages = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "complex_func",
                            "arguments": complex_json,
                        },
                    }
                ],
            }
        ]
        result = self.handler._ensure_all_tool_calls_have_valid_json_arguments(messages)
        self.assertEqual(
            result[0]["tool_calls"][0]["function"]["arguments"], complex_json
        )


if __name__ == "__main__":
    unittest.main()
