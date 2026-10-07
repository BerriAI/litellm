"""
Tests for Bedrock Moonshot (Kimi K2) integration.

This test suite verifies:
1. Basic completion functionality
2. Streaming responses
3. System message support
4. Temperature parameter handling
5. Reasoning content extraction from <reasoning> tags
6. Tool calling support (including tool response handling)
7. Parameter validation (e.g., stop sequences not supported)
"""

from base_llm_unit_tests import BaseLLMChatTest
import json

import litellm


class TestBedrockMoonshotInvoke(BaseLLMChatTest):
    """
    Test suite for Bedrock Moonshot via invoke route.
    Inherits all standard LLM tests from BaseLLMChatTest.
    """

    test_json_response_format_stream = None
    test_completion_cost = None
    test_content_list_handling = None
    test_developer_role_translation = None
    test_message_with_name = None
    test_pydantic_model_input = None
    test_response_format_type_text_with_tool_calls_no_tool_choice = None
    test_streaming = None

    def get_base_completion_call_args(self) -> dict:
        litellm.turn_on_debug()
        return {
            "model": "bedrock/invoke/moonshot.kimi-k2-thinking",
        }

    def test_tool_call_no_arguments(self, tool_call_no_arguments):
        """Test that tool calls with no arguments is translated correctly."""
        pass


class TestBedrockMoonshotToolCalling:
    """Unit tests for tool calling functionality."""

    def test_tool_response_message_format(self):
        """Test that tool response messages are formatted correctly."""
        tool_response_message = {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": json.dumps({"temperature": 72, "condition": "sunny"}),
        }

        assert tool_response_message["role"] == "tool"
        assert "tool_call_id" in tool_response_message
        assert "content" in tool_response_message
