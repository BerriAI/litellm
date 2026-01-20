"""
Unit tests for the fix to merge multiple consecutive function_call items
into a single assistant message.

This test verifies the fix for the issue where multiple function_call items
were creating separate assistant messages, causing Claude's Messages API
to fail with: "Expected toolResult blocks at messages.2.content for the following Ids..."
"""

import pytest

from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
)


class TestMultipleFunctionCallsMerging:
    """Test cases for merging multiple function_call items into a single assistant message"""

    def test_multiple_function_calls_merge_into_single_assistant_message(self):
        """
        Test that multiple consecutive function_call items are merged into a single
        assistant message with multiple tool_calls, fixing the Claude Messages API error.
        """
        # Simulate the input from the Responses API after a streaming response
        # This is what we get when Claude makes multiple tool calls
        input_items = [
            # Initial user message
            {
                "content": [
                    {
                        "text": "\n\n\nStructured Inputs:\ncompany: REI.com\n\n",
                        "type": "input_text",
                        "valid": True,
                    }
                ],
                "role": "user",
                "type": "message",
                "valid": True,
            },
            # Empty assistant message (from the initial streaming response)
            {
                "id": "chatcmpl-0363cad9-9ecf-4a2a-b181-b2e68bb52913",
                "content": [{"annotations": [], "text": "", "type": "output_text", "valid": True}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
                "valid": True,
            },
            # Multiple function calls from the assistant
            {
                "arguments": '{"actionText": "Researching REI homepage", "contextText": "to gather company overview and positioning", "domain": "rei.com"}',
                "call_id": "tooluse_gtWtOTKhTnmCQKrs60fSlg",
                "name": "Research_company_web_page",
                "type": "function_call",
                "id": "tooluse_gtWtOTKhTnmCQKrs60fSlg",
                "status": "completed",
                "valid": True,
            },
            {
                "arguments": '{"actionText": "Analyzing REI offerings", "contextText": "to understand products and target market", "domain": "rei.com"}',
                "call_id": "tooluse_1M_saUAKTj-MC8EbawtbPg",
                "name": "Infer_company_value_proposition_and_ICP",
                "type": "function_call",
                "id": "tooluse_1M_saUAKTj-MC8EbawtbPg",
                "status": "completed",
                "valid": True,
            },
            {
                "arguments": '{"actionText": "Gathering recent news", "contextText": "to identify trigger events and company updates", "companyName": "REI", "domain": "rei.com"}',
                "call_id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw",
                "name": "Research_company_news",
                "type": "function_call",
                "id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw",
                "status": "completed",
                "valid": True,
            },
            # Tool outputs
            {
                "call_id": "tooluse_gtWtOTKhTnmCQKrs60fSlg",
                "output": '"research: data here"',
                "type": "function_call_output",
                "id": "tooluse_gtWtOTKhTnmCQKrs60fSlg",
                "status": "completed",
                "valid": True,
            },
            {
                "call_id": "tooluse_1M_saUAKTj-MC8EbawtbPg",
                "output": '"targetIcp: data here"',
                "type": "function_call_output",
                "id": "tooluse_1M_saUAKTj-MC8EbawtbPg",
                "status": "completed",
                "valid": True,
            },
            {
                "call_id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw",
                "output": '"sources: data here"',
                "type": "function_call_output",
                "id": "tooluse_ynl6q84sSSOVTJ6mfTV5Qw",
                "status": "completed",
                "valid": True,
            },
        ]

        # Transform to chat completion messages
        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=input_items
        )

        # Expected behavior:
        # 1. User message
        # 2. Assistant message with ALL 3 tool calls merged into it
        # 3-5. Three tool messages

        # Count assistant messages
        assistant_messages = [
            m
            for m in messages
            if (m.get("role") if isinstance(m, dict) else m.role) == "assistant"
        ]

        # The fix: We should have exactly 1 assistant message (with all tool calls merged)
        assert (
            len(assistant_messages) == 1
        ), f"Expected 1 assistant message, got {len(assistant_messages)}"

        # Check that the single assistant message has all 3 tool calls
        assistant_msg = assistant_messages[0]
        tool_calls = (
            assistant_msg.get("tool_calls")
            if isinstance(assistant_msg, dict)
            else assistant_msg.tool_calls
        )
        assert tool_calls is not None, "Expected tool_calls to be present"
        assert len(tool_calls) == 3, f"Expected 3 tool calls, got {len(tool_calls)}"

        # Verify all tool call IDs are present
        tool_call_ids = {
            (tc.get("id") if isinstance(tc, dict) else tc.id) for tc in tool_calls
        }
        expected_ids = {
            "tooluse_gtWtOTKhTnmCQKrs60fSlg",
            "tooluse_1M_saUAKTj-MC8EbawtbPg",
            "tooluse_ynl6q84sSSOVTJ6mfTV5Qw",
        }
        assert tool_call_ids == expected_ids, f"Expected tool call IDs {expected_ids}, got {tool_call_ids}"

    def test_function_calls_without_preceding_assistant_message(self):
        """
        Test that function_call items create a new assistant message when there's
        no preceding assistant message to merge into.
        """
        input_items = [
            # User message
            {
                "content": [{"text": "Test message", "type": "input_text"}],
                "role": "user",
                "type": "message",
            },
            # Multiple function calls (no preceding assistant message)
            {
                "arguments": '{"arg": "value1"}',
                "call_id": "call_1",
                "name": "tool1",
                "type": "function_call",
            },
            {
                "arguments": '{"arg": "value2"}',
                "call_id": "call_2",
                "name": "tool2",
                "type": "function_call",
            },
        ]

        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=input_items
        )

        # Should have user message + assistant message with 2 tool calls
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

        assistant_msg = messages[1]
        assert (assistant_msg.get("role") if isinstance(assistant_msg, dict) else assistant_msg.role) == "assistant"

        tool_calls = (
            assistant_msg.get("tool_calls")
            if isinstance(assistant_msg, dict)
            else assistant_msg.tool_calls
        )
        assert tool_calls is not None
        assert len(tool_calls) == 2, f"Expected 2 tool calls, got {len(tool_calls)}"

    def test_function_calls_with_non_empty_assistant_message(self):
        """
        Test that function_call items create a NEW assistant message when the
        preceding assistant message has non-empty content.
        """
        input_items = [
            # User message
            {
                "content": [{"text": "Test message", "type": "input_text"}],
                "role": "user",
                "type": "message",
            },
            # Assistant message with content
            {
                "content": [{"text": "I will call some tools", "type": "output_text"}],
                "role": "assistant",
                "type": "message",
            },
            # Function calls
            {
                "arguments": '{"arg": "value1"}',
                "call_id": "call_1",
                "name": "tool1",
                "type": "function_call",
            },
        ]

        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=input_items
        )

        # Should have user message + assistant message with content + assistant message with tool call
        assert len(messages) == 3, f"Expected 3 messages, got {len(messages)}"

        # First assistant message should have content, no tool calls
        first_assistant = messages[1]
        assert (first_assistant.get("role") if isinstance(first_assistant, dict) else first_assistant.role) == "assistant"
        content = first_assistant.get("content") if isinstance(first_assistant, dict) else first_assistant.content
        assert content is not None

        # Second assistant message should have tool calls, no content
        second_assistant = messages[2]
        assert (second_assistant.get("role") if isinstance(second_assistant, dict) else second_assistant.role) == "assistant"
        tool_calls = (
            second_assistant.get("tool_calls")
            if isinstance(second_assistant, dict)
            else second_assistant.tool_calls
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1

    def test_single_function_call_behavior_unchanged(self):
        """
        Test that a single function_call item still works correctly
        (regression test for existing behavior).
        """
        input_items = [
            # User message
            {
                "content": [{"text": "Test message", "type": "input_text"}],
                "role": "user",
                "type": "message",
            },
            # Single function call
            {
                "arguments": '{"arg": "value1"}',
                "call_id": "call_1",
                "name": "tool1",
                "type": "function_call",
            },
        ]

        messages = LiteLLMCompletionResponsesConfig._transform_response_input_param_to_chat_completion_message(
            input=input_items
        )

        # Should have user message + assistant message with 1 tool call
        assert len(messages) == 2

        assistant_msg = messages[1]
        tool_calls = (
            assistant_msg.get("tool_calls")
            if isinstance(assistant_msg, dict)
            else assistant_msg.tool_calls
        )
        assert tool_calls is not None
        assert len(tool_calls) == 1
