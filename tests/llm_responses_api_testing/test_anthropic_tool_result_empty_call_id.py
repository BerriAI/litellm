"""
Test to reproduce and verify fix for Anthropic tool_result issue with empty call_id.

This test reproduces the exact error:
"messages.0.content.0: unexpected `tool_use_id` found in `tool_result` blocks: tool_use_id. 
Each `tool_result` block must have a corresponding `tool_use` block in the previous message."

The issue occurs when:
1. Using previous_response_id to reconstruct messages
2. A tool_result message has an empty tool_call_id
3. The message is sent to Anthropic without a corresponding tool_use block
"""
import os
import sys
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    TOOL_CALLS_CACHE
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


def test_empty_tool_call_id_is_skipped():
    """
    Test that tool messages with empty tool_call_id are skipped
    when transforming function_call_output to chat completion messages.
    """
    # Simulate a function_call_output with empty call_id (the bug scenario)
    tool_call_output_empty = {
        "type": "function_call_output",
        "call_id": "",  # Empty call_id - this causes the issue
        "output": '{"output":"test output","metadata":{"exit_code":0}}'
    }
    
    # Transform should return empty list (skip the message)
    result = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output_empty
    )
    
    assert result == [], (
        "Tool messages with empty call_id should be skipped, not created"
    )
    print("[OK] Empty call_id messages are correctly skipped")


def test_empty_tool_call_id_in_messages_list_is_removed():
    """
    Test that tool messages with empty tool_call_id are removed
    from the messages list when ensuring tool_results have corresponding tool_calls.
    """
    # Simulate messages with a tool message that has empty tool_call_id
    messages = [
        {
            "role": "assistant",
            "content": "I'll help you with that."
        },
        {
            "role": "tool",
            "content": '{"output":"test"}',
            "tool_call_id": ""  # Empty tool_call_id - should be removed
        }
    ]
    
    # The fix should remove messages with empty tool_call_id
    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages,
        tools=None
    )
    
    # The tool message with empty tool_call_id should be removed
    tool_messages = [msg for msg in fixed_messages if msg.get("role") == "tool"]
    assert len(tool_messages) == 0, (
        "Tool messages with empty tool_call_id should be removed from the list"
    )
    print("[OK] Empty tool_call_id messages are correctly removed from messages list")


def test_tool_call_id_recovered_from_previous_assistant():
    """
    Test that empty tool_call_id can be recovered from the previous assistant message's tool_calls.
    """
    tool_call_id = "toolu_0123456789abcdef"
    
    messages = [
        {
            "role": "assistant",
            "content": "I'll call the tool.",
            "tool_calls": [
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": "shell",
                        "arguments": '{"command": ["echo", "hello"]}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "content": '{"output":"hello"}',
            "tool_call_id": ""  # Empty, but should be recovered from assistant message
        }
    ]
    
    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages,
        tools=None
    )
    
    # The tool message should have its tool_call_id recovered
    tool_message = next((msg for msg in fixed_messages if msg.get("role") == "tool"), None)
    assert tool_message is not None, "Tool message should still be present"
    assert tool_message.get("tool_call_id") == tool_call_id, (
        f"Tool call_id should be recovered from assistant message. "
        f"Expected: {tool_call_id}, Got: {tool_message.get('tool_call_id')}"
    )
    print(f"[OK] Tool call_id recovered: {tool_message.get('tool_call_id')}")


def test_tool_calls_added_when_missing():
    """
    Test that tool_calls are added to assistant message when tool_result is present
    but tool_calls are missing (the main fix scenario).
    """
    tool_call_id = "toolu_0123456789abcdef"
    
    # Cache the tool_call definition
    TOOL_CALLS_CACHE.set_cache(
        key=tool_call_id,
        value={
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": '{"command": ["echo", "hello"]}'
            }
        }
    )
    
    shell_tool = {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Runs a shell command"
        }
    }
    
    # Messages with tool_result but missing tool_calls in assistant message
    messages = [
        {
            "role": "assistant",
            "content": "I'll call the tool."
            # Missing tool_calls - this is the bug scenario
        },
        {
            "role": "tool",
            "content": '{"output":"hello"}',
            "tool_call_id": tool_call_id
        }
    ]
    
    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages,
        tools=[shell_tool]
    )
    
    # The assistant message should now have tool_calls
    assistant_message = next((msg for msg in fixed_messages if msg.get("role") == "assistant"), None)
    assert assistant_message is not None, "Assistant message should be present"
    
    tool_calls = assistant_message.get("tool_calls", [])
    assert len(tool_calls) > 0, (
        "Assistant message should have tool_calls added when tool_result is present"
    )
    
    # Verify the tool_call has the correct ID
    first_tool_call = tool_calls[0]
    tool_call_id_from_message = first_tool_call.get("id") if isinstance(first_tool_call, dict) else getattr(first_tool_call, "id", None)
    assert tool_call_id_from_message == tool_call_id, (
        f"Tool call ID should match. Expected: {tool_call_id}, Got: {tool_call_id_from_message}"
    )
    print(f"[OK] Tool calls added to assistant message: {len(tool_calls)} tool_call(s)")


def test_anthropic_transformation_with_fixed_messages():
    """
    Test that the fixed messages work correctly with Anthropic transformation.
    """
    tool_call_id = "toolu_0123456789abcdef"
    
    # Cache the tool_call
    TOOL_CALLS_CACHE.set_cache(
        key=tool_call_id,
        value={
            "id": tool_call_id,
            "type": "function",
            "function": {
                "name": "shell",
                "arguments": '{"command": ["echo", "hello"]}'
            }
        }
    )
    
    shell_tool = {
        "name": "shell",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "array", "items": {"type": "string"}}
            }
        },
        "description": "Runs a shell command"
    }
    
    # Messages that would cause the error without the fix
    messages = [
        {
            "role": "assistant",
            "content": "I'll help you."
            # Missing tool_calls
        },
        {
            "role": "tool",
            "content": '{"output":"hello"}',
            "tool_call_id": tool_call_id
        }
    ]
    
    # Apply the fix
    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages,
        tools=[shell_tool]
    )
    
    # Transform to Anthropic format
    anthropic_config = AnthropicConfig()
    optional_params = {"tools": [shell_tool]}
    
    anthropic_data = anthropic_config.transform_request(
        model="claude-3-7-sonnet-latest",
        messages=fixed_messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )
    
    anthropic_messages = anthropic_data.get("messages", [])
    
    # Find the assistant message
    anthropic_assistant_msg = next(
        (msg for msg in anthropic_messages if msg.get("role") == "assistant"),
        None
    )
    
    assert anthropic_assistant_msg is not None, "Assistant message should be present"
    
    # Verify it has tool_use blocks
    assistant_content = anthropic_assistant_msg.get("content", [])
    tool_use_blocks = [
        block for block in assistant_content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    
    assert len(tool_use_blocks) > 0, (
        f"After fix, assistant message should have tool_use blocks. "
        f"Found content: {assistant_content}"
    )
    
    # Verify the tool_use block has the correct ID
    tool_use_id = tool_use_blocks[0].get("id")
    assert tool_use_id == tool_call_id, (
        f"Tool use ID should match. Expected: {tool_call_id}, Got: {tool_use_id}"
    )
    
    print(f"[OK] Anthropic transformation successful with {len(tool_use_blocks)} tool_use block(s)")


if __name__ == "__main__":
    test_empty_tool_call_id_is_skipped()
    test_empty_tool_call_id_in_messages_list_is_removed()
    test_tool_call_id_recovered_from_previous_assistant()
    test_tool_calls_added_when_missing()
    test_anthropic_transformation_with_fixed_messages()
    print("\n" + "=" * 80)
    print("[PASS] All tests passed - fix verified!")
    print("=" * 80)
