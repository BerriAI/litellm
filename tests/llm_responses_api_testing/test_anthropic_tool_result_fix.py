"""
Test to verify the fix for Anthropic tool_result issue.

This test verifies that when using previous_response_id with tool_result,
the fix ensures tool_calls are added to the previous assistant message.
"""
import os
import sys
import pytest
import json
from unittest.mock import patch, AsyncMock

sys.path.insert(0, os.path.abspath("../.."))
import litellm
from litellm.responses.litellm_completion_transformation.transformation import (
    LiteLLMCompletionResponsesConfig,
    TOOL_CALLS_CACHE
)
from litellm.llms.anthropic.chat.transformation import AnthropicConfig


def test_fix_ensures_tool_calls_for_tool_results():
    """
    Test that the fix ensures tool_calls are added to assistant messages
    when tool_results are present but tool_calls are missing.
    """
    shell_tool = {
        "type": "function",
        "function": {
            "name": "shell",
            "description": "Runs a shell command, and returns its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "array", "items": {"type": "string"}},
                    "workdir": {"type": "string", "description": "The working directory for the command."}
                },
                "required": ["command"]
            }
        }
    }
    
    tool_call_id = "toolu_0123456789abcdef"
    
    # Cache the tool_call definition (simulating what happens when a response is returned)
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
    
    # Simulate messages that would be reconstructed from spend logs
    # The assistant message is missing tool_calls (the bug scenario)
    messages_missing_tool_calls = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "make a hello world html file"}]
        },
        {
            "role": "assistant",
            "content": "I'll help you create that HTML file."
            # NOTE: Missing tool_calls here - this is the bug scenario
        },
        {
            "role": "tool",
            "content": '{"output":"<html>...</html>"}',
            "tool_call_id": tool_call_id
        }
    ]
    
    # Apply the fix
    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages_missing_tool_calls,
        tools=[shell_tool]
    )
    
    # Verify the fix worked
    assistant_message = None
    for msg in fixed_messages:
        if msg.get("role") == "assistant":
            assistant_message = msg
            break
    
    assert assistant_message is not None, "Assistant message should be present"
    
    # Check if tool_calls were added
    tool_calls = assistant_message.get("tool_calls") or []
    assert len(tool_calls) > 0, (
        f"Fix should have added tool_calls to assistant message. "
        f"Found: {json.dumps(assistant_message, indent=2)}"
    )
    
    # Verify the tool_call has the correct ID
    found_tool_call = False
    for tool_call in tool_calls:
        tool_call_id_from_msg = tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)
        if tool_call_id_from_msg == tool_call_id:
            found_tool_call = True
            break
    
    assert found_tool_call, (
        f"Tool call with ID {tool_call_id} should be present in assistant message. "
        f"Found tool_calls: {json.dumps(tool_calls, indent=2, default=str)}"
    )
    
    # Now verify the Anthropic transformation works
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
    
    # Find the assistant message in Anthropic format
    anthropic_assistant_msg = None
    for msg in anthropic_messages:
        if msg.get("role") == "assistant":
            anthropic_assistant_msg = msg
            break
    
    assert anthropic_assistant_msg is not None, "Assistant message should be present in Anthropic format"
    
    # Verify the assistant message has tool_use blocks
    assistant_content = anthropic_assistant_msg.get("content", [])
    tool_use_blocks = [
        block for block in assistant_content
        if isinstance(block, dict) and block.get("type") == "tool_use"
    ]
    
    assert len(tool_use_blocks) > 0, (
        f"After fix, assistant message should have tool_use blocks. "
        f"Found content: {json.dumps(assistant_content, indent=2)}"
    )
    
    # Verify the tool_use block has the correct ID
    tool_use_id = tool_use_blocks[0].get("id")
    assert tool_use_id == tool_call_id, (
        f"Tool use ID {tool_use_id} should match tool_call_id {tool_call_id}"
    )
    
    print("\n" + "=" * 80)
    print("[PASS] Fix verified: tool_calls are added when missing")
    print("=" * 80)
    print(f"  Tool use blocks: {len(tool_use_blocks)}")
    print(f"  Tool use ID: {tool_use_id}")
    print("\nThe fix ensures that when tool_results are present but tool_calls are")
    print("missing from the assistant message, they are added from cache or tools.")


if __name__ == "__main__":
    test_fix_ensures_tool_calls_for_tool_results()
