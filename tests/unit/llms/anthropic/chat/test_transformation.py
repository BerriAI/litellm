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

import asyncio
import importlib
import json

import pytest

import litellm
from litellm.llms.anthropic.chat.transformation import AnthropicConfig
from litellm.responses.litellm_completion_transformation.transformation import (
    TOOL_CALLS_CACHE,
    LiteLLMCompletionResponsesConfig,
)
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_empty_tool_call_id_is_skipped():
    """
    Test that tool messages with empty tool_call_id are skipped
    when transforming function_call_output to chat completion messages.
    """
    # Simulate a function_call_output with empty call_id (the bug scenario)
    tool_call_output_empty = {
        "type": "function_call_output",
        "call_id": "",  # Empty call_id - this causes the issue
        "output": '{"output":"test output","metadata":{"exit_code":0}}',
    }

    # Transform should return empty list (skip the message)
    result = LiteLLMCompletionResponsesConfig._transform_responses_api_tool_call_output_to_chat_completion_message(
        tool_call_output_empty
    )

    assert result == [], "Tool messages with empty call_id should be skipped, not created"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
def test_empty_tool_call_id_in_messages_list_is_removed():
    """
    Test that tool messages with empty tool_call_id are removed
    from the messages list when ensuring tool_results have corresponding tool_calls.
    """
    # Simulate messages with a tool message that has empty tool_call_id
    messages = [
        {"role": "assistant", "content": "I'll help you with that."},
        {
            "role": "tool",
            "content": '{"output":"test"}',
            "tool_call_id": "",  # Empty tool_call_id - should be removed
        },
    ]

    # The fix should remove messages with empty tool_call_id
    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages, tools=None
    )

    # The tool message with empty tool_call_id should be removed
    tool_messages = [msg for msg in fixed_messages if msg.get("role") == "tool"]
    assert len(tool_messages) == 0, "Tool messages with empty tool_call_id should be removed from the list"


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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
                        "arguments": '{"command": ["echo", "hello"]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "content": '{"output":"hello"}',
            "tool_call_id": "",  # Empty, but should be recovered from assistant message
        },
    ]

    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages, tools=None
    )

    # The tool message should have its tool_call_id recovered
    tool_message = next((msg for msg in fixed_messages if msg.get("role") == "tool"), None)
    assert tool_message is not None, "Tool message should still be present"
    assert tool_message.get("tool_call_id") == tool_call_id, (
        f"Tool call_id should be recovered from assistant message. "
        f"Expected: {tool_call_id}, Got: {tool_message.get('tool_call_id')}"
    )
    print(f"[OK] Tool call_id recovered: {tool_message.get('tool_call_id')}")


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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
                "arguments": '{"command": ["echo", "hello"]}',
            },
        },
    )

    shell_tool = {
        "type": "function",
        "function": {"name": "shell", "description": "Runs a shell command"},
    }

    # Messages with tool_result but missing tool_calls in assistant message
    messages = [
        {
            "role": "assistant",
            "content": "I'll call the tool.",
            # Missing tool_calls - this is the bug scenario
        },
        {"role": "tool", "content": '{"output":"hello"}', "tool_call_id": tool_call_id},
    ]

    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages, tools=[shell_tool]
    )

    # The assistant message should now have tool_calls
    assistant_message = next((msg for msg in fixed_messages if msg.get("role") == "assistant"), None)
    assert assistant_message is not None, "Assistant message should be present"

    tool_calls = assistant_message.get("tool_calls", [])
    assert len(tool_calls) > 0, "Assistant message should have tool_calls added when tool_result is present"

    # Verify the tool_call has the correct ID
    first_tool_call = tool_calls[0]
    tool_call_id_from_message = (
        first_tool_call.get("id") if isinstance(first_tool_call, dict) else getattr(first_tool_call, "id", None)
    )
    assert tool_call_id_from_message == tool_call_id, (
        f"Tool call ID should match. Expected: {tool_call_id}, Got: {tool_call_id_from_message}"
    )
    print(f"[OK] Tool calls added to assistant message: {len(tool_calls)} tool_call(s)")


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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
                "arguments": '{"command": ["echo", "hello"]}',
            },
        },
    )

    shell_tool = {
        "name": "shell",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "array", "items": {"type": "string"}}},
        },
        "description": "Runs a shell command",
    }

    # Messages that would cause the error without the fix
    messages = [
        {
            "role": "assistant",
            "content": "I'll help you.",
            # Missing tool_calls
        },
        {"role": "tool", "content": '{"output":"hello"}', "tool_call_id": tool_call_id},
    ]

    # Apply the fix
    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages, tools=[shell_tool]
    )

    # Transform to Anthropic format
    anthropic_config = AnthropicConfig()
    optional_params = {"tools": [shell_tool]}

    anthropic_data = anthropic_config.transform_request(
        model="claude-sonnet-4-5",
        messages=fixed_messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
    )

    anthropic_messages = anthropic_data.get("messages", [])

    # Find the assistant message
    anthropic_assistant_msg = next((msg for msg in anthropic_messages if msg.get("role") == "assistant"), None)

    assert anthropic_assistant_msg is not None, "Assistant message should be present"

    # Verify it has tool_use blocks
    assistant_content = anthropic_assistant_msg.get("content", [])
    tool_use_blocks = [
        block for block in assistant_content if isinstance(block, dict) and block.get("type") == "tool_use"
    ]

    assert len(tool_use_blocks) > 0, (
        f"After fix, assistant message should have tool_use blocks. Found content: {assistant_content}"
    )

    # Verify the tool_use block has the correct ID
    tool_use_id = tool_use_blocks[0].get("id")
    assert tool_use_id == tool_call_id, f"Tool use ID should match. Expected: {tool_call_id}, Got: {tool_use_id}"

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


@pytest.fixture()
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function")
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception as e:
        print(f"Error reloading litellm.proxy.proxy_server: {e}")
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    print(litellm)
    yield
    loop.close()
    asyncio.set_event_loop(None)


@pytest.mark.usefixtures("_vcr_outcome_gate", "setup_and_teardown")
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
                    "workdir": {
                        "type": "string",
                        "description": "The working directory for the command.",
                    },
                },
                "required": ["command"],
            },
        },
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
                "arguments": '{"command": ["echo", "hello"]}',
            },
        },
    )

    # Simulate messages that would be reconstructed from spend logs
    # The assistant message is missing tool_calls (the bug scenario)
    messages_missing_tool_calls = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "make a hello world html file"}],
        },
        {
            "role": "assistant",
            "content": "I'll help you create that HTML file.",
            # NOTE: Missing tool_calls here - this is the bug scenario
        },
        {
            "role": "tool",
            "content": '{"output":"<html>...</html>"}',
            "tool_call_id": tool_call_id,
        },
    ]

    # Apply the fix
    fixed_messages = LiteLLMCompletionResponsesConfig._ensure_tool_results_have_corresponding_tool_calls(
        messages=messages_missing_tool_calls, tools=[shell_tool]
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
        f"Fix should have added tool_calls to assistant message. Found: {json.dumps(assistant_message, indent=2)}"
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
        model="claude-sonnet-4-5",
        messages=fixed_messages,
        optional_params=optional_params,
        litellm_params={},
        headers={},
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
        block for block in assistant_content if isinstance(block, dict) and block.get("type") == "tool_use"
    ]

    assert len(tool_use_blocks) > 0, (
        f"After fix, assistant message should have tool_use blocks. "
        f"Found content: {json.dumps(assistant_content, indent=2)}"
    )

    # Verify the tool_use block has the correct ID
    tool_use_id = tool_use_blocks[0].get("id")
    assert tool_use_id == tool_call_id, f"Tool use ID {tool_use_id} should match tool_call_id {tool_call_id}"

    print("\n" + "=" * 80)
    print("[PASS] Fix verified: tool_calls are added when missing")
    print("=" * 80)
    print(f"  Tool use blocks: {len(tool_use_blocks)}")
    print(f"  Tool use ID: {tool_use_id}")
    print("\nThe fix ensures that when tool_results are present but tool_calls are")
    print("missing from the assistant message, they are added from cache or tools.")


if __name__ == "__main__":
    test_fix_ensures_tool_calls_for_tool_results()
