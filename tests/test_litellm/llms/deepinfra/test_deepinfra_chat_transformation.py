import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add litellm to path
sys.path.insert(0, os.path.abspath("../../../.."))
import litellm


def test_deepseek_supported_openai_params():
    """
    Test "reasoning_effort" is an openai param supported for the DeepSeek model on deepinfra
    """
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig

    # Ensure we're using the local model cost map
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")

    supported_openai_params = DeepInfraConfig().get_supported_openai_params(model="deepinfra/deepseek-ai/DeepSeek-V3.1")
    print(supported_openai_params)
    assert "reasoning_effort" in supported_openai_params


def test_deepinfra_tool_message_content_transformation():
    """
    Test that DeepInfra transforms tool message content from array to string.
    
    This fixes the issue where LibreChat sends tool messages with content as an array:
    {"role": "tool", "content": [{"type": "text", "text": "20"}]}
    
    DeepInfra requires content to be a string, so we transform it to:
    {"role": "tool", "content": "20"}
    
    Related to issue #13982
    """
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig

    config = DeepInfraConfig()
    
    # Test case 1: Simple single text item in array (common case from LibreChat)
    messages_with_array_content = [
        {
            "role": "user",
            "content": "Calculate 10 + 10"
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"input": "10 + 10"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "calculator",
            "content": [{"type": "text", "text": "20"}]  # Array format from LibreChat
        }
    ]
    
    transformed_messages = config._transform_messages(
        messages=messages_with_array_content,
        model="deepinfra/Qwen/Qwen3-235B-A22B"
    )
    
    # Verify the tool message content was converted to string
    tool_message = transformed_messages[2]
    assert tool_message["role"] == "tool"
    assert isinstance(tool_message["content"], str)
    assert tool_message["content"] == "20"
    print(f"✓ Test case 1 passed: {tool_message['content']}")
    
    # Test case 2: Complex content array (multiple items)
    messages_with_complex_content = [
        {
            "role": "user",
            "content": "Test"
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_456",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_456",
            "content": [
                {"type": "text", "text": "Result 1"},
                {"type": "text", "text": "Result 2"}
            ]
        }
    ]
    
    transformed_messages_complex = config._transform_messages(
        messages=messages_with_complex_content,
        model="deepinfra/Qwen/Qwen3-235B-A22B"
    )
    
    tool_message_complex = transformed_messages_complex[2]
    assert tool_message_complex["role"] == "tool"
    assert isinstance(tool_message_complex["content"], str)
    # For complex content, it should be JSON stringified
    parsed_content = json.loads(tool_message_complex["content"])
    assert len(parsed_content) == 2
    assert parsed_content[0]["text"] == "Result 1"
    print(f"✓ Test case 2 passed: {tool_message_complex['content']}")
    
    # Test case 3: Tool message with string content (should remain unchanged)
    messages_with_string_content = [
        {
            "role": "user",
            "content": "Test"
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_789",
                    "type": "function",
                    "function": {"name": "test", "arguments": "{}"}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_789",
            "content": "Simple string result"  # Already a string
        }
    ]
    
    transformed_messages_string = config._transform_messages(
        messages=messages_with_string_content,
        model="deepinfra/Qwen/Qwen3-235B-A22B"
    )
    
    tool_message_string = transformed_messages_string[2]
    assert tool_message_string["role"] == "tool"
    assert isinstance(tool_message_string["content"], str)
    assert tool_message_string["content"] == "Simple string result"
    print(f"✓ Test case 3 passed: {tool_message_string['content']}")
    
    print("\n✅ All DeepInfra tool message transformation tests passed!")


@pytest.mark.asyncio
async def test_deepinfra_tool_message_content_transformation_async():
    """
    Test that DeepInfra transforms tool message content from array to string in async mode.
    
    This ensures the async path works correctly when is_async=True.
    
    Related to issue #13982
    """
    from litellm.llms.deepinfra.chat.transformation import DeepInfraConfig

    config = DeepInfraConfig()
    
    # Test async transformation with tool message containing array content
    messages_with_array_content = [
        {
            "role": "user",
            "content": "Calculate 10 + 10"
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": '{"input": "10 + 10"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "name": "calculator",
            "content": [{"type": "text", "text": "20"}]  # Array format from LibreChat
        }
    ]
    
    # Call with is_async=True
    transformed_messages = await config._transform_messages(
        messages=messages_with_array_content,
        model="deepinfra/Qwen/Qwen3-235B-A22B",
        is_async=True
    )
    
    # Verify the tool message content was converted to string
    tool_message = transformed_messages[2]
    assert tool_message["role"] == "tool"
    assert isinstance(tool_message["content"], str)
    assert tool_message["content"] == "20"
    print(f"✓ Async test passed: {tool_message['content']}")
    
    print("\n✅ DeepInfra async tool message transformation test passed!")
