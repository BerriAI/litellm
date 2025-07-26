import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

import litellm
from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
from litellm.types.llms.bedrock import ConverseTokenUsageBlock


def test_transform_usage():
    usage = ConverseTokenUsageBlock(
        **{
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 1789,
            "cacheWriteInputTokens": 1789,
            "inputTokens": 3,
            "outputTokens": 401,
            "totalTokens": 2193,
        }
    )
    config = AmazonConverseConfig()
    openai_usage = config._transform_usage(usage)
    assert (
        openai_usage.prompt_tokens
        == usage["inputTokens"] + usage["cacheReadInputTokens"]
    )
    assert openai_usage.completion_tokens == usage["outputTokens"]
    assert openai_usage.total_tokens == usage["totalTokens"]
    assert (
        openai_usage.prompt_tokens_details.cached_tokens
        == usage["cacheReadInputTokens"]
    )
    assert openai_usage._cache_creation_input_tokens == usage["cacheWriteInputTokens"]
    assert openai_usage._cache_read_input_tokens == usage["cacheReadInputTokens"]


def test_transform_system_message():
    config = AmazonConverseConfig()

    # Case 1:
    # System message popped
    # User message remains
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    out_messages, system_blocks = config._transform_system_message(messages.copy())
    assert len(out_messages) == 1
    assert out_messages[0]["role"] == "user"
    assert len(system_blocks) == 1
    assert system_blocks[0]["text"] == "You are a helpful assistant."

    # Case 2: System message with list content (type text)
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "System prompt 1"},
                {"type": "text", "text": "System prompt 2"},
            ],
        },
        {"role": "user", "content": "Hi!"},
    ]
    out_messages, system_blocks = config._transform_system_message(messages.copy())
    assert len(out_messages) == 1
    assert out_messages[0]["role"] == "user"
    assert len(system_blocks) == 2
    assert system_blocks[0]["text"] == "System prompt 1"
    assert system_blocks[1]["text"] == "System prompt 2"

    # Case 3: System message with cache_control (should add cachePoint)
    messages = [
        {
            "role": "system",
            "content": "Cache this!",
            "cache_control": {"type": "ephemeral"},
        },
        {"role": "user", "content": "Hi!"},
    ]
    out_messages, system_blocks = config._transform_system_message(messages.copy())
    assert len(out_messages) == 1
    assert len(system_blocks) == 2
    assert system_blocks[0]["text"] == "Cache this!"
    assert "cachePoint" in system_blocks[1]
    assert system_blocks[1]["cachePoint"]["type"] == "default"

    # Case 3b: System message with two blocks, one with cache_control and one without
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "Cache this!",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Don't cache this!"},
            ],
        },
        {"role": "user", "content": "Hi!"},
    ]
    out_messages, system_blocks = config._transform_system_message(messages.copy())
    assert len(out_messages) == 1
    assert len(system_blocks) == 3
    assert system_blocks[0]["text"] == "Cache this!"
    assert "cachePoint" in system_blocks[1]
    assert system_blocks[1]["cachePoint"]["type"] == "default"
    assert system_blocks[2]["text"] == "Don't cache this!"

    # Case 4: Non-system messages are not affected
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi!"},
    ]
    out_messages, system_blocks = config._transform_system_message(messages.copy())
    assert len(out_messages) == 2
    assert out_messages[0]["role"] == "user"
    assert out_messages[1]["role"] == "assistant"
    assert system_blocks == []


def test_transform_thinking_blocks_with_redacted_content():
    thinking_blocks = [
        {
            "reasoningText": {
                "text": "This is a test",
                "signature": "test_signature",
            }
        },
        {
            "redactedContent": "This is a redacted content",
        },
    ]
    config = AmazonConverseConfig()
    transformed_thinking_blocks = config._transform_thinking_blocks(thinking_blocks)
    assert len(transformed_thinking_blocks) == 2
    assert transformed_thinking_blocks[0]["type"] == "thinking"
    assert transformed_thinking_blocks[1]["type"] == "redacted_thinking"


def test_apply_tool_call_transformation_if_needed():
    from litellm.types.utils import Message

    config = AmazonConverseConfig()
    tool_calls = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "arguments": "test_arguments",
            },
        },
    ]
    tool_response = {
        "type": "function",
        "name": "test_function",
        "parameters": {"test": "test"},
    }
    message = Message(
        role="user",
        content=json.dumps(tool_response),
    )
    transformed_message, _ = config.apply_tool_call_transformation_if_needed(
        message, tool_calls
    )
    assert len(transformed_message.tool_calls) == 1
    assert transformed_message.tool_calls[0].function.name == "test_function"
    assert transformed_message.tool_calls[0].function.arguments == json.dumps(
        tool_response["parameters"]
    )


def test_transform_tool_call_with_cache_control():
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Am I lost?"}]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_location",
                "description": "Get the user's location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
            "cache_control": {"type": "ephemeral"},
        },
    ]

    result = config.transform_request(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={"tools": tools},
        litellm_params={},
        headers={},
    )

    assert "toolConfig" in result
    assert "tools" in result["toolConfig"]

    assert len(result["toolConfig"]["tools"]) == 2

    function_out_msg = result["toolConfig"]["tools"][0]
    print(function_out_msg)
    assert function_out_msg["toolSpec"]["name"] == "get_location"
    assert function_out_msg["toolSpec"]["description"] == "Get the user's location"
    assert (
        function_out_msg["toolSpec"]["inputSchema"]["json"]["properties"]["location"][
            "type"
        ]
        == "string"
    )

    transformed_cache_msg = result["toolConfig"]["tools"][1]
    assert "cachePoint" in transformed_cache_msg
    assert transformed_cache_msg["cachePoint"]["type"] == "default"

def test_get_supported_openai_params():
    config = AmazonConverseConfig()
    supported_params = config.get_supported_openai_params(
        model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0"
    )
    assert "tools" in supported_params
    assert "tool_choice" in supported_params
    assert "thinking" in supported_params
    assert "reasoning_effort" in supported_params


def test_get_supported_openai_params_bedrock_converse():
    """
    Test that all documented bedrock converse models have the same set of supported openai params when using 
    `bedrock/converse/` or `bedrock/` prefix.

    Note: This test is critical for routing, if we ever remove `litellm.BEDROCK_CONVERSE_MODELS`, 
    please update this test to read `bedrock_converse` models from the model cost map.
    """
    for model in litellm.BEDROCK_CONVERSE_MODELS:
        print(f"Testing model: {model}")
        config = AmazonConverseConfig()
        supported_params_without_prefix = config.get_supported_openai_params(
            model=model
        )

        supported_params_with_prefix = config.get_supported_openai_params(
            model=f"bedrock/converse/{model}"
        )

        assert set(supported_params_without_prefix) == set(supported_params_with_prefix), f"Supported params mismatch for model: {model}. Without prefix: {supported_params_without_prefix}, With prefix: {supported_params_with_prefix}"
        print(f"✅ Passed for model: {model}")


def test_cache_control_tool_call_results():
    """Test that cache control is properly applied to tool call results"""
    
    config = AmazonConverseConfig()
    
    # Test tool call result with cache control
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant", 
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Boston"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1", 
            "content": "It's sunny in Boston",
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    result = config.transform_request(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
        },
        litellm_params={},
        headers={},
    )
    
    # Check that the tool result has both the content and cache point
    assert "messages" in result
    found_tool_result = False
    found_cache_point = False
    
    for message in result["messages"]:
        if message["role"] == "user" and len(message["content"]) > 1:
            for content_block in message["content"]:
                if content_block.get("toolResult"):
                    found_tool_result = True
                elif content_block.get("cachePoint"):
                    found_cache_point = True
                    assert content_block["cachePoint"]["type"] == "default"
    
    assert found_tool_result, "Tool result not found in transformed messages"
    assert found_cache_point, "Cache point not found after tool result"


def test_cache_control_assistant_text_content():
    """Test that cache control is properly applied to assistant text content"""
    
    config = AmazonConverseConfig()
    
    # Test assistant message with cache control (list content)
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant", 
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I help you today?",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
    
    result = config.transform_request(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    
    # Check that the assistant message has both text and cache point
    assert "messages" in result
    assistant_message = result["messages"][1]
    assert assistant_message["role"] == "assistant"
    assert len(assistant_message["content"]) == 2
    
    # First block should be text
    assert assistant_message["content"][0]["text"] == "Hello! How can I help you today?"
    
    # Second block should be cache point
    assert "cachePoint" in assistant_message["content"][1]
    assert assistant_message["content"][1]["cachePoint"]["type"] == "default"


def test_cache_control_assistant_string_content():
    """Test that cache control is properly applied to assistant string content"""
    
    config = AmazonConverseConfig()
    
    # Test assistant message with cache control (string content)
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant", 
            "content": "Hello! How can I help you today?",
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    result = config.transform_request(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    )
    
    # Check that the assistant message has both text and cache point
    assert "messages" in result
    assistant_message = result["messages"][1]
    assert assistant_message["role"] == "assistant"
    assert len(assistant_message["content"]) == 2
    
    # First block should be text
    assert assistant_message["content"][0]["text"] == "Hello! How can I help you today?"
    
    # Second block should be cache point
    assert "cachePoint" in assistant_message["content"][1]
    assert assistant_message["content"][1]["cachePoint"]["type"] == "default"


def test_cache_control_mixed_conversation():
    """Test cache control in a realistic conversation with multiple message types"""
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    
    config = AmazonConverseConfig()
    
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant.",
            "cache_control": {"type": "ephemeral"}
        },
        {"role": "user", "content": "What's the weather in Boston?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function", 
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Boston"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Temperature: 72°F, Conditions: Sunny",
            "cache_control": {"type": "ephemeral"}
        },
        {
            "role": "assistant",
            "content": "The weather in Boston is sunny with a temperature of 72°F!",
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    result = config.transform_request(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather info",
                        "parameters": {"type": "object", "properties": {}}
                    }
                }
            ]
        },
        litellm_params={},
        headers={},
    )
    
    # Check system message cache control
    assert "system" in result
    assert len(result["system"]) == 2
    assert result["system"][0]["text"] == "You are a helpful assistant."
    assert "cachePoint" in result["system"][1]
    assert result["system"][1]["cachePoint"]["type"] == "default"
    
    # Check that we have the right number of messages
    assert len(result["messages"]) == 4  # user, assistant with tool calls, tool result, final assistant
    
    # Check tool result cache control
    tool_message = result["messages"][2]  # Should contain tool results
    found_tool_result = False
    found_cache_point = False
    
    for content_block in tool_message["content"]:
        if content_block.get("toolResult"):
            found_tool_result = True
        elif content_block.get("cachePoint"):
            found_cache_point = True
    
    assert found_tool_result and found_cache_point, "Tool result cache control not properly applied"
    
    # Check final assistant message cache control
    final_assistant = result["messages"][3]
    assert final_assistant["role"] == "assistant"
    assert len(final_assistant["content"]) == 2
    assert final_assistant["content"][0]["text"] == "The weather in Boston is sunny with a temperature of 72°F!"
    assert "cachePoint" in final_assistant["content"][1]
    assert final_assistant["content"][1]["cachePoint"]["type"] == "default"
