import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

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
