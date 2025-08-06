import json
import os
import sys
import asyncio

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path
from unittest.mock import MagicMock, patch

import litellm
from litellm import completion, RateLimitError, ModelResponse
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


def test_transform_request_helper_includes_anthropic_beta_and_tools():
    """Test _transform_request_helper includes anthropic_beta for computer tools."""
    config = AmazonConverseConfig()
    system_content_blocks = []
    optional_params = {
        "anthropic_beta": ["computer-use-2024-10-22"],
        "tools": [
            {
                "type": "computer_20241022",
                "name": "computer",
                "display_height_px": 768,
                "display_width_px": 1024,
                "display_number": 0,
            }
        ],
        "some_other_param": 123,
    }
    data = config._transform_request_helper(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        system_content_blocks=system_content_blocks,
        optional_params=optional_params,
        messages=None,
    )
    assert "additionalModelRequestFields" in data
    fields = data["additionalModelRequestFields"]
    assert "anthropic_beta" in fields
    assert fields["anthropic_beta"] == ["computer-use-2024-10-22"]
    # Verify computer tool is included
    assert "tools" in fields
    assert len(fields["tools"]) == 1
    assert fields["tools"][0]["type"] == "computer_20241022"


def test_transform_response_with_computer_use_tool():
    """Test response transformation with computer use tool call."""
    import httpx
    from litellm.types.llms.bedrock import ConverseResponseBlock, ConverseTokenUsageBlock
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    from litellm.types.utils import ModelResponse

    # Simulate a Bedrock Converse response with a computer-use tool call
    response_json = {
        "additionalModelResponseFields": {},
        "metrics": {"latencyMs": 100.0},
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_123",
                            "name": "computer",
                            "input": {
                                "display_height_px": 768,
                                "display_width_px": 1024,
                                "display_number": 0,
                            },
                        }
                    }
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {
            "inputTokens": 10,
            "outputTokens": 5,
            "totalTokens": 15,
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
        },
    }
    # Mock httpx.Response
    class MockResponse:
        def json(self):
            return response_json
        @property
        def text(self):
            return json.dumps(response_json)
    
    config = AmazonConverseConfig()
    model_response = ModelResponse()
    optional_params = {
        "tools": [
            {
                "type": "computer_20241022",
                "function": {
                    "name": "computer",
                    "parameters": {
                        "display_height_px": 768,
                        "display_width_px": 1024,
                        "display_number": 0,
                    },
                },
            }
        ]
    }
    # Call the transformation logic
    result = config._transform_response(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        response=MockResponse(),
        model_response=model_response,
        stream=False,
        logging_obj=None,
        optional_params=optional_params,
        api_key=None,
        data=None,
        messages=[],
        encoding=None,
    )
    # Check that the tool call is present in the returned message
    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) == 1
    tool_call = result.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "computer"
    args = json.loads(tool_call.function.arguments)
    assert args["display_height_px"] == 768
    assert args["display_width_px"] == 1024
    assert args["display_number"] == 0


def test_transform_response_with_bash_tool():
    """Test response transformation with bash tool call."""
    import httpx
    from litellm.types.llms.bedrock import ConverseResponseBlock, ConverseTokenUsageBlock
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    from litellm.types.utils import ModelResponse

    # Simulate a Bedrock Converse response with a bash tool call
    response_json = {
        "additionalModelResponseFields": {},
        "metrics": {"latencyMs": 100.0},
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tooluse_456",
                            "name": "bash",
                            "input": {
                                "command": "ls -la *.py"
                            },
                        }
                    }
                ]
            }
        },
        "stopReason": "tool_use",
        "usage": {
            "inputTokens": 8,
            "outputTokens": 3,
            "totalTokens": 11,
            "cacheReadInputTokenCount": 0,
            "cacheReadInputTokens": 0,
            "cacheWriteInputTokenCount": 0,
            "cacheWriteInputTokens": 0,
        },
    }
    # Mock httpx.Response
    class MockResponse:
        def json(self):
            return response_json
        @property
        def text(self):
            return json.dumps(response_json)
    
    config = AmazonConverseConfig()
    model_response = ModelResponse()
    optional_params = {
        "tools": [
            {
                "type": "bash_20241022",
                "function": {
                    "name": "bash",
                    "parameters": {},
                },
            }
        ]
    }
    # Call the transformation logic
    result = config._transform_response(
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        response=MockResponse(),
        model_response=model_response,
        stream=False,
        logging_obj=None,
        optional_params=optional_params,
        api_key=None,
        data=None,
        messages=[],
        encoding=None,
    )
    # Check that the tool call is present in the returned message
    assert result.choices[0].message.tool_calls is not None
    assert len(result.choices[0].message.tool_calls) == 1
    tool_call = result.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "bash"
    args = json.loads(tool_call.function.arguments)
    assert args["command"] == "ls -la *.py"


@pytest.mark.asyncio
async def test_bedrock_bash_tool_acompletion():
    """Test Bedrock with bash tool for ls command using acompletion."""
    
    # Test with bash tool instead of computer tool
    tools = [
        {
            "type": "bash_20241022",
            "name": "bash",
        }
    ]
    
    messages = [
        {
            "role": "user", 
            "content": "run ls command and find all python files"
        }
    ]
    
    try:
        response = await litellm.acompletion(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=messages,
            tools=tools,
            # Using dummy API key - test should fail with auth error, proving request formatting works
            api_key="dummy-key-for-testing"
        )
        # If we get here, something's wrong - we expect an auth error
        assert False, "Expected authentication error but got successful response"
    except Exception as e:
        error_str = str(e).lower()
        
        # Check if it's an expected authentication/credentials error
        auth_error_indicators = [
            "credentials", "authentication", "unauthorized", "access denied", 
            "aws", "region", "profile", "token", "invalid", "signature"
        ]
        
        if any(auth_error in error_str for auth_error in auth_error_indicators):
            # This is expected - request formatting succeeded, auth failed as expected
            assert True
        else:
            # Unexpected error - might be tool handling issue
            pytest.fail(f"Unexpected error (might be tool handling issue): {e}")


@pytest.mark.asyncio
async def test_bedrock_computer_use_acompletion():
    """Test Bedrock computer use with acompletion function."""
    
    # Test with computer use tool
    tools = [
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_height_px": 768,
            "display_width_px": 1024,
            "display_number": 0,
        }
    ]
    
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": "Go to the bedrock console"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                    }
                }
            ]
        }
    ]
    
    try:
        response = await litellm.acompletion(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=messages,
            tools=tools,
            # Using dummy API key - test should fail with auth error, proving request formatting works
            api_key="dummy-key-for-testing"
        )
        # If we get here, something's wrong - we expect an auth error
        assert False, "Expected authentication error but got successful response"
    except Exception as e:
        error_str = str(e).lower()
        
        # Check if it's an expected authentication/credentials error
        auth_error_indicators = [
            "credentials", "authentication", "unauthorized", "access denied", 
            "aws", "region", "profile", "token", "invalid", "signature"
        ]
        
        if any(auth_error in error_str for auth_error in auth_error_indicators):
            # This is expected - request formatting succeeded, auth failed as expected
            assert True
        else:
            # Unexpected error - might be tool handling issue
            pytest.fail(f"Unexpected error (might be tool handling issue): {e}")


@pytest.mark.asyncio
async def test_transformation_directly():
    """Test the transformation directly to verify the request structure."""
    
    config = AmazonConverseConfig()
    
    tools = [
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_height_px": 768,
            "display_width_px": 1024,
            "display_number": 0,
        },
        {
            "type": "bash_20241022",
            "name": "bash",
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "run ls command and find all python files"
        }
    ]
    
    # Transform request
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        optional_params={"tools": tools},
        litellm_params={},
        headers={}
    )
    
    # Verify the structure
    assert "additionalModelRequestFields" in request_data
    additional_fields = request_data["additionalModelRequestFields"]
    
    # Check that anthropic_beta is set correctly for computer use
    assert "anthropic_beta" in additional_fields
    assert additional_fields["anthropic_beta"] == ["computer-use-2024-10-22"]
    
    # Check that tools are present
    assert "tools" in additional_fields
    assert len(additional_fields["tools"]) == 2
    
    # Verify tool types
    tool_types = [tool.get("type") for tool in additional_fields["tools"]]
    assert "computer_20241022" in tool_types
    assert "bash_20241022" in tool_types


def test_transform_request_helper_includes_anthropic_beta_and_tools_bash():
    """Test _transform_request_helper includes anthropic_beta for bash tools."""
    config = AmazonConverseConfig()
    system_content_blocks = []
    optional_params = {
        "anthropic_beta": ["computer-use-2024-10-22"],
        "tools": [
            {
                "type": "bash_20241022",
                "name": "bash",
            }
        ],
        "some_other_param": 123,
    }
    data = config._transform_request_helper(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        system_content_blocks=system_content_blocks,
        optional_params=optional_params,
        messages=None,
    )
    assert "additionalModelRequestFields" in data
    fields = data["additionalModelRequestFields"]
    assert "anthropic_beta" in fields
    assert fields["anthropic_beta"] == ["computer-use-2024-10-22"]
    # Verify bash tool is included
    assert "tools" in fields
    assert len(fields["tools"]) == 1
    assert fields["tools"][0]["type"] == "bash_20241022"


def test_transform_request_with_multiple_tools():
    """Test transformation with multiple tools including computer, bash, and function tools."""
    config = AmazonConverseConfig()
    
    # Use the exact payload from the user's error
    tools = [
        {
            "type": "computer_20241022",
            "function": {
                "name": "computer",
                "parameters": {
                    "display_height_px": 768,
                    "display_width_px": 1024,
                    "display_number": 0,
                },
            },
        },
        {
            "type": "bash_20241022",
            "name": "bash",
        },
        {
            "type": "text_editor_20241022",
            "name": "str_replace_editor",
        },
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
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
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "run ls command and find all python files"
        }
    ]
    
    # Transform request
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        optional_params={"tools": tools},
        litellm_params={},
        headers={}
    )
    
    # Verify the structure
    assert "additionalModelRequestFields" in request_data
    additional_fields = request_data["additionalModelRequestFields"]
    
    # Check that anthropic_beta is set correctly for computer use
    assert "anthropic_beta" in additional_fields
    assert additional_fields["anthropic_beta"] == ["computer-use-2024-10-22"]
    
    # Check that tools are present
    assert "tools" in additional_fields
    assert len(additional_fields["tools"]) == 3  # computer, bash, text_editor tools
    
    # Verify tool types
    tool_types = [tool.get("type") for tool in additional_fields["tools"]]
    assert "computer_20241022" in tool_types
    assert "bash_20241022" in tool_types
    assert "text_editor_20241022" in tool_types
    
    # Function tools are processed separately and not included in computer use tools
    # They would be in toolConfig if present


def test_transform_request_with_computer_tool_only():
    """Test transformation with only computer tool."""
    config = AmazonConverseConfig()
    
    tools = [
        {
            "type": "computer_20241022",
            "name": "computer",
            "display_height_px": 768,
            "display_width_px": 1024,
            "display_number": 0,
        }
    ]
    
    messages = [
        {
            "role": "user", 
            "content": [
                {
                    "type": "text",
                    "text": "Go to the bedrock console"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
                    }
                }
            ]
        }
    ]
    
    # Transform request
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        optional_params={"tools": tools},
        litellm_params={},
        headers={}
    )
    
    # Verify the structure
    assert "additionalModelRequestFields" in request_data
    additional_fields = request_data["additionalModelRequestFields"]
    
    # Check that anthropic_beta is set correctly for computer use
    assert "anthropic_beta" in additional_fields
    assert additional_fields["anthropic_beta"] == ["computer-use-2024-10-22"]
    
    # Check that tools are present
    assert "tools" in additional_fields
    assert len(additional_fields["tools"]) == 1
    assert additional_fields["tools"][0]["type"] == "computer_20241022"


def test_transform_request_with_bash_tool_only():
    """Test transformation with only bash tool."""
    config = AmazonConverseConfig()
    
    tools = [
        {
            "type": "bash_20241022",
            "name": "bash",
        }
    ]
    
    messages = [
        {
            "role": "user", 
            "content": "run ls command and find all python files"
        }
    ]
    
    # Transform request
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        optional_params={"tools": tools},
        litellm_params={},
        headers={}
    )
    
    # Verify the structure
    assert "additionalModelRequestFields" in request_data
    additional_fields = request_data["additionalModelRequestFields"]
    
    # Check that anthropic_beta is set correctly for computer use
    assert "anthropic_beta" in additional_fields
    assert additional_fields["anthropic_beta"] == ["computer-use-2024-10-22"]
    
    # Check that tools are present
    assert "tools" in additional_fields
    assert len(additional_fields["tools"]) == 1
    assert additional_fields["tools"][0]["type"] == "bash_20241022"


def test_transform_request_with_text_editor_tool():
    """Test transformation with text editor tool."""
    config = AmazonConverseConfig()
    
    tools = [
        {
            "type": "text_editor_20241022",
            "name": "str_replace_editor",
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "Edit this text file"
        }
    ]
    
    # Transform request
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        optional_params={"tools": tools},
        litellm_params={},
        headers={}
    )
    
    # Verify the structure
    assert "additionalModelRequestFields" in request_data
    additional_fields = request_data["additionalModelRequestFields"]
    
    # Check that anthropic_beta is set correctly for computer use
    assert "anthropic_beta" in additional_fields
    assert additional_fields["anthropic_beta"] == ["computer-use-2024-10-22"]
    
    # Check that tools are present
    assert "tools" in additional_fields
    assert len(additional_fields["tools"]) == 1
    assert additional_fields["tools"][0]["type"] == "text_editor_20241022"


def test_transform_request_with_function_tool():
    """Test transformation with function tool."""
    config = AmazonConverseConfig()
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
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
            }
        }
    ]
    
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco?"
        }
    ]
    
    # Transform request
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        optional_params={"tools": tools},
        litellm_params={},
        headers={}
    )
    
    # Verify the structure
    assert "additionalModelRequestFields" in request_data
    additional_fields = request_data["additionalModelRequestFields"]
    
    # Function tools are not computer use tools, so they don't get anthropic_beta
    # They are processed through the regular tool config
    assert "toolConfig" in request_data
    assert "tools" in request_data["toolConfig"]
    assert len(request_data["toolConfig"]["tools"]) == 1
    assert request_data["toolConfig"]["tools"][0]["toolSpec"]["name"] == "get_weather"