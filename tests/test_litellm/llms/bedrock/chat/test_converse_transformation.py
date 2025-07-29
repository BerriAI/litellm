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


# Computer Use Tools Tests
def test_bedrock_computer_use_tools_single():
    """Test Bedrock with single computer use tool."""
    try:
        litellm.set_verbose = True
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
                        "text": "Take a screenshot"
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
        response: ModelResponse = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=messages,
            tools=tools,
        )
        print(f"Computer use response: {response}")
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert response.choices[0].message.tool_calls[0].function.name == "computer"
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_computer_use_tools_function_format():
    """Test Bedrock with computer use tool in function format."""
    try:
        litellm.set_verbose = True
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
            }
        ]
        messages = [
            {
                "role": "user",
                "content": "Take a screenshot and describe what you see"
            }
        ]
        response: ModelResponse = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=messages,
            tools=tools,
        )
        print(f"Computer use function format response: {response}")
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert response.choices[0].message.tool_calls[0].function.name == "computer"
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_bash_tool():
    """Test Bedrock with bash tool."""
    try:
        litellm.set_verbose = True
        tools = [
            {
                "type": "bash_20241022",
                "name": "bash",
            }
        ]
        messages = [
            {
                "role": "user",
                "content": "List all python files in the current directory"
            }
        ]
        response: ModelResponse = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=messages,
            tools=tools,
        )
        print(f"Bash tool response: {response}")
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert response.choices[0].message.tool_calls[0].function.name == "bash"
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_text_editor_tool():
    """Test Bedrock with text editor tool."""
    try:
        litellm.set_verbose = True
        tools = [
            {
                "type": "text_editor_20241022",
                "name": "str_replace_editor",
            }
        ]
        messages = [
            {
                "role": "user",
                "content": "Create a hello world Python script"
            }
        ]
        response: ModelResponse = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=messages,
            tools=tools,
        )
        print(f"Text editor tool response: {response}")
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert response.choices[0].message.tool_calls[0].function.name == "str_replace_editor"
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_mixed_computer_use_and_function_calling():
    """Test Bedrock with both computer use tools and function calling tools."""
    try:
        litellm.set_verbose = True
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
                "content": "Check the weather in Boston and then take a screenshot"
            }
        ]
        response: ModelResponse = completion(
            model="bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
            messages=messages,
            tools=tools,
        )
        print(f"Mixed tools response: {response}")
        # Should have tool calls
        assert len(response.choices[0].message.tool_calls) > 0
        # Check that we have function names
        tool_names = [tc.function.name for tc in response.choices[0].message.tool_calls]
        print(f"Tool names called: {tool_names}")
        # Should call weather function or computer tool
        assert any(name in ["get_weather", "computer"] for name in tool_names)
    except RateLimitError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_bedrock_computer_use_tools_transformation():
    """Test that computer use tools are transformed correctly for Bedrock format."""
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    config = AmazonConverseConfig()

    # Test mixed computer use and function calling tools
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

    messages = [{"role": "user", "content": "Test message"}]

    # Transform request
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20241022-v2:0",
        messages=messages,
        optional_params={"tools": tools},
        litellm_params={},
        headers={}
    )

    # Verify computer use tools are in additionalModelRequestFields
    assert "anthropic_beta" in request_data["additionalModelRequestFields"]
    assert request_data["additionalModelRequestFields"]["anthropic_beta"] == ["computer-use-2024-10-22"]
    assert "tools" in request_data["additionalModelRequestFields"]

    computer_tools = request_data["additionalModelRequestFields"]["tools"]
    assert len(computer_tools) == 3  # computer, bash, text_editor

    # Check computer tool transformation
    computer_tool = next(t for t in computer_tools if t["type"] == "computer_20241022")
    assert computer_tool["name"] == "computer"
    assert computer_tool["display_height_px"] == 768
    assert computer_tool["display_width_px"] == 1024
    assert computer_tool["display_number"] == 0

    # Check bash tool transformation
    bash_tool = next(t for t in computer_tools if t["type"] == "bash_20241022")
    assert bash_tool["name"] == "bash"

    # Check text editor tool transformation
    text_editor_tool = next(t for t in computer_tools if t["type"] == "text_editor_20241022")
    assert text_editor_tool["name"] == "str_replace_editor"

    # Verify function calling tool is in toolConfig
    assert "toolConfig" in request_data
    assert len(request_data["toolConfig"]["tools"]) == 1  # get_weather function
    function_tool = request_data["toolConfig"]["tools"][0]
    assert function_tool["toolSpec"]["name"] == "get_weather"
    assert function_tool["toolSpec"]["description"] == "Get the current weather in a given location"

    print("✓ Computer use tools transformation test passed\!")


def test_bedrock_computer_use_tools_without_name():
    """Test that computer use tools without explicit names get default names."""
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig

    config = AmazonConverseConfig()

    # Test computer use tools without explicit names
    tools = [
        {
            "type": "bash_20241022",
            # No name field
        },
        {
            "type": "text_editor_20241022",
            # No name field
        }
    ]

    transformed_tools = config._transform_computer_use_tools(tools)

    # Check that default names are assigned
    bash_tool = next(t for t in transformed_tools if t["type"] == "bash_20241022")
    assert bash_tool["name"] == "bash"

    text_editor_tool = next(t for t in transformed_tools if t["type"] == "text_editor_20241022")
    assert text_editor_tool["name"] == "str_replace_editor"

    print("✓ Computer use tools default names test passed\!")

