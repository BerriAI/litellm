import asyncio
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
from litellm import ModelResponse, RateLimitError, completion
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

    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    from litellm.types.llms.bedrock import (
        ConverseResponseBlock,
        ConverseTokenUsageBlock,
    )
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

    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    from litellm.types.llms.bedrock import (
        ConverseResponseBlock,
        ConverseTokenUsageBlock,
    )
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


def test_transform_response_with_structured_response_being_called():
    """Test response transformation with structured response."""
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
                            "name": "json_tool_call",
                            "input": {
                                "Current_Temperature": 62, 
                                "Weather_Explanation": "San Francisco typically has mild, cool weather year-round due to its coastal location and marine influence. The city is known for its fog, moderate temperatures, and relatively stable climate with little seasonal variation."},
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
        "json_mode": True,
        "tools": [
            {
                'type': 'function', 
                'function': {
                    'name': 'get_weather', 
                    'description': 'Get the current weather in a given location', 
                    'parameters': {
                        'type': 'object', 
                        'properties': {
                            'location': {
                                'type': 'string', 
                                'description': 'The city and state, e.g. San Francisco, CA'
                            }, 
                            'unit': {
                                'type': 'string', 
                                'enum': ['celsius', 'fahrenheit']
                            }
                        }, 
                        'required': ['location']
                    }
                }
            }, 
            {
                'type': 'function', 
                'function': {
                    'name': 'json_tool_call', 
                    'parameters': {
                        '$schema': 'http://json-schema.org/draft-07/schema#', 
                        'type': 'object', 
                        'required': ['Weather_Explanation', 'Current_Temperature'], 
                        'properties': {
                            'Weather_Explanation': {
                                'type': ['string', 'null'], 
                                'description': '1-2 sentences explaining the weather in the location'
                            }, 
                            'Current_Temperature': {
                                'type': ['number', 'null'], 
                                'description': 'Current temperature in the location'
                            }
                        }, 
                        'additionalProperties': False
                    }
                }
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
    assert result.choices[0].message.tool_calls is None

    assert result.choices[0].message.content is not None
    assert result.choices[0].message.content ==  '{"Current_Temperature": 62, "Weather_Explanation": "San Francisco typically has mild, cool weather year-round due to its coastal location and marine influence. The city is known for its fog, moderate temperatures, and relatively stable climate with little seasonal variation."}'

def test_transform_response_with_structured_response_calling_tool():
    """Test response transformation with structured response."""
    from litellm.llms.bedrock.chat.converse_transformation import AmazonConverseConfig
    from litellm.types.utils import ModelResponse

    # Simulate a Bedrock Converse response with a bash tool call
    response_json = {
        "metrics": {
            "latencyMs": 1148
        }, 
        "output": {
            "message": 
            {
                "content": [
                    {
                        "text": "I\'ll check the current weather in San Francisco for you."
                    }, 
                    {
                        "toolUse": {
                            "input": {
                                "location": "San Francisco, CA",
                                "unit": "celsius"
                            }, 
                            "name": "get_weather", 
                            "toolUseId": "tooluse_oKk__QrqSUmufMw3Q7vGaQ"
                        }
                    }
                ], 
                "role": "assistant"
            }
        }, 
        "stopReason": "tool_use", 
        "usage": {
            "cacheReadInputTokenCount": 0, 
            "cacheReadInputTokens": 0, 
            "cacheWriteInputTokenCount": 0, 
            "cacheWriteInputTokens": 0, 
            "inputTokens": 534, 
            "outputTokens": 69, 
            "totalTokens": 603
        }
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
        "json_mode": True,
        "tools": [
            {
                'type': 'function', 
                'function': {
                    'name': 'get_weather', 
                    'description': 'Get the current weather in a given location', 
                    'parameters': {
                        'type': 'object', 
                        'properties': {
                            'location': {
                                'type': 'string', 
                                'description': 'The city and state, e.g. San Francisco, CA'
                            }, 
                            'unit': {
                                'type': 'string', 
                                'enum': ['celsius', 'fahrenheit']
                            }
                        }, 
                        'required': ['location']
                    }
                }
            }, 
            {
                'type': 'function', 
                'function': {
                    'name': 'json_tool_call', 
                    'parameters': {
                        '$schema': 'http://json-schema.org/draft-07/schema#', 
                        'type': 'object', 
                        'required': ['Weather_Explanation', 'Current_Temperature'], 
                        'properties': {
                            'Weather_Explanation': {
                                'type': ['string', 'null'], 
                                'description': '1-2 sentences explaining the weather in the location'
                            }, 
                            'Current_Temperature': {
                                'type': ['number', 'null'], 
                                'description': 'Current temperature in the location'
                            }
                        }, 
                        'additionalProperties': False
                    }
                }
            }
        ]
    }
    # Call the transformation logic
    result = config._transform_response(
        model="bedrock/eu.anthropic.claude-sonnet-4-20250514-v1:0",
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
    assert result.choices[0].message.tool_calls[0].function.name == "get_weather"
    assert result.choices[0].message.tool_calls[0].function.arguments == '{"location": "San Francisco, CA", "unit": "celsius"}'


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


def test_map_openai_params_with_response_format():
    """Test map_openai_params with response_format."""
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

    json_schema = {
        "type": "json_schema",
        "json_schema": {
            "name": "WeatherResult",
            "schema": {
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["Weather_Explanation", "Current_Temperature"],
                "properties": {
                    "Weather_Explanation": {
                        "type": ["string", "null"],
                        "description": "1-2 sentences explaining the weather in the location",
                    },
                    "Current_Temperature": {
                        "type": ["number", "null"],
                        "description": "Current temperature in the location",
                    },
                },
                "additionalProperties": False,
            },
            "strict": False,
        },
    }

    optional_params = config.map_openai_params(
        non_default_params={"response_format": json_schema},
        optional_params={"tools": tools},
        model="eu.anthropic.claude-sonnet-4-20250514-v1:0",
        drop_params=False
    )

    assert "tools" in optional_params
    assert len(optional_params["tools"]) == 2
    assert optional_params["tools"][1]["type"] == "function"
    assert optional_params["tools"][1]["function"]["name"] == "json_tool_call"


@pytest.mark.asyncio
async def test_assistant_message_cache_control():
    """Test that assistant messages with cache_control generate cachePoint blocks."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        BedrockConverseMessagesProcessor,
        _bedrock_converse_messages_pt,
    )

    # Test assistant message with string content and cache_control
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant", 
            "content": "Hi there!",
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )

    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )
    
    assert result == async_result

    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )
    
    assert result == async_result
    
    # Should have user message and assistant message
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[1]["role"] == "assistant"
    
    # Assistant message should have text content and cachePoint
    assistant_content = result[1]["content"]
    assert len(assistant_content) == 2
    assert assistant_content[0]["text"] == "Hi there!"
    assert "cachePoint" in assistant_content[1]
    assert assistant_content[1]["cachePoint"]["type"] == "default"


@pytest.mark.asyncio
async def test_assistant_message_list_content_cache_control():
    """Test assistant messages with list content and cache_control."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        BedrockConverseMessagesProcessor,
        _bedrock_converse_messages_pt,
    )
    
    messages = [
        {"role": "user", "content": "Hello"},
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "This should be cached",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
    
    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )

    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )
    
    assert result == async_result
    
    # Assistant message should have text content and cachePoint
    assistant_content = result[1]["content"]
    assert len(assistant_content) == 2
    assert assistant_content[0]["text"] == "This should be cached"
    assert "cachePoint" in assistant_content[1]
    assert assistant_content[1]["cachePoint"]["type"] == "default"


@pytest.mark.asyncio
async def test_tool_message_cache_control():
    """Test that tool messages with cache_control generate cachePoint blocks."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        BedrockConverseMessagesProcessor,
        _bedrock_converse_messages_pt,
    )
    
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": [
                {
                    "type": "text",
                    "text": "Weather data: sunny, 25°C",
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
    
    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )

    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )
    
    assert result == async_result
    
    # Should have user, assistant, and user (tool results) messages
    assert len(result) == 3
    
    # Last message should contain tool result and cachePoint
    tool_message_content = result[2]["content"]
    assert len(tool_message_content) == 2
    
    # First should be tool result
    assert "toolResult" in tool_message_content[0]
    assert tool_message_content[0]["toolResult"]["content"][0]["text"] == "Weather data: sunny, 25°C"
    
    # Second should be cachePoint
    assert "cachePoint" in tool_message_content[1]
    assert tool_message_content[1]["cachePoint"]["type"] == "default"


@pytest.mark.asyncio
async def test_tool_message_string_content_cache_control():
    """Test tool messages with string content and message-level cache_control."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        BedrockConverseMessagesProcessor,
        _bedrock_converse_messages_pt,
    )
    
    messages = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function", 
                    "function": {"name": "get_weather", "arguments": "{}"}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Weather: sunny, 25°C",
            "cache_control": {"type": "ephemeral"}
        }
    ]
    
    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )

    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )
    
    assert result == async_result
    
    # Last message should contain tool result and cachePoint
    tool_message_content = result[2]["content"]
    assert len(tool_message_content) == 2
    
    # First should be tool result
    assert "toolResult" in tool_message_content[0]
    assert tool_message_content[0]["toolResult"]["content"][0]["text"] == "Weather: sunny, 25°C"
    
    # Second should be cachePoint
    assert "cachePoint" in tool_message_content[1]
    assert tool_message_content[1]["cachePoint"]["type"] == "default"


@pytest.mark.asyncio
async def test_assistant_tool_calls_cache_control():
    """Test that assistant tool_calls with cache_control generate cachePoint blocks."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        BedrockConverseMessagesProcessor,
        _bedrock_converse_messages_pt,
    )
    
    messages = [
        {"role": "user", "content": "Calculate 2+2"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_proxy_123",
                    "type": "function",
                    "function": {"name": "calc", "arguments": "{}"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        }
    ]
    
    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )

    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )
    
    assert result == async_result
    
    # Assistant message should have tool use and cachePoint
    assistant_content = result[1]["content"]
    assert len(assistant_content) == 2
    
    # First should be tool use
    assert "toolUse" in assistant_content[0]
    assert assistant_content[0]["toolUse"]["name"] == "calc"
    assert assistant_content[0]["toolUse"]["toolUseId"] == "call_proxy_123"
    
    # Second should be cachePoint
    assert "cachePoint" in assistant_content[1]
    assert assistant_content[1]["cachePoint"]["type"] == "default"


@pytest.mark.asyncio
async def test_multiple_tool_calls_with_mixed_cache_control():
    """Test multiple tool calls where only some have cache_control."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        BedrockConverseMessagesProcessor,
        _bedrock_converse_messages_pt,
    )
    
    messages = [
        {"role": "user", "content": "Do multiple calculations"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"expr": "2+2"}'},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {"name": "calc", "arguments": '{"expr": "3+3"}'}
                    # No cache_control
                }
            ]
        }
    ]
    
    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )

    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )
    
    assert result == async_result
    
    # Assistant message should have: toolUse1, cachePoint, toolUse2
    assistant_content = result[1]["content"]
    assert len(assistant_content) == 3
    
    # First tool use with cache
    assert "toolUse" in assistant_content[0]
    assert assistant_content[0]["toolUse"]["toolUseId"] == "call_1"
    
    # Cache point for first tool
    assert "cachePoint" in assistant_content[1]
    assert assistant_content[1]["cachePoint"]["type"] == "default"
    
    # Second tool use without cache
    assert "toolUse" in assistant_content[2]
    assert assistant_content[2]["toolUse"]["toolUseId"] == "call_2"


@pytest.mark.asyncio
async def test_no_cache_control_no_cache_point():
    """Test that messages without cache_control don't generate cachePoint blocks."""
    from litellm.litellm_core_utils.prompt_templates.factory import (
        BedrockConverseMessagesProcessor,
        _bedrock_converse_messages_pt,
    )
    
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},  # No cache_control
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "Tool result"  # No cache_control
        }
    ]
    
    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )

    async_result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
        llm_provider="bedrock_converse"
    )
    
    assert result == async_result
    
    # Assistant message should only have text content, no cachePoint
    assistant_content = result[1]["content"]
    assert len(assistant_content) == 1
    assert assistant_content[0]["text"] == "Hi there!"
    
    # Tool message should only have tool result, no cachePoint
    tool_content = result[2]["content"]
    assert len(tool_content) == 1
    assert "toolResult" in tool_content[0]


# ============================================================================
# Guarded Text Feature Tests
# ============================================================================

def test_guarded_text_wraps_in_guardrail_converse_content():
    """Test that guarded_text content type gets wrapped in guardContent blocks."""
    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_converse_messages_pt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Regular text content"},
                {"type": "guarded_text", "text": "This should be guarded"},
                {"type": "text", "text": "More regular text"}
            ]
        }
    ]

    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="us.amazon.nova-pro-v1:0",
        llm_provider="bedrock_converse"
    )

    # Should have 1 message
    assert len(result) == 1
    assert result[0]["role"] == "user"

    # Should have 3 content blocks
    content = result[0]["content"]
    assert len(content) == 3

    # First and third should be regular text
    assert "text" in content[0]
    assert content[0]["text"] == "Regular text content"
    assert "text" in content[2]
    assert content[2]["text"] == "More regular text"
    # Second should be guardContent
    assert "guardContent" in content[1]
    assert content[1]["guardContent"]["text"]["text"] == "This should be guarded"

def test_guarded_text_with_system_messages():
    """Test guarded_text with system messages using the full transformation."""
    config = AmazonConverseConfig()

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What is the main topic of this legal document?"},
                {"type": "guarded_text", "text": "This is a set of very long instructions that you will follow. Here is a legal document that you will use to answer the user's question."}
            ]
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "DRAFT"
        }
    }

    result = config._transform_request(
        model="us.amazon.nova-pro-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )

    # Should have system content blocks
    assert "system" in result
    assert len(result["system"]) == 1
    assert result["system"][0]["text"] == "You are a helpful assistant."

    # Should have 1 message (system messages are removed)
    assert "messages" in result
    assert len(result["messages"]) == 1

    # User message should have both regular text and guarded text
    user_message = result["messages"][0]
    assert user_message["role"] == "user"
    content = user_message["content"]
    assert len(content) == 2

    # First should be regular text
    assert "text" in content[0]
    assert content[0]["text"] == "What is the main topic of this legal document?"
    # Second should be guardContent
    assert "guardContent" in content[1]
    assert content[1]["guardContent"]["text"]["text"] == "This is a set of very long instructions that you will follow. Here is a legal document that you will use to answer the user's question."


def test_guarded_text_with_mixed_content_types():
    """Test guarded_text with mixed content types including images."""
    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_converse_messages_pt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this image"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}},
                {"type": "guarded_text", "text": "This sensitive content should be guarded"}
            ]
        }
    ]

    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="us.amazon.nova-pro-v1:0",
        llm_provider="bedrock_converse"
    )

    # Should have 1 message
    assert len(result) == 1
    assert result[0]["role"] == "user"

    # Should have 3 content blocks
    content = result[0]["content"]
    assert len(content) == 3

    # First should be regular text
    assert "text" in content[0]
    assert content[0]["text"] == "Look at this image"

    # Second should be image
    assert "image" in content[1]

    # Third should be guardContent
    assert "guardContent" in content[2]
    assert content[2]["guardContent"]["text"]["text"] == "This sensitive content should be guarded"

@pytest.mark.asyncio
async def test_async_guarded_text():
    """Test async version of guarded_text processing."""
    from litellm.litellm_core_utils.prompt_templates.factory import BedrockConverseMessagesProcessor

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "guarded_text", "text": "This should be guarded"}
            ]
        }
    ]

    result = await BedrockConverseMessagesProcessor._bedrock_converse_messages_pt_async(
        messages=messages,
        model="us.amazon.nova-pro-v1:0",
        llm_provider="bedrock_converse"
    )

    # Should have 1 message
    assert len(result) == 1
    assert result[0]["role"] == "user"

    # Should have 2 content blocks
    content = result[0]["content"]
    assert len(content) == 2

    # First should be regular text
    assert "text" in content[0]
    assert content[0]["text"] == "Hello"

    # Second should be guardContent
    assert "guardContent" in content[1]
    assert content[1]["guardContent"]["text"]["text"] == "This should be guarded"


def test_guarded_text_with_tool_calls():
    """Test guarded_text with tool calls in the conversation."""
    from litellm.litellm_core_utils.prompt_templates.factory import _bedrock_converse_messages_pt

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's the weather?"},
                {"type": "guarded_text", "text": "Please be careful with sensitive information"}
            ]
        },
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": "{}"}
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_123",
            "content": "It's sunny and 25°C"
        }
    ]

    result = _bedrock_converse_messages_pt(
        messages=messages,
        model="us.amazon.nova-pro-v1:0",
        llm_provider="bedrock_converse"
    )

    # Should have 3 messages
    assert len(result) == 3

    # First message (user) should have both text and guarded_text
    user_message = result[0]
    assert user_message["role"] == "user"
    content = user_message["content"]
    assert len(content) == 2

    # First should be regular text
    assert "text" in content[0]
    assert content[0]["text"] == "What's the weather?"
    
    # Second should be guardContent
    assert "guardContent" in content[1]
    assert content[1]["guardContent"]["text"]["text"] == "Please be careful with sensitive information"
    
    # Other messages should not have guardContent
    for i in range(1, 3):
        content = result[i]["content"]
        for block in content:
            assert "guardContent" not in block


def test_guarded_text_guardrail_config_preserved():
    """Test that guardrailConfig is preserved when using guarded_text."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "guarded_text", "text": "This should be guarded"}
            ]
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "DRAFT"
        }
    }

    result = config._transform_request(
        model="us.amazon.nova-pro-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )

    # GuardrailConfig should be present at top level
    assert "guardrailConfig" in result
    assert result["guardrailConfig"]["guardrailIdentifier"] == "gr-abc123"

    # GuardrailConfig should also be in inferenceConfig
    assert "inferenceConfig" in result
    assert "guardrailConfig" in result["inferenceConfig"]
    assert result["inferenceConfig"]["guardrailConfig"]["guardrailIdentifier"] == "gr-abc123"


def test_auto_convert_last_user_message_to_guarded_text():
    """Test that last user message is automatically converted to guarded_text when guardrailConfig is present."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the main topic of this legal document?"
                }
            ]
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1"
        }
    }

    # Test the helper method directly
    converted_messages = config._convert_consecutive_user_messages_to_guarded_text(messages, optional_params)

    # Verify the conversion
    assert len(converted_messages) == 1
    assert converted_messages[0]["role"] == "user"
    assert len(converted_messages[0]["content"]) == 1
    assert converted_messages[0]["content"][0]["type"] == "guarded_text"
    assert converted_messages[0]["content"][0]["text"] == "What is the main topic of this legal document?"


def test_auto_convert_last_user_message_string_content():
    """Test that last user message with string content is automatically converted to guarded_text when guardrailConfig is present."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": "What is the main topic of this legal document?"
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1"
        }
    }

    # Test the helper method directly
    converted_messages = config._convert_consecutive_user_messages_to_guarded_text(messages, optional_params)

    # Verify the conversion
    assert len(converted_messages) == 1
    assert converted_messages[0]["role"] == "user"
    assert len(converted_messages[0]["content"]) == 1
    assert converted_messages[0]["content"][0]["type"] == "guarded_text"
    assert converted_messages[0]["content"][0]["text"] == "What is the main topic of this legal document?"


def test_no_conversion_when_no_guardrail_config():
    """Test that no conversion happens when guardrailConfig is not present."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the main topic of this legal document?"
                }
            ]
        }
    ]

    optional_params = {}

    # Test the helper method directly
    converted_messages = config._convert_consecutive_user_messages_to_guarded_text(messages, optional_params)

    # Verify no conversion happened
    assert converted_messages == messages


def test_no_conversion_when_guarded_text_already_present():
    """Test that no conversion happens when guarded_text is already present in the last user message."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "guarded_text",
                    "text": "This is already guarded"
                }
            ]
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1"
        }
    }

    # Test the helper method directly
    converted_messages = config._convert_consecutive_user_messages_to_guarded_text(messages, optional_params)

    # Verify no conversion happened
    assert converted_messages == messages


def test_auto_convert_with_mixed_content():
    """Test that only text elements are converted to guarded_text, other content types are preserved."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the main topic of this legal document?"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/image.jpg"}
                }
            ]
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1"
        }
    }

    # Test the helper method directly
    converted_messages = config._convert_consecutive_user_messages_to_guarded_text(messages, optional_params)

    # Verify the conversion
    assert len(converted_messages) == 1
    assert converted_messages[0]["role"] == "user"
    assert len(converted_messages[0]["content"]) == 2

    # First element should be converted to guarded_text
    assert converted_messages[0]["content"][0]["type"] == "guarded_text"
    assert converted_messages[0]["content"][0]["text"] == "What is the main topic of this legal document?"

    # Second element should remain unchanged
    assert converted_messages[0]["content"][1]["type"] == "image_url"
    assert converted_messages[0]["content"][1]["image_url"]["url"] == "https://example.com/image.jpg"


def test_auto_convert_in_full_transformation():
    """Test that the automatic conversion works in the full transformation pipeline."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What is the main topic of this legal document?"
                }
            ]
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1"
        }
    }

    # Test the full transformation
    result = config._transform_request(
        model="anthropic.claude-3-sonnet-20240229-v1:0",
        messages=messages,
        optional_params=optional_params,
        litellm_params={},
        headers={}
    )

    # Verify the transformation worked
    assert "messages" in result
    assert len(result["messages"]) == 1
    
    # The message should have guardContent
    message = result["messages"][0]
    assert "content" in message
    assert len(message["content"]) == 1
    assert "guardContent" in message["content"][0]
    assert message["content"][0]["guardContent"]["text"]["text"] == "What is the main topic of this legal document?"


def test_convert_consecutive_user_messages_to_guarded_text():
    """Test that consecutive user messages at the end are converted to guarded_text."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "First user message"
                }
            ]
        },
        {
            "role": "assistant",
            "content": "Assistant response"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Second user message"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Third user message"
                }
            ]
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1"
        }
    }

    # Test the helper method directly
    converted_messages = config._convert_consecutive_user_messages_to_guarded_text(messages, optional_params)

    # Verify the conversion - only the last two user messages should be converted
    assert len(converted_messages) == 4

    # First user message should remain unchanged
    assert converted_messages[0]["role"] == "user"
    assert converted_messages[0]["content"][0]["type"] == "text"
    assert converted_messages[0]["content"][0]["text"] == "First user message"

    # Assistant message should remain unchanged
    assert converted_messages[1]["role"] == "assistant"
    assert converted_messages[1]["content"] == "Assistant response"

    # Second user message should be converted to guarded_text
    assert converted_messages[2]["role"] == "user"
    assert converted_messages[2]["content"][0]["type"] == "guarded_text"
    assert converted_messages[2]["content"][0]["text"] == "Second user message"

    # Third user message should be converted to guarded_text
    assert converted_messages[3]["role"] == "user"
    assert converted_messages[3]["content"][0]["type"] == "guarded_text"
    assert converted_messages[3]["content"][0]["text"] == "Third user message"


def test_convert_all_user_messages_when_all_consecutive():
    """Test that all user messages are converted when they are all consecutive at the end."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "First user message"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Second user message"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Third user message"
                }
            ]
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1"
        }
    }

    # Test the helper method directly
    converted_messages = config._convert_consecutive_user_messages_to_guarded_text(messages, optional_params)

    # Verify all three user messages are converted
    assert len(converted_messages) == 3

    for i in range(3):
        assert converted_messages[i]["role"] == "user"
        assert converted_messages[i]["content"][0]["type"] == "guarded_text"

    assert converted_messages[0]["content"][0]["text"] == "First user message"
    assert converted_messages[1]["content"][0]["text"] == "Second user message"
    assert converted_messages[2]["content"][0]["text"] == "Third user message"


def test_convert_consecutive_user_messages_with_string_content():
    """Test that consecutive user messages with string content are converted to guarded_text."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "assistant",
            "content": "Assistant response"
        },
        {
            "role": "user",
            "content": "First user message"
        },
        {
            "role": "user",
            "content": "Second user message"
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1"
        }
    }

    # Test the helper method directly
    converted_messages = config._convert_consecutive_user_messages_to_guarded_text(messages, optional_params)

    # Verify the conversion
    assert len(converted_messages) == 3

    # Assistant message should remain unchanged
    assert converted_messages[0]["role"] == "assistant"
    assert converted_messages[0]["content"] == "Assistant response"

    # Both user messages should be converted to guarded_text
    assert converted_messages[1]["role"] == "user"
    assert len(converted_messages[1]["content"]) == 1
    assert converted_messages[1]["content"][0]["type"] == "guarded_text"
    assert converted_messages[1]["content"][0]["text"] == "First user message"

    assert converted_messages[2]["role"] == "user"
    assert len(converted_messages[2]["content"]) == 1
    assert converted_messages[2]["content"][0]["type"] == "guarded_text"
    assert converted_messages[2]["content"][0]["text"] == "Second user message"


def test_skip_consecutive_user_messages_with_existing_guarded_text():
    """Test that consecutive user messages with existing guarded_text are skipped."""
    config = AmazonConverseConfig()

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "guarded_text",
                    "text": "Already guarded"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Should be converted"
                }
            ]
        }
    ]

    optional_params = {
        "guardrailConfig": {
            "guardrailIdentifier": "gr-abc123",
            "guardrailVersion": "1"
        }
    }

    # Test the helper method directly
    converted_messages = config._convert_consecutive_user_messages_to_guarded_text(messages, optional_params)

    # Verify the conversion
    assert len(converted_messages) == 2

    # First message should remain unchanged (already has guarded_text)
    assert converted_messages[0]["role"] == "user"
    assert converted_messages[0]["content"][0]["type"] == "guarded_text"
    assert converted_messages[0]["content"][0]["text"] == "Already guarded"

    # Second message should be converted
    assert converted_messages[1]["role"] == "user"
    assert converted_messages[1]["content"][0]["type"] == "guarded_text"
    assert converted_messages[1]["content"][0]["text"] == "Should be converted"


def test_request_metadata_parameter_support():
    """Test that requestMetadata is in supported parameters."""
    config = AmazonConverseConfig()
    supported_params = config.get_supported_openai_params(
        model="bedrock/converse/us.anthropic.claude-sonnet-4-20250514-v1:0"
    )
    assert "requestMetadata" in supported_params


def test_request_metadata_transformation():
    """Test that requestMetadata is properly transformed to top-level field."""
    config = AmazonConverseConfig()

    request_metadata = {
        "cost_center": "engineering",
        "user_id": "user123",
        "session_id": "sess_abc123"
    }

    messages = [
        {"role": "user", "content": "Hello!"},
    ]

    # Transform request with requestMetadata
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={"requestMetadata": request_metadata},
        litellm_params={},
        headers={}
    )

    # Verify that requestMetadata appears as top-level field
    assert "requestMetadata" in request_data
    assert request_data["requestMetadata"] == request_metadata


def test_request_metadata_validation():
    """Test validation of requestMetadata constraints."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]

    # Test valid metadata
    valid_metadata = {
        "cost_center": "engineering",
        "user_id": "user123",
    }

    # Should not raise exception
    config.transform_request(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={"requestMetadata": valid_metadata},
        litellm_params={},
        headers={}
    )

    # Test too many items (max 16)
    too_many_items = {f"key_{i}": f"value_{i}" for i in range(17)}

    try:
        config.transform_request(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=messages,
            optional_params={"requestMetadata": too_many_items},
            litellm_params={},
            headers={}
        )
        assert False, "Should have raised validation error for too many items"
    except Exception as e:
        assert "maximum of 16 items" in str(e).lower()


def test_request_metadata_key_constraints():
    """Test key constraint validation."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]

    # Test key too long (max 256 characters)
    long_key = "a" * 257
    invalid_metadata = {long_key: "value"}

    try:
        config.transform_request(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=messages,
            optional_params={"requestMetadata": invalid_metadata},
            litellm_params={},
            headers={}
        )
        assert False, "Should have raised validation error for key too long"
    except Exception as e:
        assert "key length" in str(e).lower() or "256 characters" in str(e).lower()

    # Test empty key
    invalid_metadata = {"": "value"}

    try:
        config.transform_request(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=messages,
            optional_params={"requestMetadata": invalid_metadata},
            litellm_params={},
            headers={}
        )
        assert False, "Should have raised validation error for empty key"
    except Exception as e:
        assert "key length" in str(e).lower() or "empty" in str(e).lower()


def test_request_metadata_value_constraints():
    """Test value constraint validation."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]

    # Test value too long (max 256 characters)
    long_value = "a" * 257
    invalid_metadata = {"key": long_value}

    try:
        config.transform_request(
            model="anthropic.claude-3-5-sonnet-20240620-v1:0",
            messages=messages,
            optional_params={"requestMetadata": invalid_metadata},
            litellm_params={},
            headers={}
        )
        assert False, "Should have raised validation error for value too long"
    except Exception as e:
        assert "value length" in str(e).lower() or "256 characters" in str(e).lower()

    # Test empty value (should be allowed)
    valid_metadata = {"key": ""}

    # Should not raise exception
    config.transform_request(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={"requestMetadata": valid_metadata},
        litellm_params={},
        headers={}
    )


def test_request_metadata_character_pattern():
    """Test character pattern validation for keys and values."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]

    # Test valid characters
    valid_metadata = {
        "cost-center_2024": "engineering@team#1",
        "user:id": "$100.00",
        "session+token": "/path/to=resource",
    }

    # Should not raise exception
    config.transform_request(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={"requestMetadata": valid_metadata},
        litellm_params={},
        headers={}
    )


def test_request_metadata_with_other_params():
    """Test that requestMetadata works alongside other parameters."""
    config = AmazonConverseConfig()

    request_metadata = {
        "experiment": "test_A",
        "user_type": "premium"
    }

    messages = [
        {"role": "user", "content": "What's the weather?"},
    ]

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    # Transform request with multiple parameters including request_metadata
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={
            "requestMetadata": request_metadata,
            "tools": tools,
            "max_tokens": 100,
            "temperature": 0.7
        },
        litellm_params={},
        headers={}
    )

    # Verify requestMetadata is at top level
    assert "requestMetadata" in request_data
    assert request_data["requestMetadata"] == request_metadata

    # Verify other parameters are also processed correctly
    assert "toolConfig" in request_data
    assert "inferenceConfig" in request_data
    assert request_data["inferenceConfig"]["temperature"] == 0.7


def test_request_metadata_empty():
    """Test handling of empty requestMetadata."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]

    # Empty dict should be allowed
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={"requestMetadata": {}},
        litellm_params={},
        headers={}
    )

    assert "requestMetadata" in request_data
    assert request_data["requestMetadata"] == {}


def test_request_metadata_not_provided():
    """Test that requestMetadata is not included when not provided."""
    config = AmazonConverseConfig()

    messages = [{"role": "user", "content": "Hello!"}]

    # No requestMetadata provided
    request_data = config.transform_request(
        model="anthropic.claude-3-5-sonnet-20240620-v1:0",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={}
    )

    # requestMetadata should not be in the request
    assert "requestMetadata" not in request_data
