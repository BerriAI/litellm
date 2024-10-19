import json
import os
import sys
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path


from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_embedding_response import (
    convert_dict_to_embedding_response,
)
from litellm.utils import EmbeddingResponse
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_image_generation_response import (
    convert_dict_to_image_generation_response,
)
from litellm.utils import ImageResponse

import litellm
import pytest
from datetime import timedelta

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_chat_completion_response import (
    convert_dict_to_chat_completion_response,
)
from litellm.types.utils import ModelResponse, Message, Choices


def test_convert_dict_to_chat_completion_response_basic():
    """Test basic conversion with all fields present."""
    response_object = {
        "id": "chatcmpl-123456",
        "object": "chat.completion",
        "created": 1728933352,
        "model": "gpt-4o-2024-08-06",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hi there! How can I assist you today?",
                    "refusal": None,
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 19,
            "completion_tokens": 10,
            "total_tokens": 29,
            "prompt_tokens_details": {"cached_tokens": 0},
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "system_fingerprint": "fp_6b68a8204b",
    }

    result = convert_dict_to_chat_completion_response(
        model_response_object=ModelResponse(),
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
        hidden_params=None,
        _response_headers=None,
        convert_tool_call_to_json_mode=False,
    )

    assert isinstance(result, ModelResponse)
    assert result.id == "chatcmpl-123456"
    assert len(result.choices) == 1
    assert isinstance(result.choices[0], Choices)

    # Model details
    assert result.model == "gpt-4o-2024-08-06"
    assert result.object == "chat.completion"
    assert result.created == 1728933352

    # Choices assertions
    choice = result.choices[0]
    assert choice.index == 0
    assert isinstance(choice.message, Message)
    assert choice.message.role == "assistant"
    assert choice.message.content == "Hi there! How can I assist you today?"
    assert choice.finish_reason == "stop"
    assert choice.logprobs is None

    # Usage assertions
    assert result.usage.prompt_tokens == 19
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 29
    assert result.usage.prompt_tokens_details == {"cached_tokens": 0}
    assert result.usage.completion_tokens_details == {"reasoning_tokens": 0}

    # Other fields
    assert result.system_fingerprint == "fp_6b68a8204b"

    # hidden params and response headers
    assert result._hidden_params is None
    assert result._response_headers is None


def test_convert_dict_to_chat_completion_response_tool_calls_invalid_json_arguments():
    """
    Critical test - this is a basic response from OpenAI API

    Test conversion with tool calls.

    """
    response_object = {
        "id": "chatcmpl-AK1uqisVA9OjUNkEuE53GJc8HPYlz",
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "logprobs": None,
                "message": {
                    "content": None,
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [
                        {
                            "id": "call_GED1Xit8lU7cNsjVM6dt2fTq",
                            "function": {
                                "arguments": '{"location":"Boston, MA","unit":"fahren',
                                "name": "get_current_weather",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
        "created": 1729337288,
        "model": "gpt-4o-2024-08-06",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": "fp_45c6de4934",
        "usage": {
            "completion_tokens": 10,
            "prompt_tokens": 92,
            "total_tokens": 102,
            "completion_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0},
            "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 0},
        },
    }
    result = convert_dict_to_chat_completion_response(
        model_response_object=ModelResponse(),
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
        hidden_params=None,
        _response_headers=None,
        convert_tool_call_to_json_mode=False,
    )

    assert isinstance(result, ModelResponse)
    assert result.id == "chatcmpl-AK1uqisVA9OjUNkEuE53GJc8HPYlz"
    assert len(result.choices) == 1
    assert result.choices[0].message.content is None
    assert len(result.choices[0].message.tool_calls) == 1
    assert (
        result.choices[0].message.tool_calls[0].function.name == "get_current_weather"
    )
    assert (
        result.choices[0].message.tool_calls[0].function.arguments
        == '{"location":"Boston, MA","unit":"fahren'
    )
    assert result.choices[0].finish_reason == "length"
    assert result.model == "gpt-4o-2024-08-06"
    assert result.created == 1729337288
    assert result.usage.completion_tokens == 10
    assert result.usage.prompt_tokens == 92
    assert result.usage.total_tokens == 102
    assert result.system_fingerprint == "fp_45c6de4934"


def test_convert_dict_to_chat_completion_response_tool_calls_valid_json_arguments():
    """
    Critical test - this is a basic response from OpenAI API

    Test conversion with tool calls.

    """
    response_object = {
        "id": "chatcmpl-AK1uqisVA9OjUNkEuE53GJc8HPYlz",
        "choices": [
            {
                "index": 0,
                "finish_reason": "length",
                "logprobs": None,
                "message": {
                    "content": None,
                    "refusal": None,
                    "role": "assistant",
                    "audio": None,
                    "function_call": None,
                    "tool_calls": [
                        {
                            "id": "call_GED1Xit8lU7cNsjVM6dt2fTq",
                            "function": {
                                "arguments": '{"location":"Boston, MA","unit":"fahrenheit"}',
                                "name": "get_current_weather",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
        "created": 1729337288,
        "model": "gpt-4o-2024-08-06",
        "object": "chat.completion",
        "service_tier": None,
        "system_fingerprint": "fp_45c6de4934",
        "usage": {
            "completion_tokens": 10,
            "prompt_tokens": 92,
            "total_tokens": 102,
            "completion_tokens_details": {"audio_tokens": None, "reasoning_tokens": 0},
            "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 0},
        },
    }
    result = convert_dict_to_chat_completion_response(
        model_response_object=ModelResponse(),
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
        hidden_params=None,
        _response_headers=None,
        convert_tool_call_to_json_mode=False,
    )

    assert isinstance(result, ModelResponse)
    assert result.id == "chatcmpl-AK1uqisVA9OjUNkEuE53GJc8HPYlz"
    assert len(result.choices) == 1
    assert result.choices[0].message.content is None
    assert len(result.choices[0].message.tool_calls) == 1
    assert (
        result.choices[0].message.tool_calls[0].function.name == "get_current_weather"
    )
    assert (
        result.choices[0].message.tool_calls[0].function.arguments
        == '{"location":"Boston, MA","unit":"fahrenheit"}'
    )
    assert result.choices[0].finish_reason == "length"
    assert result.model == "gpt-4o-2024-08-06"
    assert result.created == 1729337288
    assert result.usage.completion_tokens == 10
    assert result.usage.prompt_tokens == 92
    assert result.usage.total_tokens == 102
    assert result.system_fingerprint == "fp_45c6de4934"


def test_convert_dict_to_chat_completion_response_json_mode():
    """
    This test is verifying that when convert_tool_call_to_json_mode is True, a single tool call's arguments are correctly converted into the message content of the response.
    """
    model_response_object = ModelResponse(model="gpt-3.5-turbo")
    response_object = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [{"function": {"arguments": '{"key": "value"}'}}],
                },
                "finish_reason": None,
            }
        ],
        "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        "model": "gpt-3.5-turbo",
    }

    # Call the function
    result = convert_dict_to_chat_completion_response(
        model_response_object=model_response_object,
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
        hidden_params=None,
        _response_headers=None,
        convert_tool_call_to_json_mode=True,
    )

    # Assertions
    assert isinstance(result, ModelResponse)
    assert len(result.choices) == 1
    assert result.choices[0].message.content == '{"key": "value"}'
    assert result.choices[0].finish_reason == "stop"
    assert result.model == "gpt-3.5-turbo"
    assert result.usage.total_tokens == 10
    assert result.usage.prompt_tokens == 5
    assert result.usage.completion_tokens == 5


def test_convert_dict_to_chat_completion_response_error():
    """Test error handling for None response object."""
    with pytest.raises(Exception, match="Error in response object format"):
        convert_dict_to_chat_completion_response(
            model_response_object=None,
            response_object=None,
            stream=False,
            start_time=None,
            end_time=None,
            hidden_params=None,
            _response_headers=None,
            convert_tool_call_to_json_mode=False,
        )
