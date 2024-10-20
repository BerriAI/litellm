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
    _process_choices_in_response,
)
from litellm.types.utils import (
    ModelResponse,
    Message,
    Choices,
    PromptTokensDetails,
    CompletionTokensDetails,
    ChatCompletionMessageToolCall,
)


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
    print("choice[0]", choice)
    assert choice.index == 0
    assert isinstance(choice.message, Message)
    assert choice.message.role == "assistant"
    assert choice.message.content == "Hi there! How can I assist you today?"
    assert choice.finish_reason == "stop"

    # Usage assertions
    assert result.usage.prompt_tokens == 19
    assert result.usage.completion_tokens == 10
    assert result.usage.total_tokens == 29
    assert result.usage.prompt_tokens_details == PromptTokensDetails(cached_tokens=0)
    assert result.usage.completion_tokens_details == CompletionTokensDetails(
        reasoning_tokens=0
    )

    # Other fields
    assert result.system_fingerprint == "fp_6b68a8204b"

    # hidden params
    assert result._hidden_params is not None


def test_convert_image_input_dict_response_to_chat_completion_response():
    """Test conversion on a response with an image input."""
    response_object = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_44709d6fcb",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nThis image shows a wooden boardwalk extending through a lush green marshland.",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 12,
            "total_tokens": 21,
            "completion_tokens_details": {"reasoning_tokens": 0},
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
    assert result.id == "chatcmpl-123"
    assert result.object == "chat.completion"
    assert result.created == 1677652288
    assert result.model == "gpt-4o-mini"
    assert result.system_fingerprint == "fp_44709d6fcb"

    assert len(result.choices) == 1
    choice = result.choices[0]
    assert choice.index == 0
    assert isinstance(choice.message, Message)
    assert choice.message.role == "assistant"
    assert (
        choice.message.content
        == "\n\nThis image shows a wooden boardwalk extending through a lush green marshland."
    )
    assert choice.finish_reason == "stop"

    assert result.usage.prompt_tokens == 9
    assert result.usage.completion_tokens == 12
    assert result.usage.total_tokens == 21
    assert result.usage.completion_tokens_details == CompletionTokensDetails(
        reasoning_tokens=0
    )

    assert result._hidden_params is not None


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


def test_convert_dict_to_chat_completion_response_function_output():
    """
    Test conversion with function output.

    From here: https://platform.openai.com/docs/api-reference/chat/create

    """
    response_object = {
        "id": "chatcmpl-abc123",
        "object": "chat.completion",
        "created": 1699896916,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_current_weather",
                                "arguments": '{\n"location": "Boston, MA"\n}',
                            },
                        }
                    ],
                },
                "logprobs": None,
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {
            "prompt_tokens": 82,
            "completion_tokens": 17,
            "total_tokens": 99,
            "completion_tokens_details": {"reasoning_tokens": 0},
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
    assert result.id == "chatcmpl-abc123"
    assert result.object == "chat.completion"
    assert result.created == 1699896916
    assert result.model == "gpt-4o-mini"

    assert len(result.choices) == 1
    choice = result.choices[0]
    assert choice.index == 0
    assert isinstance(choice.message, Message)
    assert choice.message.role == "assistant"
    assert choice.message.content is None
    assert choice.finish_reason == "tool_calls"

    assert len(choice.message.tool_calls) == 1
    tool_call = choice.message.tool_calls[0]
    assert tool_call.id == "call_abc123"
    assert tool_call.type == "function"
    assert tool_call.function.name == "get_current_weather"
    assert tool_call.function.arguments == '{\n"location": "Boston, MA"\n}'

    assert result.usage.prompt_tokens == 82
    assert result.usage.completion_tokens == 17
    assert result.usage.total_tokens == 99
    assert result.usage.completion_tokens_details == CompletionTokensDetails(
        reasoning_tokens=0
    )

    assert result._hidden_params is not None


def test_convert_dict_to_chat_completion_response_with_logprobs():
    """

    Test conversion with logprobs in the response.

    From here: https://platform.openai.com/docs/api-reference/chat/create

    """
    response_object = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1702685778,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I assist you today?",
                },
                "logprobs": {
                    "content": [
                        {
                            "token": "Hello",
                            "logprob": -0.31725305,
                            "bytes": [72, 101, 108, 108, 111],
                            "top_logprobs": [
                                {
                                    "token": "Hello",
                                    "logprob": -0.31725305,
                                    "bytes": [72, 101, 108, 108, 111],
                                },
                                {
                                    "token": "Hi",
                                    "logprob": -1.3190403,
                                    "bytes": [72, 105],
                                },
                            ],
                        },
                        {
                            "token": "!",
                            "logprob": -0.02380986,
                            "bytes": [33],
                            "top_logprobs": [
                                {"token": "!", "logprob": -0.02380986, "bytes": [33]},
                                {
                                    "token": " there",
                                    "logprob": -3.787621,
                                    "bytes": [32, 116, 104, 101, 114, 101],
                                },
                            ],
                        },
                        {
                            "token": " How",
                            "logprob": -0.000054669687,
                            "bytes": [32, 72, 111, 119],
                            "top_logprobs": [
                                {
                                    "token": " How",
                                    "logprob": -0.000054669687,
                                    "bytes": [32, 72, 111, 119],
                                },
                                {
                                    "token": "<|end|>",
                                    "logprob": -10.953937,
                                    "bytes": None,
                                },
                            ],
                        },
                        {
                            "token": " can",
                            "logprob": -0.015801601,
                            "bytes": [32, 99, 97, 110],
                            "top_logprobs": [
                                {
                                    "token": " can",
                                    "logprob": -0.015801601,
                                    "bytes": [32, 99, 97, 110],
                                },
                                {
                                    "token": " may",
                                    "logprob": -4.161023,
                                    "bytes": [32, 109, 97, 121],
                                },
                            ],
                        },
                        {
                            "token": " I",
                            "logprob": -3.7697225e-6,
                            "bytes": [32, 73],
                            "top_logprobs": [
                                {
                                    "token": " I",
                                    "logprob": -3.7697225e-6,
                                    "bytes": [32, 73],
                                },
                                {
                                    "token": " assist",
                                    "logprob": -13.596657,
                                    "bytes": [32, 97, 115, 115, 105, 115, 116],
                                },
                            ],
                        },
                        {
                            "token": " assist",
                            "logprob": -0.04571125,
                            "bytes": [32, 97, 115, 115, 105, 115, 116],
                            "top_logprobs": [
                                {
                                    "token": " assist",
                                    "logprob": -0.04571125,
                                    "bytes": [32, 97, 115, 115, 105, 115, 116],
                                },
                                {
                                    "token": " help",
                                    "logprob": -3.1089056,
                                    "bytes": [32, 104, 101, 108, 112],
                                },
                            ],
                        },
                        {
                            "token": " you",
                            "logprob": -5.4385737e-6,
                            "bytes": [32, 121, 111, 117],
                            "top_logprobs": [
                                {
                                    "token": " you",
                                    "logprob": -5.4385737e-6,
                                    "bytes": [32, 121, 111, 117],
                                },
                                {
                                    "token": " today",
                                    "logprob": -12.807695,
                                    "bytes": [32, 116, 111, 100, 97, 121],
                                },
                            ],
                        },
                        {
                            "token": " today",
                            "logprob": -0.0040071653,
                            "bytes": [32, 116, 111, 100, 97, 121],
                            "top_logprobs": [
                                {
                                    "token": " today",
                                    "logprob": -0.0040071653,
                                    "bytes": [32, 116, 111, 100, 97, 121],
                                },
                                {"token": "?", "logprob": -5.5247097, "bytes": [63]},
                            ],
                        },
                        {
                            "token": "?",
                            "logprob": -0.0008108172,
                            "bytes": [63],
                            "top_logprobs": [
                                {"token": "?", "logprob": -0.0008108172, "bytes": [63]},
                                {
                                    "token": "?\n",
                                    "logprob": -7.184561,
                                    "bytes": [63, 10],
                                },
                            ],
                        },
                    ]
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 9,
            "completion_tokens": 9,
            "total_tokens": 18,
            "completion_tokens_details": {"reasoning_tokens": 0},
        },
        "system_fingerprint": None,
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
    assert result.id == "chatcmpl-123"
    assert result.object == "chat.completion"
    assert result.created == 1702685778
    assert result.model == "gpt-4o-mini"

    assert len(result.choices) == 1
    choice = result.choices[0]
    assert choice.index == 0
    assert isinstance(choice.message, Message)
    assert choice.message.role == "assistant"
    assert choice.message.content == "Hello! How can I assist you today?"
    assert choice.finish_reason == "stop"

    # Check logprobs
    assert choice.logprobs is not None
    assert len(choice.logprobs["content"]) == 9

    # Check each logprob entry
    expected_tokens = [
        "Hello",
        "!",
        " How",
        " can",
        " I",
        " assist",
        " you",
        " today",
        "?",
    ]
    for i, logprob in enumerate(choice.logprobs["content"]):
        assert logprob["token"] == expected_tokens[i]
        assert isinstance(logprob["logprob"], float)
        assert isinstance(logprob["bytes"], list)
        assert len(logprob["top_logprobs"]) == 2
        assert isinstance(logprob["top_logprobs"][0]["token"], str)
        assert isinstance(logprob["top_logprobs"][0]["logprob"], float)
        assert isinstance(logprob["top_logprobs"][0]["bytes"], (list, type(None)))

    assert result.usage.prompt_tokens == 9
    assert result.usage.completion_tokens == 9
    assert result.usage.total_tokens == 18
    assert result.usage.completion_tokens_details == CompletionTokensDetails(
        reasoning_tokens=0
    )

    assert result.system_fingerprint is None
    assert result._hidden_params is not None


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


def test_process_choices_in_response():
    # Test case 1: Basic response without tool calls
    response_object = {
        "choices": [
            {
                "message": {
                    "content": "Hello, how can I help you?",
                    "role": "assistant",
                },
                "finish_reason": "stop",
            }
        ]
    }
    result = _process_choices_in_response(response_object, False)
    assert len(result) == 1
    assert isinstance(result[0], Choices)
    assert result[0].message.content == "Hello, how can I help you?"
    assert result[0].message.role == "assistant"
    assert result[0].finish_reason == "stop"

    # Test case 2: Response with tool calls
    response_object = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "New York"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ]
    }
    result = _process_choices_in_response(response_object, False)
    assert len(result) == 1
    assert isinstance(result[0], Choices)
    assert result[0].message.content is None
    assert result[0].message.role == "assistant"
    assert result[0].message.tool_calls is not None
    assert len(result[0].message.tool_calls) == 1
    assert isinstance(result[0].message.tool_calls[0], ChatCompletionMessageToolCall)
    assert result[0].finish_reason == "tool_calls"
    assert result[0].message.tool_calls[0].id == "call_123"
    assert result[0].message.tool_calls[0].type == "function"
    assert result[0].message.tool_calls[0].function.name == "get_weather"
    assert (
        result[0].message.tool_calls[0].function.arguments == '{"location": "New York"}'
    )

    # Test case 3: JSON mode conversion
    response_object = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "New York"}',
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ]
    }
    result = _process_choices_in_response(response_object, True)
    assert len(result) == 1
    assert isinstance(result[0], Choices)
    assert result[0].message.content == '{"location": "New York"}'
    assert result[0].message.role == "assistant"
    assert result[0].finish_reason == "stop"

    # Test case 4: Multiple choices
    response_object = {
        "choices": [
            {
                "message": {"content": "Response 1", "role": "assistant"},
                "finish_reason": "length",
            },
            {
                "message": {"content": "Response 2", "role": "assistant"},
                "finish_reason": "stop",
            },
        ]
    }
    result = _process_choices_in_response(response_object, False)
    assert len(result) == 2
    assert result[0].message.content == "Response 1"
    assert result[0].finish_reason == "length"
    assert result[1].message.content == "Response 2"
    assert result[1].finish_reason == "stop"
