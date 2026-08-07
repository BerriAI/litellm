import json
import os
import sys
from datetime import datetime

sys.path.insert(
    0, os.path.abspath("../../../")
)  # Adds the parent directory to the system path

import litellm
import pytest
from datetime import timedelta

from litellm.types.utils import (
    ModelResponse,
    Message,
    Choices,
    PromptTokensDetailsWrapper,
    CompletionTokensDetailsWrapper,
    Usage,
)

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)


def test_convert_to_model_response_object_basic():
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

    result = convert_to_model_response_object(
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
    assert result.usage.prompt_tokens_details == PromptTokensDetailsWrapper(
        cached_tokens=0
    )
    assert result.usage.completion_tokens_details == CompletionTokensDetailsWrapper(
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

    result = convert_to_model_response_object(
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
    assert result.usage.completion_tokens_details == CompletionTokensDetailsWrapper(
        reasoning_tokens=0
    )

    assert result._hidden_params is not None


def test_convert_to_model_response_object_tool_calls_invalid_json_arguments():
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
    result = convert_to_model_response_object(
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


def test_convert_to_model_response_object_tool_calls_valid_json_arguments():
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
    result = convert_to_model_response_object(
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


def test_convert_to_model_response_object_json_mode():
    """
    This test is verifying that when convert_tool_call_to_json_mode is True, a single tool call's arguments are correctly converted into the message content of the response.
    """
    model_response_object = ModelResponse(model="gpt-3.5-turbo")
    from litellm.constants import RESPONSE_FORMAT_TOOL_NAME

    response_object = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "arguments": '{"key": "value"}',
                                "name": RESPONSE_FORMAT_TOOL_NAME,
                            }
                        }
                    ],
                },
                "finish_reason": None,
            }
        ],
        "usage": {"total_tokens": 10, "prompt_tokens": 5, "completion_tokens": 5},
        "model": "gpt-3.5-turbo",
    }

    # Call the function
    result = convert_to_model_response_object(
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


def test_convert_to_model_response_object_function_output():
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

    result = convert_to_model_response_object(
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
    assert result.usage.completion_tokens_details == CompletionTokensDetailsWrapper(
        reasoning_tokens=0
    )

    assert result._hidden_params is not None


def test_convert_to_model_response_object_with_logprobs():
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

    print("ENTERING CONVERT")
    try:
        result = convert_to_model_response_object(
            model_response_object=ModelResponse(),
            response_object=response_object,
            stream=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            hidden_params=None,
            _response_headers=None,
            convert_tool_call_to_json_mode=False,
        )
    except Exception as e:
        print(f"ERROR: {e}")
        raise e

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
    assert len(choice.logprobs.content) == 9

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
    for i, logprob in enumerate(choice.logprobs.content):
        assert logprob.token == expected_tokens[i]
        assert isinstance(logprob.logprob, float)
        assert isinstance(logprob.bytes, list)
        assert len(logprob.top_logprobs) == 2
        assert isinstance(logprob.top_logprobs[0].token, str)
        assert isinstance(logprob.top_logprobs[0].logprob, float)
        assert isinstance(logprob.top_logprobs[0].bytes, (list, type(None)))

    assert result.usage.prompt_tokens == 9
    assert result.usage.completion_tokens == 9
    assert result.usage.total_tokens == 18
    assert result.usage.completion_tokens_details == CompletionTokensDetailsWrapper(
        reasoning_tokens=0
    )

    assert result.system_fingerprint is None
    assert result._hidden_params is not None


def test_convert_to_model_response_object_error():
    """Test error handling for None response object."""
    with pytest.raises(Exception, match="Error in response object format"):
        convert_to_model_response_object(
            model_response_object=None,
            response_object=None,
            stream=False,
            start_time=None,
            end_time=None,
            hidden_params=None,
            _response_headers=None,
            convert_tool_call_to_json_mode=False,
        )


def test_image_generation_openai_with_pydantic_warning(caplog):
    try:
        import logging
        from litellm.types.utils import ImageResponse, ImageObject

        convert_response_args = {
            "response_object": {
                "created": 1729709945,
                "data": [
                    {
                        "b64_json": None,
                        "revised_prompt": "Generate an image of a baby sea otter. It should look incredibly cute, with big, soulful eyes and a fluffy, wet fur coat. The sea otter should be on its back, as sea otters often do, with its tiny hands holding onto a shell as if it is its precious toy. The background should be a tranquil sea under a clear sky, with soft sunlight reflecting off the waters. The color palette should be soothing with blues, browns, and white.",
                        "url": "https://oaidalleapiprodscus.blob.core.windows.net/private/org-ikDc4ex8NB5ZzfTf8m5WYVB7/user-JpwZsbIXubBZvan3Y3GchiiB/img-LL0uoOv4CFJIvNYxoNCKB8oc.png?st=2024-10-23T17%3A59%3A05Z&se=2024-10-23T19%3A59%3A05Z&sp=r&sv=2024-08-04&sr=b&rscd=inline&rsct=image/png&skoid=d505667d-d6c1-4a0a-bac7-5c84a87759f8&sktid=a48cca56-e6da-484e-a814-9c849652bcb3&skt=2024-10-22T19%3A26%3A22Z&ske=2024-10-23T19%3A26%3A22Z&sks=b&skv=2024-08-04&sig=Hl4wczJ3H2vZNdLRt/7JvNi6NvQGDnbNkDy15%2Bl3k5s%3D",
                    }
                ],
            },
            "model_response_object": ImageResponse(
                created=1729709929,
                data=[],
            ),
            "response_type": "image_generation",
            "stream": False,
            "start_time": None,
            "end_time": None,
            "hidden_params": None,
            "_response_headers": None,
            "convert_tool_call_to_json_mode": None,
        }

        resp: ImageResponse = convert_to_model_response_object(**convert_response_args)
        assert resp is not None
        assert resp.data is not None
        assert len(resp.data) == 1
        assert isinstance(resp.data[0], ImageObject)
    except Exception as e:
        pytest.fail(f"Test failed with exception: {e}")


def test_convert_to_model_response_object_with_empty_str():
    """Test that convert_to_model_response_object handles empty strings correctly."""

    args = {
        "response_object": {
            "id": "chatcmpl-B0b1BmxhH4iSoRvFVbBJdLbMwr346",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": "",
                        "refusal": None,
                        "role": "assistant",
                        "audio": None,
                        "function_call": None,
                        "tool_calls": None,
                    },
                }
            ],
            "created": 1739481997,
            "model": "gpt-4o-mini-2024-07-18",
            "object": "chat.completion",
            "service_tier": "default",
            "system_fingerprint": "fp_bd83329f63",
            "usage": {
                "completion_tokens": 1,
                "prompt_tokens": 121,
                "total_tokens": 122,
                "completion_tokens_details": {
                    "accepted_prediction_tokens": 0,
                    "audio_tokens": 0,
                    "reasoning_tokens": 0,
                    "rejected_prediction_tokens": 0,
                },
                "prompt_tokens_details": {"audio_tokens": 0, "cached_tokens": 0},
            },
        },
        "model_response_object": ModelResponse(
            id="chatcmpl-9f9e5ad2-d570-46fe-a5e0-4983e9774318",
            created=1739481997,
            model=None,
            object="chat.completion",
            system_fingerprint=None,
            choices=[
                Choices(
                    finish_reason="stop",
                    index=0,
                    message=Message(
                        content=None,
                        role="assistant",
                        tool_calls=None,
                        function_call=None,
                        provider_specific_fields=None,
                    ),
                )
            ],
            usage=Usage(
                completion_tokens=0,
                prompt_tokens=0,
                total_tokens=0,
                completion_tokens_details=None,
                prompt_tokens_details=None,
            ),
        ),
        "response_type": "completion",
        "stream": False,
        "start_time": None,
        "end_time": None,
        "hidden_params": None,
        "_response_headers": {
            "date": "Thu, 13 Feb 2025 21:26:37 GMT",
            "content-type": "application/json",
            "transfer-encoding": "chunked",
            "connection": "keep-alive",
            "access-control-expose-headers": "X-Request-ID",
            "openai-organization": "reliablekeystest",
            "openai-processing-ms": "297",
            "openai-version": "2020-10-01",
            "x-ratelimit-limit-requests": "30000",
            "x-ratelimit-limit-tokens": "150000000",
            "x-ratelimit-remaining-requests": "29999",
            "x-ratelimit-remaining-tokens": "149999846",
            "x-ratelimit-reset-requests": "2ms",
            "x-ratelimit-reset-tokens": "0s",
            "x-request-id": "req_651030cbda2c80353086eba8fd0a54ec",
            "strict-transport-security": "max-age=31536000; includeSubDomains; preload",
            "cf-cache-status": "DYNAMIC",
            "set-cookie": "__cf_bm=0ihEMDdqKfEr0I8iP4XZ7C6xEA5rJeAc11XFXNxZgyE-1739481997-1.0.1.1-v5jbjAWhMUZ0faO8q2izQljUQC.R85Vexb18A2MCyS895bur5eRxcguP0.WGY6EkxXSaOKN55VL3Pg3NOdq_xA; path=/; expires=Thu, 13-Feb-25 21:56:37 GMT; domain=.api.openai.com; HttpOnly; Secure; SameSite=None, _cfuvid=jrNMSOBRrxUnGgJ62BltpZZSNImfnEqPX9Uu8meGFLY-1739481997919-0.0.1.1-604800000; path=/; domain=.api.openai.com; HttpOnly; Secure; SameSite=None",
            "x-content-type-options": "nosniff",
            "server": "cloudflare",
            "cf-ray": "9117e5d4caa1f7b5-LAX",
            "content-encoding": "gzip",
            "alt-svc": 'h3=":443"; ma=86400',
        },
        "convert_tool_call_to_json_mode": None,
    }

    resp: ModelResponse = convert_to_model_response_object(**args)
    assert resp is not None
    assert resp.choices[0].message.content is not None


def test_convert_to_model_response_object_with_thinking_content():
    """Test that convert_to_model_response_object handles thinking content correctly."""

    args = {
        "response_object": {
            "id": "chatcmpl-8cc87354-70f3-4a14-b71b-332e965d98d2",
            "created": 1741057687,
            "model": "claude-4-sonnet-20250514",
            "object": "chat.completion",
            "system_fingerprint": None,
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "# LiteLLM\n\nLiteLLM is an open-source library that provides a unified interface for working with various Large Language Models (LLMs). It acts as an abstraction layer that lets developers interact with multiple LLM providers through a single, consistent API.\n\n## Key features:\n\n- **Universal API**: Standardizes interactions with models from OpenAI, Anthropic, Cohere, Azure, and many other providers\n- **Simple switching**: Easily swap between different LLM providers without changing your code\n- **Routing capabilities**: Manage load balancing, fallbacks, and cost optimization\n- **Prompt templates**: Handle different model-specific prompt formats automatically\n- **Logging and observability**: Track usage, performance, and costs across providers\n\nLiteLLM is particularly useful for teams who want flexibility in their LLM infrastructure without creating custom integration code for each provider.",
                        "role": "assistant",
                        "tool_calls": None,
                        "function_call": None,
                        "reasoning_content": "The person is asking about \"litellm\" and included what appears to be a UUID or some form of identifier at the end of their message (fffffe14-7991-43d0-acd8-d3e606db31a8).\n\nLiteLLM is an open-source library/project that provides a unified interface for working with various Large Language Models (LLMs). It's essentially a lightweight package that standardizes the way developers can work with different LLM APIs like OpenAI, Anthropic, Cohere, etc. through a consistent interface.\n\nSome key features and aspects of LiteLLM:\n\n1. Unified API for multiple LLM providers (OpenAI, Anthropic, Azure, etc.)\n2. Standardized input/output formats\n3. Handles routing, fallbacks, and load balancing\n4. Provides logging and observability\n5. Can help with cost tracking across different providers\n6. Makes it easier to switch between different LLM providers\n\nThe UUID-like string they included doesn't seem directly related to the question, unless it's some form of identifier they're including for tracking purposes.",
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "The person is asking about \"litellm\" and included what appears to be a UUID or some form of identifier at the end of their message (fffffe14-7991-43d0-acd8-d3e606db31a8).\n\nLiteLLM is an open-source library/project that provides a unified interface for working with various Large Language Models (LLMs). It's essentially a lightweight package that standardizes the way developers can work with different LLM APIs like OpenAI, Anthropic, Cohere, etc. through a consistent interface.\n\nSome key features and aspects of LiteLLM:\n\n1. Unified API for multiple LLM providers (OpenAI, Anthropic, Azure, etc.)\n2. Standardized input/output formats\n3. Handles routing, fallbacks, and load balancing\n4. Provides logging and observability\n5. Can help with cost tracking across different providers\n6. Makes it easier to switch between different LLM providers\n\nThe UUID-like string they included doesn't seem directly related to the question, unless it's some form of identifier they're including for tracking purposes.",
                                "signature": "ErUBCkYIARgCIkCf+r0qMSOMYkjlFERM00IxsY9I/m19dQGEF/Zv1E0AtvdZjKGnr+nr5vXUldmb/sUCgrQRH4YUyV0X3MoMrsNnEgxDqhUFcUTg1vM0CroaDEY1wKJ0Ca0EZ6S1jCIwF8ATum3xiF/mRSIIjoD6Virh0hFcOfH3Sz6Chtev9WUwwYMAVP4/hyzbrUDnsUlmKh0CfTayaXm6o63/6Kelr6pzLbErjQx2xZRnRjCypw==",
                            }
                        ],
                    },
                }
            ],
            "usage": {
                "completion_tokens": 460,
                "prompt_tokens": 65,
                "total_tokens": 525,
                "completion_tokens_details": None,
                "prompt_tokens_details": {"audio_tokens": None, "cached_tokens": 0},
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
            },
        },
        "model_response_object": ModelResponse(),
    }

    resp: ModelResponse = convert_to_model_response_object(**args)
    assert resp is not None
    assert resp.choices[0].message.reasoning_content is not None


def test_convert_to_model_response_object_with_empty_error_object():
    """
    Test that convert_to_model_response_object handles empty error objects gracefully.

    This is a regression test for issue #18407 where providers like Apertis return
    empty error objects even on successful responses, causing spurious APIErrors.

    The error object structure:
    {
        "error": {
            "message": "",
            "type": "",
            "param": "",
            "code": null
        }
    }
    """
    response_object = {
        "model": "minimax-m2.1",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hey! I'm doing well, thanks for asking!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 49,
            "completion_tokens": 87,
            "total_tokens": 136,
        },
        "error": {
            "message": "",
            "type": "",
            "param": "",
            "code": None,
        },
    }

    # This should NOT raise an exception
    result = convert_to_model_response_object(
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
    assert result.model == "minimax-m2.1"
    assert len(result.choices) == 1
    assert (
        result.choices[0].message.content == "Hey! I'm doing well, thanks for asking!"
    )


def test_convert_to_model_response_object_with_real_error():
    """
    Test that convert_to_model_response_object still raises for real errors.

    Ensures the empty error fix doesn't break legitimate error handling.
    """
    response_object = {
        "error": {
            "message": "Rate limit exceeded",
            "type": "rate_limit_error",
            "param": None,
            "code": 429,
        },
    }

    with pytest.raises(Exception) as exc_info:
        convert_to_model_response_object(
            model_response_object=ModelResponse(),
            response_object=response_object,
            stream=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            hidden_params=None,
            _response_headers=None,
            convert_tool_call_to_json_mode=False,
        )

    # The exception should have the error message
    assert hasattr(exc_info.value, "message")
    assert "Rate limit exceeded" in str(exc_info.value.message)


def test_convert_to_model_response_object_with_empty_dict_error():
    """
    Test that convert_to_model_response_object handles completely empty error dict.
    """
    response_object = {
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
        "error": {},  # Completely empty error object
    }

    # This should NOT raise an exception
    result = convert_to_model_response_object(
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
    assert result.choices[0].message.content == "Hello!"


def test_convert_to_model_response_object_preserves_provider_specific_fields_from_proxy():
    """
    Test that provider_specific_fields (e.g. Anthropic citations) are preserved
    when the response already contains them (e.g. from a proxy passthrough).

    Regression test for https://github.com/BerriAI/litellm/issues/21153
    """
    citations = [
        [
            {
                "type": "web_search_result_location",
                "cited_text": "The Sony WH-1000XM5 remains one of the best...",
                "url": "https://example.com/headphones-review",
                "title": "Best Headphones 2025",
                "supported_text": "Based on current reviews...",
            }
        ],
    ]
    web_search_results = [
        {
            "url": "https://example.com/headphones-review",
            "title": "Best Headphones 2025",
            "snippet": "The Sony WH-1000XM5 remains one of the best...",
        }
    ]

    response_object = {
        "id": "chatcmpl-proxy-123",
        "object": "chat.completion",
        "created": 1728933352,
        "model": "anthropic/claude-opus-4-5-20251101",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Based on current reviews, the Sony WH-1000XM5 remains one of the best headphones.",
                    "tool_calls": [
                        {
                            "id": "call_ws_123",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query": "best headphones 2025"}',
                            },
                        }
                    ],
                    "provider_specific_fields": {
                        "citations": citations,
                        "web_search_results": web_search_results,
                    },
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 50,
            "completion_tokens": 20,
            "total_tokens": 70,
        },
    }

    result = convert_to_model_response_object(
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
    assert result.id == "chatcmpl-proxy-123"

    choice = result.choices[0]
    assert (
        choice.message.content
        == "Based on current reviews, the Sony WH-1000XM5 remains one of the best headphones."
    )
    assert choice.message.provider_specific_fields is not None
    assert "citations" in choice.message.provider_specific_fields
    assert choice.message.provider_specific_fields["citations"] == citations
    assert "web_search_results" in choice.message.provider_specific_fields
    assert (
        choice.message.provider_specific_fields["web_search_results"]
        == web_search_results
    )


def test_convert_to_model_response_object_provider_specific_fields_merges_extra_keys():
    """
    Test that provider_specific_fields from the response are merged with
    any extra non-standard keys present in the message dict.

    Regression test for https://github.com/BerriAI/litellm/issues/21153
    """
    response_object = {
        "id": "chatcmpl-merge-123",
        "object": "chat.completion",
        "created": 1728933352,
        "model": "some-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "provider_specific_fields": {
                        "citations": [{"url": "https://example.com"}],
                    },
                    "custom_extra_field": "extra_value",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    result = convert_to_model_response_object(
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
    psf = result.choices[0].message.provider_specific_fields
    assert psf is not None
    # Both the existing provider_specific_fields and the extra key should be present
    assert "citations" in psf
    assert psf["citations"] == [{"url": "https://example.com"}]
    assert "custom_extra_field" in psf
    assert psf["custom_extra_field"] == "extra_value"


def test_convert_to_model_response_object_no_provider_specific_fields_still_works():
    """
    Test that responses without provider_specific_fields continue to work as before.

    Ensures the fix for https://github.com/BerriAI/litellm/issues/21153
    doesn't break normal responses.
    """
    response_object = {
        "id": "chatcmpl-normal-123",
        "object": "chat.completion",
        "created": 1728933352,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "refusal": None,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        },
    }

    result = convert_to_model_response_object(
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
    psf = result.choices[0].message.provider_specific_fields
    # refusal is not a Message model field, so it should be in provider_specific_fields
    assert psf is not None
    assert "refusal" in psf


def test_convert_to_model_response_object_with_error_code_only():
    """
    Test that errors with only a code (no message) are still treated as real errors.
    """
    response_object = {
        "error": {
            "message": "",
            "code": 500,
        },
    }

    with pytest.raises(Exception):
        convert_to_model_response_object(
            model_response_object=ModelResponse(),
            response_object=response_object,
            stream=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
            hidden_params=None,
            _response_headers=None,
            convert_tool_call_to_json_mode=False,
        )


def test_model_prefix_preservation():
    """
    Test that when model_response_object has a prefix like 'openai/gpt-4'
    and the response contains a different model name, the prefix is preserved.
    """
    response_object = {
        "id": "chatcmpl-prefix-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "gpt-4o",
    }

    result = convert_to_model_response_object(
        model_response_object=ModelResponse(model="openai/gpt-4"),
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert result.model == "openai/gpt-4o"


def test_model_without_prefix():
    """
    Test that when model_response_object has no prefix (e.g. 'gpt-4'),
    the original model is kept (provider response model is ignored).
    """
    response_object = {
        "id": "chatcmpl-no-prefix",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "model": "gpt-4o-2024-08-06",
    }

    result = convert_to_model_response_object(
        model_response_object=ModelResponse(model="gpt-4"),
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert result.model == "gpt-4"


def test_extra_response_fields_preserved():
    """
    Test that extra response fields (e.g. service_tier) are preserved
    on the returned ModelResponse object.
    """
    response_object = {
        "id": "chatcmpl-extra-fields",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "gpt-4o",
        "service_tier": "default",
    }

    result = convert_to_model_response_object(
        model_response_object=ModelResponse(),
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert result.service_tier == "default"


def test_hidden_params_and_response_headers_set():
    """
    Test that _hidden_params and _response_headers are correctly set
    on the returned ModelResponse.
    """
    response_object = {
        "id": "chatcmpl-headers",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "gpt-4o",
    }
    response_headers = {"x-request-id": "req_abc123"}

    result = convert_to_model_response_object(
        model_response_object=ModelResponse(),
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
        hidden_params={"custom_key": "custom_value"},
        _response_headers=response_headers,
    )

    assert result._hidden_params is not None
    assert result._hidden_params["custom_key"] == "custom_value"
    assert "additional_headers" in result._hidden_params
    assert result._response_headers == response_headers


def test_response_ms_computed():
    """
    Test that _response_ms is computed correctly from start_time and end_time.
    """
    response_object = {
        "id": "chatcmpl-timing",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        "model": "gpt-4o",
    }
    start = datetime(2024, 1, 1, 12, 0, 0)
    end = start + timedelta(milliseconds=250)

    result = convert_to_model_response_object(
        model_response_object=ModelResponse(),
        response_object=response_object,
        stream=False,
        start_time=start,
        end_time=end,
    )

    assert result._response_ms == pytest.approx(250.0)


def test_error_message_includes_function_args():
    """
    Test that when an exception occurs, the error message includes
    the function arguments for debugging (deferred locals() - Opt 2).
    """
    # Pass a response_object whose choices survive the missing-choices guard
    # but raise inside the conversion loop (the choice lacks a "message" key),
    # so the generic debugging handler builds the received_args message.
    response_object = {
        "choices": [{"index": 0}],
    }

    with pytest.raises(Exception) as exc_info:
        convert_to_model_response_object(
            model_response_object=ModelResponse(),
            response_object=response_object,
            stream=False,
            start_time=datetime.now(),
            end_time=datetime.now(),
        )

    error_msg = str(exc_info.value)
    assert "received_args=" in error_msg
    assert "response_object" in error_msg
    assert "response_type" in error_msg


@pytest.mark.parametrize("falsy_id", [None, ""])
def test_convert_to_model_response_object_falsy_id_preserves_auto_generated(falsy_id):
    """Test that a falsy id in response_object preserves the auto-generated id."""
    mr = ModelResponse()
    original_id = mr.id
    response_object = {
        "id": falsy_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        "model": "test-model",
    }
    result = convert_to_model_response_object(
        model_response_object=mr,
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )
    assert result.id == original_id
    assert result.id.startswith("chatcmpl-")


def test_convert_to_model_response_object_default_usage_overwritten():
    """
    Regression test: convert_to_model_response_object must properly set Usage
    on a ModelResponse that only has the default Usage from ModelResponse.__init__()
    (i.e. no extra litellm.Usage() set via setattr beforehand).

    This validates the optimization of removing the redundant
    `setattr(model_response, "usage", litellm.Usage())` in completion().
    """
    mr = ModelResponse()
    # usage is not set by default (optimization: avoid constructing throwaway Usage)
    assert not hasattr(mr, "usage")

    response_object = {
        "id": "chatcmpl-usage-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 7,
            "total_tokens": 22,
        },
        "model": "gpt-4o",
    }

    result = convert_to_model_response_object(
        model_response_object=mr,
        response_object=response_object,
        stream=False,
        start_time=datetime.now(),
        end_time=datetime.now(),
    )

    assert isinstance(result, ModelResponse)
    assert result.usage.prompt_tokens == 15
    assert result.usage.completion_tokens == 7
    assert result.usage.total_tokens == 22


def test_convert_to_model_response_object_with_null_top_logprobs():
    """
    Test that convert_to_model_response_object handles null top_logprobs
    without raising a Pydantic validation error.

    Some providers return null for top_logprobs when logprobs=true but
    top_logprobs is unset/0. The OpenAI spec requires top_logprobs to be
    an array, so litellm should normalize null to [].

    Regression test for https://github.com/BerriAI/litellm/issues/21932
    """
    response_object = {
        "id": "chatcmpl-a21e454401074fd8814736d84dcbb1e4",
        "object": "chat.completion",
        "created": 1771632698,
        "model": "my-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Silent light above.",
                },
                "finish_reason": "stop",
                "logprobs": {
                    "content": [
                        {
                            "token": "Sil",
                            "bytes": [83, 105, 108],
                            "logprob": -2.1518118381500244,
                            "top_logprobs": None,
                        },
                        {
                            "token": "ent",
                            "bytes": [101, 110, 116],
                            "logprob": -0.13957086205482483,
                            "top_logprobs": None,
                        },
                        {
                            "token": " light",
                            "bytes": [32, 108, 105, 103, 104, 116],
                            "logprob": -1.3923776149749756,
                            "top_logprobs": None,
                        },
                        {
                            "token": " above",
                            "bytes": [32, 97, 98, 111, 118, 101],
                            "logprob": -1.137486219406128,
                            "top_logprobs": None,
                        },
                        {
                            "token": ".",
                            "bytes": [46],
                            "logprob": -0.1709611415863037,
                            "top_logprobs": None,
                        },
                    ],
                    "refusal": None,
                },
            }
        ],
        "usage": {
            "prompt_tokens": 73,
            "completion_tokens": 5,
            "total_tokens": 78,
        },
    }

    result = convert_to_model_response_object(
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
    assert len(result.choices) == 1

    choice = result.choices[0]
    assert choice.logprobs is not None
    assert len(choice.logprobs.content) == 5

    # Verify all null top_logprobs were normalized to empty lists
    for token_logprob in choice.logprobs.content:
        assert token_logprob.top_logprobs == []
        assert isinstance(token_logprob.top_logprobs, list)


class TestMissingChoicesGuard:
    """
    Tests for the defense-in-depth guard that raises APIError when a provider
    returns a response with no 'choices' field.

    See: https://github.com/BerriAI/litellm/issues/29391
    """

    def test_convert_to_model_response_object_no_choices_raises_api_error(self):
        """Missing choices in non-streaming path raises APIError, not IndexError."""
        from litellm.exceptions import APIError

        response_object = {
            "id": "msg_123",
            "model": "some-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

        with pytest.raises(APIError) as exc_info:
            convert_to_model_response_object(
                response_object=response_object,
                model_response_object=ModelResponse(),
            )

        assert "no 'choices'" in exc_info.value.message

    def test_convert_to_model_response_object_empty_choices_raises_api_error(self):
        """Empty choices list raises APIError, same as missing/null choices.

        Provider-specific repair (e.g. github_copilot synthesizing choices for
        Anthropic-native responses) happens before this guard, in the provider
        config; the core utility keeps treating empty choices as an error.
        """
        from litellm.exceptions import APIError

        response_object = {
            "id": "msg_123",
            "model": "some-model",
            "choices": [],
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

        with pytest.raises(APIError) as exc_info:
            convert_to_model_response_object(
                response_object=response_object,
                model_response_object=ModelResponse(),
            )

        assert "no 'choices'" in exc_info.value.message

    def test_convert_to_model_response_object_null_choices_raises_api_error(self):
        """choices=None raises APIError."""
        from litellm.exceptions import APIError

        response_object = {
            "id": "msg_123",
            "model": "some-model",
            "choices": None,
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

        with pytest.raises(APIError) as exc_info:
            convert_to_model_response_object(
                response_object=response_object,
                model_response_object=ModelResponse(),
            )

        assert "no 'choices'" in exc_info.value.message

    def test_convert_to_streaming_response_no_choices_raises_api_error(self):
        """Missing choices in streaming cache-hit path raises APIError."""
        from litellm.exceptions import APIError
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_streaming_response,
        )

        response_object = {
            "id": "msg_123",
            "model": "some-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

        with pytest.raises(APIError) as exc_info:
            # convert_to_streaming_response is a generator, must consume it
            list(convert_to_streaming_response(response_object=response_object))

        assert "no 'choices'" in exc_info.value.message

    def test_convert_to_model_response_object_stream_true_no_choices_raises_api_error(
        self,
    ):
        """Missing choices via stream=True path raises APIError when generator is consumed."""
        from litellm.exceptions import APIError

        response_object = {
            "id": "msg_123",
            "model": "some-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

        with pytest.raises(APIError) as exc_info:
            list(
                convert_to_model_response_object(
                    response_object=response_object,
                    model_response_object=ModelResponse(),
                    stream=True,
                )
            )

        assert "no 'choices'" in exc_info.value.message

    def test_convert_to_streaming_response_async_no_choices_raises_api_error(self):
        """Missing choices in async streaming path raises APIError."""
        import asyncio

        from litellm.exceptions import APIError
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_streaming_response_async,
        )

        response_object = {
            "id": "msg_123",
            "model": "some-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
        }

        async def consume():
            chunks = []
            async for chunk in convert_to_streaming_response_async(
                response_object=response_object
            ):
                chunks.append(chunk)
            return chunks

        with pytest.raises(APIError) as exc_info:
            asyncio.run(consume())

        assert "no 'choices'" in exc_info.value.message

    def test_error_message_includes_response_keys(self):
        """The error message should include the keys present in the response for debugging."""
        from litellm.exceptions import APIError

        response_object = {
            "id": "msg_123",
            "model": "some-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11},
            "copilot_usage": {"total_nano_aiu": 9500000},
        }

        with pytest.raises(APIError) as exc_info:
            convert_to_model_response_object(
                response_object=response_object,
                model_response_object=ModelResponse(),
            )

        assert "copilot_usage" in exc_info.value.message


class TestNormalizeImagesForMessage:
    def test_none_returns_none(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _normalize_images_for_message,
        )

        assert _normalize_images_for_message(None) is None

    def test_empty_list_returns_empty(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _normalize_images_for_message,
        )

        assert _normalize_images_for_message([]) == []

    def test_adds_index_when_missing(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _normalize_images_for_message,
        )

        images = [{"url": "http://a.png"}, {"url": "http://b.png"}]
        result = _normalize_images_for_message(images)
        assert result[0]["index"] == 0
        assert result[1]["index"] == 1
        assert result[0]["url"] == "http://a.png"

    def test_preserves_existing_index(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _normalize_images_for_message,
        )

        images = [{"url": "http://a.png", "index": 5}]
        result = _normalize_images_for_message(images)
        assert result[0]["index"] == 5


class TestSafeConvertCreatedField:
    def test_none_returns_current_time(self):
        import time

        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _safe_convert_created_field,
        )

        result = _safe_convert_created_field(None)
        assert abs(result - int(time.time())) <= 1

    def test_int_passthrough(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _safe_convert_created_field,
        )

        assert _safe_convert_created_field(1700000000) == 1700000000

    def test_float_truncated(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _safe_convert_created_field,
        )

        assert _safe_convert_created_field(1700000000.999) == 1700000000

    def test_string_converted(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _safe_convert_created_field,
        )

        assert _safe_convert_created_field("1700000000.5") == 1700000000

    def test_invalid_string_returns_current_time(self):
        import time

        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _safe_convert_created_field,
        )

        result = _safe_convert_created_field("not-a-number")
        assert abs(result - int(time.time())) <= 1


class TestConvertToStreamingResponse:
    def test_none_raises(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_streaming_response,
        )

        with pytest.raises(Exception, match="Error in response object format"):
            list(convert_to_streaming_response(response_object=None))

    def test_happy_path_basic(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_streaming_response,
        )

        response_object = {
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "created": 1700000000,
            "system_fingerprint": "fp_abc",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "Hello!", "role": "assistant"},
                }
            ],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "total_tokens": 7,
            },
        }

        chunks = list(convert_to_streaming_response(response_object=response_object))
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.id == "chatcmpl-123"
        assert chunk.model == "gpt-4"
        assert chunk.created == 1700000000
        assert chunk.system_fingerprint == "fp_abc"
        assert chunk.choices[0].delta.content == "Hello!"
        assert chunk.choices[0].delta.role == "assistant"
        assert chunk.choices[0].finish_reason == "stop"
        assert chunk.usage.prompt_tokens == 5
        assert chunk.usage.completion_tokens == 2

    def test_finish_details_fallback(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_streaming_response,
        )

        response_object = {
            "choices": [
                {
                    "finish_reason": None,
                    "finish_details": "length",
                    "message": {"content": "Hi", "role": "assistant"},
                }
            ],
        }

        chunks = list(convert_to_streaming_response(response_object=response_object))
        assert chunks[0].choices[0].finish_reason == "length"

    def test_tool_calls_in_streaming(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_streaming_response_async,
        )
        import asyncio

        response_object = {
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "index": 0,
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                }
            ],
        }

        async def run():
            chunks = []
            async for chunk in convert_to_streaming_response_async(
                response_object=response_object
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        assert len(chunks) == 1
        assert chunks[0].choices[0].delta.tool_calls[0].id == "call_1"
        assert chunks[0].choices[0].delta.tool_calls[0].function.name == "get_weather"


class TestConvertToStreamingResponseAsync:
    def test_none_raises(self):
        import asyncio

        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_streaming_response_async,
        )

        async def run():
            async for _ in convert_to_streaming_response_async(response_object=None):
                pass

        with pytest.raises(Exception, match="Error in response object format"):
            asyncio.run(run())

    def test_happy_path(self):
        import asyncio

        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_streaming_response_async,
        )

        response_object = {
            "id": "msg_async_1",
            "model": "claude-3",
            "created": 1700000000,
            "system_fingerprint": "fp_xyz",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "Hi there", "role": "assistant"},
                }
            ],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 2,
                "total_tokens": 5,
            },
        }

        async def run():
            chunks = []
            async for chunk in convert_to_streaming_response_async(
                response_object=response_object
            ):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.run(run())
        # Cached replay is sliced into word-shaped chunks to preserve
        # streaming cadence; joining the slices reconstructs the content.
        assert len(chunks) == 2
        assert all(c.id == "msg_async_1" for c in chunks)
        assert all(c.model == "claude-3" for c in chunks)
        assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Hi there"
        assert chunks[0].choices[0].finish_reason is None
        assert chunks[-1].choices[0].finish_reason == "stop"
        assert chunks[-1].usage.prompt_tokens == 3

    def test_preserves_reasoning_fields(self):
        import asyncio

        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_to_streaming_response_async,
        )

        thinking_blocks = [
            {
                "type": "thinking",
                "thinking": "cached reasoning",
                "signature": "sig-cache",
            }
        ]
        response_object = {
            "id": "msg_async_reasoning_cache",
            "model": "claude-3",
            "created": 1700000000,
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "Final answer",
                        "role": "assistant",
                        "reasoning_content": "cached reasoning",
                        "thinking_blocks": thinking_blocks,
                    },
                }
            ],
        }

        async def run():
            return [
                chunk
                async for chunk in convert_to_streaming_response_async(
                    response_object=response_object
                )
            ]

        chunks = asyncio.run(run())

        assert (
            "".join(c.choices[0].delta.content or "" for c in chunks) == "Final answer"
        )
        assert chunks[0].choices[0].delta.reasoning_content == "cached reasoning"
        assert chunks[0].choices[0].delta.thinking_blocks == thinking_blocks
        assert not any(
            hasattr(c.choices[0].delta, "reasoning_content") for c in chunks[1:]
        )
        assert not any(
            hasattr(c.choices[0].delta, "thinking_blocks") for c in chunks[1:]
        )


class TestHandleInvalidParallelToolCalls:
    def test_none_input(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _handle_invalid_parallel_tool_calls,
        )

        assert _handle_invalid_parallel_tool_calls(None) is None

    def test_normal_tool_calls_unchanged(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _handle_invalid_parallel_tool_calls,
        )
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(name="get_weather", arguments='{"city": "NYC"}'),
            )
        ]
        result = _handle_invalid_parallel_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0].function.name == "get_weather"

    def test_multi_tool_use_parallel_expanded(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _handle_invalid_parallel_tool_calls,
        )
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name="multi_tool_use.parallel",
                    arguments=json.dumps(
                        {
                            "tool_uses": [
                                {
                                    "recipient_name": "functions.get_weather",
                                    "parameters": {"city": "NYC"},
                                },
                                {
                                    "recipient_name": "functions.get_time",
                                    "parameters": {"tz": "EST"},
                                },
                            ]
                        }
                    ),
                ),
            )
        ]
        result = _handle_invalid_parallel_tool_calls(tool_calls)
        assert len(result) == 2
        assert result[0].function.name == "get_weather"
        assert result[0].id == "call_1_0"
        assert json.loads(result[0].function.arguments) == {"city": "NYC"}
        assert result[1].function.name == "get_time"
        assert result[1].id == "call_1_1"

    def test_invalid_json_returns_original(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _handle_invalid_parallel_tool_calls,
        )
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(name="some_func", arguments="not valid json{{{"),
            )
        ]
        result = _handle_invalid_parallel_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0].id == "call_1"


class TestShouldConvertToolCallToJsonMode:
    def test_returns_true_when_conditions_met(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _should_convert_tool_call_to_json_mode,
        )
        from litellm.constants import RESPONSE_FORMAT_TOOL_NAME

        tool_calls = [{"function": {"name": RESPONSE_FORMAT_TOOL_NAME}}]
        assert (
            _should_convert_tool_call_to_json_mode(
                tool_calls=tool_calls, convert_tool_call_to_json_mode=True
            )
            is True
        )

    def test_returns_false_when_flag_off(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _should_convert_tool_call_to_json_mode,
        )
        from litellm.constants import RESPONSE_FORMAT_TOOL_NAME

        tool_calls = [{"function": {"name": RESPONSE_FORMAT_TOOL_NAME}}]
        assert (
            _should_convert_tool_call_to_json_mode(
                tool_calls=tool_calls, convert_tool_call_to_json_mode=False
            )
            is False
        )

    def test_returns_false_when_wrong_tool_name(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _should_convert_tool_call_to_json_mode,
        )

        tool_calls = [{"function": {"name": "some_other_tool"}}]
        assert (
            _should_convert_tool_call_to_json_mode(
                tool_calls=tool_calls, convert_tool_call_to_json_mode=True
            )
            is False
        )

    def test_returns_false_when_multiple_tool_calls(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _should_convert_tool_call_to_json_mode,
        )
        from litellm.constants import RESPONSE_FORMAT_TOOL_NAME

        tool_calls = [
            {"function": {"name": RESPONSE_FORMAT_TOOL_NAME}},
            {"function": {"name": "other"}},
        ]
        assert (
            _should_convert_tool_call_to_json_mode(
                tool_calls=tool_calls, convert_tool_call_to_json_mode=True
            )
            is False
        )

    def test_returns_false_when_none(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            _should_convert_tool_call_to_json_mode,
        )

        assert (
            _should_convert_tool_call_to_json_mode(
                tool_calls=None, convert_tool_call_to_json_mode=True
            )
            is False
        )


class TestConvertToolCallToJsonMode:
    def test_converts_when_should(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_tool_call_to_json_mode as convert_fn,
        )
        from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name=RESPONSE_FORMAT_TOOL_NAME,
                    arguments='{"key": "value"}',
                ),
            )
        ]
        message, finish_reason = convert_fn(
            tool_calls=tool_calls, convert_tool_call_to_json_mode=True
        )
        assert message is not None
        assert message.content == '{"key": "value"}'
        assert finish_reason == "stop"

    def test_no_conversion_when_flag_false(self):
        from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
            convert_tool_call_to_json_mode as convert_fn,
        )
        from litellm.constants import RESPONSE_FORMAT_TOOL_NAME
        from litellm.types.utils import ChatCompletionMessageToolCall, Function

        tool_calls = [
            ChatCompletionMessageToolCall(
                id="call_1",
                type="function",
                function=Function(
                    name=RESPONSE_FORMAT_TOOL_NAME,
                    arguments='{"key": "value"}',
                ),
            )
        ]
        message, finish_reason = convert_fn(
            tool_calls=tool_calls, convert_tool_call_to_json_mode=False
        )
        assert message is None
        assert finish_reason is None


class TestConvertToModelResponseObjectEmbedding:
    def test_basic_embedding_response(self):
        from litellm.types.utils import EmbeddingResponse

        response_object = {
            "model": "text-embedding-ada-002",
            "object": "list",
            "data": [{"embedding": [0.1, 0.2, 0.3], "index": 0}],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 0,
                "total_tokens": 5,
            },
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=EmbeddingResponse(),
            response_type="embedding",
        )
        assert result.model == "text-embedding-ada-002"
        assert result.object == "list"
        assert result.data == [{"embedding": [0.1, 0.2, 0.3], "index": 0}]
        assert result.usage.prompt_tokens == 5


class TestConvertToModelResponseObjectAudioTranscription:
    def test_basic_transcription(self):
        from litellm.types.utils import TranscriptionResponse

        response_object = {
            "text": "Hello world",
            "language": "en",
            "duration": 1.5,
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=TranscriptionResponse(),
            response_type="audio_transcription",
        )
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 1.5

    def test_transcription_with_duration_usage(self):
        from litellm.types.utils import TranscriptionResponse

        response_object = {
            "text": "Hello",
            "usage": {"type": "duration", "seconds": 3.0},
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=TranscriptionResponse(),
            response_type="audio_transcription",
        )
        assert result.text == "Hello"
        assert result.usage.seconds == 3.0

    def test_transcription_with_token_usage(self):
        from litellm.types.utils import TranscriptionResponse

        response_object = {
            "text": "Hi",
            "usage": {
                "type": "tokens",
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "input_token_details": {"audio_tokens": 4, "text_tokens": 6},
            },
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=TranscriptionResponse(),
            response_type="audio_transcription",
        )
        assert result.text == "Hi"
        assert result.usage.input_tokens == 10
        assert result.usage.output_tokens == 5
        assert result.usage.input_token_details.audio_tokens == 4


class TestConvertToModelResponseObjectRerank:
    def test_basic_rerank(self):
        from litellm.types.utils import RerankResponse

        response_object = {
            "id": "rerank-123",
            "meta": {"model": "rerank-v1"},
            "results": [{"index": 0, "relevance_score": 0.9}],
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=None,
            response_type="rerank",
        )
        assert result.id == "rerank-123"
        assert result.results[0]["relevance_score"] == 0.9


class TestConvertToModelResponseObjectCompletion:
    def test_tool_calls_finish_reason_override(self):
        response_object = {
            "id": "chatcmpl-1",
            "model": "gpt-4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=ModelResponse(),
        )
        assert result.choices[0].finish_reason == "tool_calls"

    def test_multiple_choices(self):
        response_object = {
            "id": "chatcmpl-2",
            "model": "gpt-4",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {"content": "Answer A", "role": "assistant"},
                },
                {
                    "finish_reason": "stop",
                    "index": 1,
                    "message": {"content": "Answer B", "role": "assistant"},
                },
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=ModelResponse(),
        )
        assert len(result.choices) == 2
        assert result.choices[0].message.content == "Answer A"
        assert result.choices[1].message.content == "Answer B"
        assert result.choices[1].index == 1

    def test_json_mode_conversion(self):
        from litellm.constants import RESPONSE_FORMAT_TOOL_NAME

        response_object = {
            "id": "chatcmpl-3",
            "model": "gpt-3.5-turbo",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": None,
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": RESPONSE_FORMAT_TOOL_NAME,
                                    "arguments": '{"result": 42}',
                                },
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=ModelResponse(),
            convert_tool_call_to_json_mode=True,
        )
        assert result.choices[0].message.content == '{"result": 42}'
        assert result.choices[0].finish_reason == "stop"

    def test_reasoning_content_extracted(self):
        response_object = {
            "id": "chatcmpl-4",
            "model": "o1",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "The answer is 4.",
                        "role": "assistant",
                        "reasoning_content": "2+2=4",
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=ModelResponse(),
        )
        assert result.choices[0].message.content == "The answer is 4."
        assert result.choices[0].message.reasoning_content == "2+2=4"

    def test_reasoning_content_not_mirrored_into_provider_specific_fields(self):
        """Mirroring reasoning_content into provider_specific_fields made
        cache-replayed messages diverge from live Anthropic messages, which
        only set it top-level, breaking cache key stability (issue #27337)."""
        response_object = {
            "id": "chatcmpl-5",
            "model": "claude-sonnet-4-5",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                        "content": "The answer is 4.",
                        "role": "assistant",
                        "reasoning_content": "2+2=4",
                        "thinking_blocks": [
                            {
                                "type": "thinking",
                                "thinking": "2+2=4",
                                "signature": "sig",
                            }
                        ],
                    },
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
        }

        result = convert_to_model_response_object(
            response_object=response_object,
            model_response_object=ModelResponse(),
        )
        message = result.choices[0].message
        assert message.reasoning_content == "2+2=4"
        assert "reasoning_content" not in (message.provider_specific_fields or {})

    def test_response_none_raises(self):
        with pytest.raises(Exception):
            convert_to_model_response_object(
                response_object=None,
                model_response_object=ModelResponse(),
            )

    def test_model_response_none_raises(self):
        with pytest.raises(Exception):
            convert_to_model_response_object(
                response_object={
                    "choices": [
                        {
                            "message": {"content": "hi", "role": "assistant"},
                            "finish_reason": "stop",
                        }
                    ]
                },
                model_response_object=None,
            )
