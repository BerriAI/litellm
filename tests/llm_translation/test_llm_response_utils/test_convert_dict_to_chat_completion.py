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
            "model": "claude-3-7-sonnet-20250219",
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
    assert result.choices[0].message.content == "Hey! I'm doing well, thanks for asking!"


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
