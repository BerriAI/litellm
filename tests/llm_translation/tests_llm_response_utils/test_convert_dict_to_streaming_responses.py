import json
import os
import sys
from datetime import datetime
import pytest

sys.path.insert(
    0, os.path.abspath("../../")
)  # Adds the parent directory to the system path

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_streaming_response import (
    convert_to_streaming_response_async,
    convert_to_streaming_response,
    _get_streaming_response_object,
)
from litellm.types.utils import (
    ModelResponse,
    ChatCompletionDeltaToolCall,
)


# Sample response object for testing
sample_response = {
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


@pytest.mark.asyncio
async def test_convert_to_streaming_response_async():
    """

    Convert a Dict of a Non-Streamed Response to a Streaming Response

    Used for caching hits when stream == True

    """
    async for response in convert_to_streaming_response_async(sample_response):
        assert isinstance(response, ModelResponse)
        assert (
            response.choices[0].delta.content == "Hi there! How can I assist you today?"
        )
        assert response.choices[0].delta.role == "assistant"
        assert response.id == "chatcmpl-123456"
        assert response.created == 1728933352
        assert response.model == "gpt-4o-2024-08-06"
        assert response.system_fingerprint == "fp_6b68a8204b"
        assert response.usage.prompt_tokens == 19
        assert response.usage.completion_tokens == 10
        assert response.usage.total_tokens == 29


def test_convert_to_streaming_response():
    for response in convert_to_streaming_response(sample_response):
        assert isinstance(response, ModelResponse)
        assert (
            response.choices[0].delta.content == "Hi there! How can I assist you today?"
        )
        assert response.choices[0].delta.role == "assistant"
        assert response.id == "chatcmpl-123456"
        assert response.created == 1728933352
        assert response.model == "gpt-4o-2024-08-06"
        assert response.system_fingerprint == "fp_6b68a8204b"
        assert response.usage.prompt_tokens == 19
        assert response.usage.completion_tokens == 10
        assert response.usage.total_tokens == 29


def test_get_streaming_response_object():
    """
    Get a Streaming Response Object from a Dict of a Non-Streamed Response
    """
    response = _get_streaming_response_object(sample_response)
    assert isinstance(response, ModelResponse)
    assert response.choices[0].delta.content == "Hi there! How can I assist you today?"
    assert response.choices[0].delta.role == "assistant"
    assert response.id == "chatcmpl-123456"
    assert response.created == 1728933352
    assert response.model == "gpt-4o-2024-08-06"
    assert response.system_fingerprint == "fp_6b68a8204b"
    assert response.usage.prompt_tokens == 19
    assert response.usage.completion_tokens == 10
    assert response.usage.total_tokens == 29


def test_get_streaming_response_object_with_tool_calls():
    """
    Get a Streaming Response Object from a Dict of a Non-Streamed Response with Tool Calls
    """
    response_with_tool_calls = {
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

    response = _get_streaming_response_object(response_with_tool_calls)
    assert isinstance(response, ModelResponse)
    assert isinstance(
        response.choices[0].delta.tool_calls[0], ChatCompletionDeltaToolCall
    )
    assert response.choices[0].delta.tool_calls[0].id == "call_GED1Xit8lU7cNsjVM6dt2fTq"
    assert (
        response.choices[0].delta.tool_calls[0].function.name == "get_current_weather"
    )
    assert (
        response.choices[0].delta.tool_calls[0].function.arguments
        == '{"location":"Boston, MA","unit":"fahren'
    )


def test_get_streaming_response_object_with_none_input():
    with pytest.raises(Exception, match="Error in response object format"):
        _get_streaming_response_object(None)
