"""
Performance benchmarks for the LLM inference (chat completion) hot path.

The end-to-end cases use ``mock_response`` so the full SDK overhead is exercised
-- provider resolution, request/response transformation, ``ModelResponse``
construction, token counting and cost calculation -- without any network I/O. The
``convert_to_model_response_object`` case isolates the provider-response to
``ModelResponse`` translation, the single deterministic core every non-streaming
completion runs.
"""

import pytest

import litellm
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    convert_to_model_response_object,
)
from litellm.types.utils import ModelResponse

SIMPLE_MESSAGES = [{"role": "user", "content": "Hello, how are you?"}]

MULTI_TURN_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
    {
        "role": "assistant",
        "content": "The capital of France is Paris. It is known as the City of Light.",
    },
    {"role": "user", "content": "Tell me more about Paris."},
]

TOOL_DEFINITIONS = [
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
        },
    }
]

MOCK_RESPONSE = "The capital of France is Paris, the country's largest city and cultural centre."

PROVIDER_RESPONSE = {
    "id": "chatcmpl-abc123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": MOCK_RESPONSE},
        }
    ],
    "usage": {"prompt_tokens": 12, "completion_tokens": 16, "total_tokens": 28},
}


@pytest.mark.benchmark
def test_completion_simple_message():
    """Benchmark a single-message completion through the full SDK path."""
    litellm.completion(model="gpt-4o", messages=SIMPLE_MESSAGES, mock_response=MOCK_RESPONSE)


@pytest.mark.benchmark
def test_completion_multi_turn():
    """Benchmark a multi-turn completion through the full SDK path."""
    litellm.completion(model="gpt-4o", messages=MULTI_TURN_MESSAGES, mock_response=MOCK_RESPONSE)


@pytest.mark.benchmark
def test_completion_with_tools():
    """Benchmark a completion that has to process tool schemas."""
    litellm.completion(
        model="gpt-4o",
        messages=SIMPLE_MESSAGES,
        tools=TOOL_DEFINITIONS,
        mock_response=MOCK_RESPONSE,
    )


@pytest.mark.benchmark
def test_completion_streaming():
    """Benchmark consuming a full streamed completion (CustomStreamWrapper)."""
    stream = litellm.completion(
        model="gpt-4o",
        messages=SIMPLE_MESSAGES,
        mock_response=MOCK_RESPONSE,
        stream=True,
    )
    for _ in stream:
        pass


@pytest.mark.benchmark
def test_response_to_model_response_object():
    """Benchmark the provider-response to ModelResponse translation core."""
    convert_to_model_response_object(
        response_object=PROVIDER_RESPONSE,
        model_response_object=ModelResponse(),
    )
