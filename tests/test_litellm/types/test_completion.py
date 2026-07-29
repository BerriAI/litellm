"""
Tests for litellm.types.completion module

This test suite validates the CompletionRequest model and its compatibility with 
OpenAI ChatCompletion API message formats.

Usage:
    pytest tests/test_litellm/types/test_completion.py -v
"""

import dataclasses
from typing import List

import pytest

from litellm.types.completion import (
    ChatCompletionMessageParam,
    CompletionRequest,
    _CompletionDispatchContext,
)


def test_completion_request_messages_type_validation():
    """
    Test that CompletionRequest.messages field accepts proper ChatCompletionMessageParam types.
    """
    # Valid message formats according to OpenAI API
    valid_messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
    ]

    request = CompletionRequest(model="gpt-3.5-turbo", messages=valid_messages)

    assert request.model == "gpt-3.5-turbo"
    assert len(request.messages) == 3


def test_completion_request_tool_message():
    """
    Test CompletionRequest with tool message format.
    """
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "Calculate 2+2"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "calculate",
                        "arguments": '{"expression": "2+2"}',
                    },
                }
            ],
        },
        {"role": "tool", "content": "4", "tool_call_id": "call_123"},
    ]

    request = CompletionRequest(model="gpt-3.5-turbo", messages=messages)

    assert len(request.messages) == 3
    assert request.messages[1]["role"] == "assistant"
    assert request.messages[2]["role"] == "tool"


def test_completion_request_function_message():
    """
    Test CompletionRequest with deprecated function message format.
    """
    messages: List[ChatCompletionMessageParam] = [
        {"role": "user", "content": "What's the weather?"},
        {
            "role": "assistant",
            "content": None,
            "function_call": {
                "name": "get_weather",
                "arguments": '{"location": "NYC"}',
            },
        },
        {"role": "function", "name": "get_weather", "content": "Sunny, 75°F"},
    ]

    request = CompletionRequest(model="gpt-3.5-turbo", messages=messages)

    assert len(request.messages) == 3
    assert request.messages[2]["role"] == "function"
    assert request.messages[2]["name"] == "get_weather"


def test_completion_request_multimodal_content():
    """
    Test CompletionRequest with multimodal content (text + image).
    """
    messages: List[ChatCompletionMessageParam] = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD..."
                    },
                },
            ],
        }
    ]

    request = CompletionRequest(model="gpt-4-vision-preview", messages=messages)

    assert len(request.messages) == 1
    assert request.messages[0]["role"] == "user"


def test_completion_request_empty_messages_default():
    """
    Test that CompletionRequest defaults to empty messages list.
    """
    request = CompletionRequest(model="gpt-3.5-turbo")

    assert request.messages == []
    assert isinstance(request.messages, list)


def test_completion_request_with_all_params():
    """
    Test CompletionRequest with various optional parameters.
    """
    messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": "Hello"}]

    request = CompletionRequest(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop={"sequences": ["END"]},
        stream=False,
        n=1,
    )

    assert request.model == "gpt-3.5-turbo"
    assert request.temperature == 0.7
    assert request.max_tokens == 100
    assert request.top_p == 0.9
    assert request.frequency_penalty == 0.0
    assert request.presence_penalty == 0.0
    assert request.stream is False
    assert request.n == 1


def _build_dispatch_context() -> _CompletionDispatchContext:
    return _CompletionDispatchContext(
        _azure_detection_model="gpt-4o",
        acompletion=False,
        api_base=None,
        api_key=None,
        api_version=None,
        client=None,
        custom_llm_provider="openai",
        custom_prompt_dict={},
        extra_headers=None,
        headers={},
        hf_model_name=None,
        kwargs={},
        litellm_params={},
        logger_fn=None,
        logging=None,  # type: ignore[arg-type]
        max_retries=None,
        max_tokens=None,
        messages=[],
        metadata=None,
        model="gpt-4o",
        model_response=None,  # type: ignore[arg-type]
        optional_params={},
        organization=None,
        provider_config=None,
        shared_session=None,
        stream=None,
        temperature=None,
        text_completion=False,
        timeout=None,
        top_p=None,
    )


def test_dispatch_context_is_frozen():
    """A helper must not be able to re-route the call by rebinding a dispatch
    input mid-flight; this pins the frozen invariant the dispatch shape relies on."""
    ctx = _build_dispatch_context()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.model = "claude-haiku-4-5"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.custom_llm_provider = "anthropic"  # type: ignore[misc]


def test_dispatch_context_uses_slots():
    """slots=True keeps the per-call context lightweight (no per-instance __dict__)."""
    ctx = _build_dispatch_context()
    assert not hasattr(ctx, "__dict__")
    assert hasattr(type(ctx), "__slots__")
