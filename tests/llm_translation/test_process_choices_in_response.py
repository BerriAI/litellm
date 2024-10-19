import pytest
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_chat_completion_response import (
    _process_choices_in_response,
)
from litellm.types.utils import Message, Choices, ChatCompletionMessageToolCall


def test_basic_response():
    """
    Basic Response from OpenAI API
    """
    response_object = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "\n\nHello there, how may I assist you today?",
                },
                "logprobs": None,
                "finish_reason": "stop",
            }
        ]
    }
    _choices = _process_choices_in_response(response_object, False)

    assert len(_choices) == 1
    assert isinstance(_choices[0], Choices)
    assert _choices[0].message.content == "\n\nHello there, how may I assist you today?"
    assert _choices[0].message.role == "assistant"
    assert _choices[0].finish_reason == "stop"


def test_tool_calls():
    """
    Basic Response with tool calls
    """
    response_object = {
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
    }
    result = _process_choices_in_response(response_object, False)

    assert len(result) == 1
    assert isinstance(result[0], Choices)
    assert result[0].message.content is None
    assert result[0].message.role == "assistant"
    assert len(result[0].message.tool_calls) == 1
    assert isinstance(result[0].message.tool_calls[0], ChatCompletionMessageToolCall)
    assert result[0].message.tool_calls[0].id == "call_abc123"
    assert result[0].message.tool_calls[0].type == "function"
    assert result[0].message.tool_calls[0].function.name == "get_current_weather"
    assert (
        result[0].message.tool_calls[0].function.arguments
        == '{\n"location": "Boston, MA"\n}'
    )

    assert result[0].finish_reason == "tool_calls"


def test_json_mode_conversion():
    """
    JSON Mode Conversion

    Should add the function arguments to the content
    """
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
                                "name": "json_output",
                                "arguments": '{"key": "value"}',
                            },
                        }
                    ],
                },
                "finish_reason": None,
            }
        ]
    }
    result = _process_choices_in_response(
        response_object=response_object, convert_tool_call_to_json_mode=True
    )

    assert len(result) == 1
    assert isinstance(result[0], Choices)
    assert result[0].message.content == '{"key": "value"}'
    assert result[0].finish_reason == "stop"


def test_multiple_choices():
    """
    Handle responses with multiple choices
    """
    response_object = {
        "choices": [
            {
                "message": {"content": "First response", "role": "assistant"},
                "finish_reason": "stop",
            },
            {
                "message": {"content": "Second response", "role": "assistant"},
                "finish_reason": "length",
            },
        ]
    }
    _choices = _process_choices_in_response(response_object, False)

    assert len(_choices) == 2
    assert _choices[0].message.content == "First response"
    assert isinstance(_choices[0], Choices)
    assert _choices[0].finish_reason == "stop"

    # Second Choice
    assert _choices[1].message.content == "Second response"
    assert isinstance(_choices[1], Choices)
    assert _choices[1].finish_reason == "length"


def test_logprobs_and_enhancements():
    """
    Handle logprobs and enhancements in the choices
    """
    response_object = {
        "choices": [
            {
                "message": {"content": "Response with extras", "role": "assistant"},
                "finish_reason": "stop",
                "logprobs": {"token_logprobs": [0.1, 0.2, 0.3]},
                "enhancements": {"some_enhancement": True},
            }
        ]
    }
    result = _process_choices_in_response(response_object, False)

    assert len(result) == 1
    assert result[0].message.content == "Response with extras"
    assert result[0].logprobs == {"token_logprobs": [0.1, 0.2, 0.3]}
    assert result[0].enhancements == {"some_enhancement": True}


def test_empty_choices():
    """
    Should return an empty list if there are no choices
    """
    response_object = {"choices": []}
    result = _process_choices_in_response(response_object, False)

    assert len(result) == 0


def test_none_role():
    """
    Should default to 'assistant' if 'role' is None
    """
    response_object = {
        "choices": [
            {
                "message": {"content": "Response with None role", "role": None},
                "finish_reason": "stop",
            }
        ]
    }
    result = _process_choices_in_response(response_object, False)

    assert len(result) == 1
    assert result[0].message.role == "assistant"
    assert result[0].message.content == "Response with None role"
