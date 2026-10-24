"""Unit tests for stream_chunk_builder and Delta tool call handling."""

from litellm import stream_chunk_builder
from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta,
    Function,
    ModelResponse,
)


def test_stream_chunk_builder_single_dict_tool_call_coercion() -> None:
    """Test that stream_chunk_builder properly handles delta.tool_calls as a single dict."""
    chunk = {
        "id": "chatcmpl-test-single",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": {
                        "index": 0,
                        "id": "call_single_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"location": "Dhaka"}',
                        },
                    },
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    rebuilt_response = stream_chunk_builder(chunks=[chunk])

    assert isinstance(rebuilt_response, ModelResponse)
    assert len(rebuilt_response.choices) > 0
    tool_calls = rebuilt_response.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_single_1"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"location": "Dhaka"}'


def test_delta_single_object_tool_calls_coercion() -> None:
    """Test that Delta.__init__ coerces single object / dict tool_calls into a list."""
    delta_from_dict = Delta(tool_calls={"id": "call_1", "function": {"name": "test"}})
    assert delta_from_dict.tool_calls is not None
    assert isinstance(delta_from_dict.tool_calls, list)
    assert len(delta_from_dict.tool_calls) == 1
    assert delta_from_dict.tool_calls[0].id == "call_1"
    assert delta_from_dict.tool_calls[0].function.name == "test"

    tool_call_obj = ChatCompletionDeltaToolCall(
        id="call_2",
        function=Function(name="test_obj", arguments="{}"),
        type="function",
        index=0,
    )
    delta_from_obj = Delta(tool_calls=tool_call_obj)
    assert delta_from_obj.tool_calls is not None
    assert isinstance(delta_from_obj.tool_calls, list)
    assert len(delta_from_obj.tool_calls) == 1
    assert delta_from_obj.tool_calls[0].id == "call_2"


def test_stream_chunk_builder_list_tool_call_coercion() -> None:
    """Verify existing list-based tool call chunks continue to build correctly."""
    chunk = {
        "id": "chatcmpl-test-list",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "call_list_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "Dhaka"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    rebuilt_response = stream_chunk_builder(chunks=[chunk])

    assert isinstance(rebuilt_response, ModelResponse)
    assert len(rebuilt_response.choices) > 0
    tool_calls = rebuilt_response.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_list_1"
    assert tool_calls[0].function.name == "get_weather"


def test_stream_chunk_builder_fragmented_single_dict_tool_call() -> None:
    """Verify fragmented chunks with single dict tool_calls correctly reconstruct function arguments."""
    chunks = [
        {
            "id": "chatcmpl-frag-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": {
                            "index": 0,
                            "id": "call_frag_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "",
                            },
                        },
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-frag-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": {
                            "index": 0,
                            "function": {
                                "arguments": '{"location": ',
                            },
                        },
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-frag-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": {
                            "index": 0,
                            "function": {
                                "arguments": '"Dhaka"}',
                            },
                        },
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        },
    ]

    rebuilt_response = stream_chunk_builder(chunks=chunks)

    assert isinstance(rebuilt_response, ModelResponse)
    assert len(rebuilt_response.choices) > 0
    tool_calls = rebuilt_response.choices[0].message.tool_calls
    assert tool_calls is not None
    assert len(tool_calls) == 1
    assert tool_calls[0].id == "call_frag_1"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"location": "Dhaka"}'
