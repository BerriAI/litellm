"""Unit tests for stream_chunk_builder and Delta tool call handling."""

from litellm import stream_chunk_builder
from litellm.litellm_core_utils.streaming_chunk_builder_utils import ChunkProcessor
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

    class CustomToolCall:
        pass

    custom_tc = CustomToolCall()
    delta_from_custom = Delta(tool_calls=custom_tc)
    assert delta_from_custom.tool_calls is not None
    assert isinstance(delta_from_custom.tool_calls, list)
    assert len(delta_from_custom.tool_calls) == 1
    assert delta_from_custom.tool_calls[0] is custom_tc


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


def test_stream_chunk_builder_single_object_tool_call_coercion() -> None:
    """Test that stream_chunk_builder handles delta.tool_calls as a single ChatCompletionDeltaToolCall object."""
    tool_call_obj = ChatCompletionDeltaToolCall(
        index=0,
        id="call_obj_1",
        type="function",
        function=Function(name="get_weather", arguments='{"location": "Dhaka"}'),
    )
    chunk = {
        "id": "chatcmpl-test-single-obj",
        "object": "chat.completion.chunk",
        "created": 1234567890,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "delta": {
                    "role": "assistant",
                    "tool_calls": tool_call_obj,
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
    assert tool_calls[0].id == "call_obj_1"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"location": "Dhaka"}'


def test_stream_chunk_builder_fragmented_single_object_tool_call() -> None:
    """Verify fragmented chunks with single object tool_calls correctly reconstruct function arguments."""
    chunks = [
        {
            "id": "chatcmpl-frag-obj-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": ChatCompletionDeltaToolCall(
                            index=0,
                            id="call_frag_obj_1",
                            type="function",
                            function=Function(name="get_weather", arguments=""),
                        ),
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-frag-obj-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": ChatCompletionDeltaToolCall(
                            index=0,
                            function=Function(arguments='{"location": '),
                        ),
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            "id": "chatcmpl-frag-obj-1",
            "object": "chat.completion.chunk",
            "created": 1234567890,
            "model": "gpt-4",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": ChatCompletionDeltaToolCall(
                            index=0,
                            function=Function(arguments='"Dhaka"}'),
                        ),
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
    assert tool_calls[0].id == "call_frag_obj_1"
    assert tool_calls[0].function.name == "get_weather"
    assert tool_calls[0].function.arguments == '{"location": "Dhaka"}'


def test_get_combined_tool_content_direct_single_dict_and_object() -> None:
    """Directly test get_combined_tool_content with a single dict and single object tool call."""
    # Single dict tool_call
    dict_chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": {
                        "index": 0,
                        "id": "call_direct_dict",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": '{"query": "litellm"}',
                        },
                    }
                },
            }
        ]
    }
    processor = ChunkProcessor(chunks=[dict_chunk])
    result_dict = processor.get_combined_tool_content([dict_chunk])
    assert len(result_dict) == 1
    assert result_dict[0].id == "call_direct_dict"
    assert result_dict[0].function.name == "lookup"
    assert result_dict[0].function.arguments == '{"query": "litellm"}'

    # Single object tool_call
    obj_chunk = {
        "choices": [
            {
                "index": 0,
                "delta": {
                    "tool_calls": ChatCompletionDeltaToolCall(
                        index=0,
                        id="call_direct_obj",
                        type="function",
                        function=Function(name="search", arguments='{"q": "test"}'),
                    )
                },
            }
        ]
    }
    result_obj = processor.get_combined_tool_content([obj_chunk])
    assert len(result_obj) == 1
    assert result_obj[0].id == "call_direct_obj"
    assert result_obj[0].function.name == "search"
    assert result_obj[0].function.arguments == '{"q": "test"}'
