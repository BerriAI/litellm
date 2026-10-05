"""
Regression tests for https://github.com/BerriAI/litellm/issues/44392

A provider (or relay) may stream a tool call's function name and id in more
than one fragment. ``ChunkProcessor.get_combined_tool_content`` used to
overwrite ``id`` and ``name`` with each new non-empty fragment, so the
rebuilt response kept only the last fragment (``call_`` + ``9f2c`` came back
as ``9f2c``), while ``arguments`` were joined correctly. The rebuilt response
is what success callbacks, observability integrations and spend logs see, and
what SDK users send back with the tool result.

These tests pin:

1. split fragments are joined (``call_`` + ``9f2c`` -> ``call_9f2c``);
2. a value repeated whole on every chunk is not duplicated;
3. parallel tool calls (distinct indexes) stay separate.
"""

import pytest

from litellm import stream_chunk_builder


def _chunk(delta_tool_calls, finish_reason=None):
    return {
        "id": "chatcmpl-fragment-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "fragmenting-server",
        "choices": [
            {
                "index": 0,
                "delta": {"tool_calls": delta_tool_calls},
                "finish_reason": finish_reason,
            }
        ],
    }


def _chunks_for_split_call():
    return [
        _chunk(
            [
                {
                    "index": 0,
                    "id": "call_",
                    "type": "function",
                    "function": {"name": "get_", "arguments": ""},
                }
            ]
        ),
        _chunk(
            [
                {
                    "index": 0,
                    "id": "9f2c",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                }
            ],
            finish_reason="tool_calls",
        ),
    ]


def test_split_tool_call_fragments_are_joined():
    chunks = _chunks_for_split_call()

    response = stream_chunk_builder(chunks)

    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.id == "call_9f2c"
    assert tool_call.function.name == "get_weather"
    assert tool_call.function.arguments == '{"city": "Paris"}'


def test_whole_value_repeated_on_every_chunk_is_not_duplicated():
    chunks = [
        _chunk(
            [
                {
                    "index": 0,
                    "id": "call_9f2c",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": ""},
                }
            ]
        ),
        _chunk(
            [
                {
                    "index": 0,
                    "id": "call_9f2c",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
            finish_reason="tool_calls",
        ),
    ]

    response = stream_chunk_builder(chunks)

    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.id == "call_9f2c"
    assert tool_call.function.name == "get_weather"


def test_parallel_tool_calls_keep_separate_ids_and_names():
    chunks = [
        _chunk(
            [
                {
                    "index": 0,
                    "id": "call_a",
                    "type": "function",
                    "function": {"name": "get_", "arguments": ""},
                },
                {
                    "index": 1,
                    "id": "call_",
                    "type": "function",
                    "function": {"name": "put_", "arguments": ""},
                },
            ]
        ),
        _chunk(
            [
                {
                    "index": 0,
                    "id": "1",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Paris"}'},
                },
                {
                    "index": 1,
                    "id": "b2",
                    "type": "function",
                    "function": {"name": "weather", "arguments": '{"city": "Berlin"}'},
                },
            ],
            finish_reason="tool_calls",
        ),
    ]

    response = stream_chunk_builder(chunks)

    first, second = response.choices[0].message.tool_calls
    assert (first.id, first.function.name) == ("call_a1", "get_weather")
    assert (second.id, second.function.name) == ("call_b2", "put_weather")
