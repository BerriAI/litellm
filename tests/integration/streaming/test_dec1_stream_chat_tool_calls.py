"""Streamed tool calls on `/v1/chat/completions`.

OpenAI streams a tool call as fragments: the first fragment of a call carries `index`, `id`,
`type: function` and `function.name`; later fragments carry only `index` and a piece of
`function.arguments`. Clients rebuild calls by `index`, so the proxy has to keep each fragment's index,
send `id` and `name` once, keep argument pieces byte exact, and end with `finish_reason: tool_calls`.
The existing SDK level contract (`test_fragmented_tool_names_and_arguments_keep_each_call_identity`)
calls `litellm.completion` directly; these go through the proxy's HTTP surface.
"""

from __future__ import annotations

import json
from typing import Final

from integration._support.client import Gateway, object_value
from integration.streaming.dec1_stream_support import (
    DONE,
    chunks_of,
    content_of,
    delta,
    deltas_of,
    finish_reasons,
    joined_arguments,
    marker,
    scripted,
    serving,
    tool_call_fragments,
)

_TOOLS: Final = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Look up the weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Look up the time",
            "parameters": {"type": "object", "properties": {"zone": {"type": "string"}}},
        },
    },
]


def _open(index: int, call_id: str, name: str) -> dict:
    return {"index": index, "id": call_id, "type": "function", "function": {"name": name, "arguments": ""}}


def _piece(index: int, arguments: str) -> dict:
    return {"index": index, "function": {"arguments": arguments}}


def _single_call(identity: str, pieces: tuple[str, ...], *, finish: str = "tool_calls") -> tuple[bytes, ...]:
    return (
        delta(identity, {"role": "assistant", "content": None, "tool_calls": [_open(0, "call_dec1_weather", "get_weather")]}),
        *(delta(identity, {"tool_calls": [_piece(0, piece)]}) for piece in pieces),
        delta(identity, {}, finish=finish),
        DONE,
    )


def test_single_tool_call_streams_id_and_name_once_then_argument_pieces(gateway: Gateway) -> None:
    """A provider streams one tool call: an opening fragment, then `{"city":` and `"Paris"}`. The caller
    sees `id == call_dec1_weather`, `type == function` and `name == get_weather` on the first fragment
    only, the argument pieces concatenate to `{"city":"Paris"}`, and the stream ends with
    `finish_reason: tool_calls`."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(_single_call(identity, ('{"city":', '"Paris"}')))) as rig:
        chunks: Final = chunks_of(rig.chat(tools=_TOOLS).text)
        fragments: Final = tool_call_fragments(chunks, 0)
        assert fragments[0]["id"] == "call_dec1_weather"
        assert fragments[0]["type"] == "function"
        assert object_value(fragments[0]["function"])["name"] == "get_weather"
        assert [item.get("id") for item in fragments[1:]] == [None] * (len(fragments) - 1)
        assert json.loads(joined_arguments(fragments)) == {"city": "Paris"}
        assert finish_reasons(chunks) == ["tool_calls"]


def test_tool_call_stream_carries_no_content_text(gateway: Gateway) -> None:
    """A provider streams only a tool call with `content: null`. No chunk the caller receives has a
    non empty `delta.content`. Agents treat any content text as a spoken answer."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(_single_call(identity, ('{"city":"Oslo"}',)))) as rig:
        assert content_of(chunks_of(rig.chat(tools=_TOOLS).text)) == ""


def test_argument_fragments_are_byte_exact_including_escapes_and_whitespace(gateway: Gateway) -> None:
    """A provider splits arguments inside a JSON escape and inside a multibyte character:
    `{"city": "São \\"Pa`, `ulo\\"\\n"`, ` }`. The caller's fragments are those same three strings, so
    the concatenation parses to the city `São "Paulo"\\n`. Re-serialising arguments per fragment would
    corrupt them because a fragment is not valid JSON on its own."""
    identity: Final = f"chatcmpl-{marker()}"
    pieces: Final = ('{"city": "São \\"Pa', 'ulo\\"\\n"', " }")
    with scripted(gateway, serving(_single_call(identity, pieces))) as rig:
        fragments: Final = tool_call_fragments(chunks_of(rig.chat(tools=_TOOLS).text), 0)
        streamed: Final = [object_value(item["function"]).get("arguments") for item in fragments]
        assert [piece for piece in streamed if piece] == list(pieces)
        assert json.loads(joined_arguments(fragments)) == {"city": 'São "Paulo"\n'}


def test_parallel_tool_calls_keep_their_indexes_ids_and_arguments_apart(gateway: Gateway) -> None:
    """A provider streams two calls interleaved: index 0 `get_weather`, index 1 `get_time`, then argument
    pieces alternating between them. Rebuilding by `index` gives call 0 with id `call_w` and arguments
    `{"city":"Rome"}` and call 1 with id `call_t` and arguments `{"zone":"CET"}`. The stream ends with one
    `tool_calls` finish."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "tool_calls": [_open(0, "call_w", "get_weather")]}),
        delta(identity, {"tool_calls": [_open(1, "call_t", "get_time")]}),
        delta(identity, {"tool_calls": [_piece(0, '{"city":')]}),
        delta(identity, {"tool_calls": [_piece(1, '{"zone":')]}),
        delta(identity, {"tool_calls": [_piece(0, '"Rome"}')]}),
        delta(identity, {"tool_calls": [_piece(1, '"CET"}')]}),
        delta(identity, {}, finish="tool_calls"),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat(tools=_TOOLS).text)
        for index, call_id, name, arguments in ((0, "call_w", "get_weather", {"city": "Rome"}), (1, "call_t", "get_time", {"zone": "CET"})):
            fragments = tool_call_fragments(chunks, index)
            assert [item["id"] for item in fragments if item.get("id")] == [call_id]
            assert "".join(str(object_value(item["function"]).get("name") or "") for item in fragments) == name
            assert json.loads(joined_arguments(fragments)) == arguments
        assert finish_reasons(chunks) == ["tool_calls"]


def test_two_tool_calls_opened_in_one_provider_chunk_both_reach_the_caller(gateway: Gateway) -> None:
    """A provider opens two calls inside a single chunk's `tool_calls` array, each with complete
    arguments. The caller receives both calls with indexes 0 and 1, their own ids, and complete
    arguments. A translator that reads only `tool_calls[0]` loses the second call."""
    identity: Final = f"chatcmpl-{marker()}"
    both: Final = [
        {**_open(0, "call_first", "get_weather"), "function": {"name": "get_weather", "arguments": '{"city":"Lima"}'}},
        {**_open(1, "call_second", "get_time"), "function": {"name": "get_time", "arguments": '{"zone":"PET"}'}},
    ]
    frames: Final = (delta(identity, {"role": "assistant", "tool_calls": both}), delta(identity, {}, finish="tool_calls"), DONE)
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat(tools=_TOOLS).text)
        assert [item["id"] for item in tool_call_fragments(chunks, 0) if item.get("id")] == ["call_first"]
        assert [item["id"] for item in tool_call_fragments(chunks, 1) if item.get("id")] == ["call_second"]
        assert json.loads(joined_arguments(tool_call_fragments(chunks, 1))) == {"zone": "PET"}


def test_text_before_a_tool_call_arrives_before_the_first_tool_fragment(gateway: Gateway) -> None:
    """A provider streams the text `Let me check.` and then a tool call. The caller receives the text
    deltas first, then the tool call fragments, then `finish_reason: tool_calls`; text and tool fragments
    never share a delta."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "content": "Let me "}),
        delta(identity, {"content": "check."}),
        delta(identity, {"tool_calls": [_open(0, "call_after_text", "get_weather")]}),
        delta(identity, {"tool_calls": [_piece(0, '{"city":"Kyiv"}')]}),
        delta(identity, {}, finish="tool_calls"),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat(tools=_TOOLS).text)
        kinds: Final = ["text" if item.get("content") else "tool" for item in deltas_of(chunks) if item.get("content") or item.get("tool_calls")]
        assert kinds == ["text", "text", "tool", "tool"]
        assert content_of(chunks) == "Let me check."
        assert json.loads(joined_arguments(tool_call_fragments(chunks, 0))) == {"city": "Kyiv"}


def test_stop_after_tool_calls_is_reported_as_tool_calls(gateway: Gateway) -> None:
    """Some OpenAI compatible servers end a tool call stream with `finish_reason: stop`. The proxy has
    seen tool call deltas, so the caller's finish reason is `tool_calls`
    (`_normalize_tool_call_finish_reason`). Agent loops only execute tools on `tool_calls`."""
    identity: Final = f"chatcmpl-{marker()}"
    with scripted(gateway, serving(_single_call(identity, ('{"city":"Bern"}',), finish="stop"))) as rig:
        assert finish_reasons(chunks_of(rig.chat(tools=_TOOLS).text)) == ["tool_calls"]


def test_tools_and_tool_choice_reach_the_provider_unchanged(gateway: Gateway) -> None:
    """The caller streams with two tools, a forced `tool_choice` and `parallel_tool_calls: false`. The
    provider receives the same `tools` array, the same `tool_choice` object and the same flag."""
    identity: Final = f"chatcmpl-{marker()}"
    choice: Final = {"type": "function", "function": {"name": "get_weather"}}
    with scripted(gateway, serving(_single_call(identity, ('{"city":"Graz"}',)))) as rig:
        assert rig.chat(tools=_TOOLS, tool_choice=choice, parallel_tool_calls=False).status_code == 200
        body: Final = rig.upstream_bodies()[0]
        assert body["tools"] == _TOOLS
        assert body["tool_choice"] == choice
        assert body["parallel_tool_calls"] is False


def test_tool_call_with_empty_arguments_still_streams_a_complete_call(gateway: Gateway) -> None:
    """A provider streams a call to a no argument tool: one opening fragment whose arguments are `{}` and
    nothing else. The caller gets the id, the name, arguments that parse to an empty object, and
    `finish_reason: tool_calls`."""
    identity: Final = f"chatcmpl-{marker()}"
    opening: Final = {**_open(0, "call_noargs", "get_time"), "function": {"name": "get_time", "arguments": "{}"}}
    frames: Final = (delta(identity, {"role": "assistant", "tool_calls": [opening]}), delta(identity, {}, finish="tool_calls"), DONE)
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat(tools=_TOOLS).text)
        fragments: Final = tool_call_fragments(chunks, 0)
        assert fragments[0]["id"] == "call_noargs"
        assert json.loads(joined_arguments(fragments)) == {}
        assert finish_reasons(chunks) == ["tool_calls"]


def test_legacy_function_call_deltas_reach_the_caller(gateway: Gateway) -> None:
    """A provider streams the legacy `delta.function_call` shape (name once, then argument pieces) and
    ends with `finish_reason: function_call`. The caller receives `function_call` deltas whose arguments
    concatenate to `{"city":"Baku"}` and the same finish reason. UNVERIFIED: read from
    `Delta.function_call` handling in `litellm/types/utils.py`; older clients still depend on it."""
    identity: Final = f"chatcmpl-{marker()}"
    frames: Final = (
        delta(identity, {"role": "assistant", "content": None, "function_call": {"name": "get_weather", "arguments": ""}}),
        delta(identity, {"function_call": {"arguments": '{"city":'}}),
        delta(identity, {"function_call": {"arguments": '"Baku"}'}}),
        delta(identity, {}, finish="function_call"),
        DONE,
    )
    with scripted(gateway, serving(frames)) as rig:
        chunks: Final = chunks_of(rig.chat().text)
        calls: Final = [object_value(item["function_call"]) for item in deltas_of(chunks) if item.get("function_call")]
        assert calls[0].get("name") == "get_weather"
        assert json.loads("".join(str(call.get("arguments") or "") for call in calls)) == {"city": "Baku"}
        assert finish_reasons(chunks) == ["function_call"]
