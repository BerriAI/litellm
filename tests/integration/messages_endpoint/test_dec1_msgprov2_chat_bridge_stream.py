"""The Anthropic SSE event sequence the chat bridge builds from OpenAI chat chunks.

A streamed `POST /v1/messages` to a chat-only provider is answered by `AnthropicStreamWrapper`
(`litellm/llms/anthropic/messages/adapters/streaming_iterator.py`). Anthropic SDKs are strict about the
event grammar, so a Rust bridge must produce exactly: one `message_start` (with `content: []`, null stop
fields, an id starting `msg_` and zeroed usage), then for each block a `content_block_start`, its deltas and a
`content_block_stop` with the same index, then one `message_delta` carrying the mapped `stop_reason` and the
provider usage, then `message_stop` last. Text deltas are `text_delta`, tool arguments are
`input_json_delta` fragments that concatenate to the provider's argument string, and `reasoning_content`
becomes a `thinking` block with `thinking_delta` before the text block.

The providers here are `hosted_vllm/` (a generic chat-only server) and Baseten. Both are PYTHON_ONLY on
`/v1/messages` today (`litellm/rust_bridge/catalog.py`).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Final

import pytest
from integration._support.client import Gateway, object_value
from integration._support.wire import Reply, Request
from integration.messages_endpoint.dec1_messages_support import Rig, sse_events
from integration.messages_endpoint.dec1_msgprov2_support import (
    HOSTED_VLLM_BACKEND,
    HOSTED_VLLM_KEY,
    INITIAL_STREAM_USAGE,
    blocks_by_index,
    chat_stream,
    collapsed,
    deltas_of,
    event_names,
    the_event,
)
from integration.messages_endpoint.dec1_msgprov_support import (
    BASETEN_BACKEND,
    BASETEN_KEY,
    CACHE_READ_COST,
    INPUT_COST,
    OUTPUT_COST,
    PRICES,
    TOOL,
    only_spend_row,
    provider_deployment,
    spend_of,
)
from pydantic import JsonValue

_TEXT_STREAM: Final = chat_stream(
    HOSTED_VLLM_BACKEND,
    (
        ({"role": "assistant", "content": ""}, None),
        ({"content": "hello "}, None),
        ({"content": "stream"}, None),
        ({}, "stop"),
    ),
)


def _vllm(gateway: Gateway, respond: Callable[[Request], Reply], **params: JsonValue) -> AbstractContextManager[Rig]:
    return provider_deployment(
        gateway,
        respond,
        model=f"hosted_vllm/{HOSTED_VLLM_BACKEND}",
        base_suffix="/v1",
        api_key=HOSTED_VLLM_KEY,
        **params,
    )


def _baseten(gateway: Gateway, respond: Callable[[Request], Reply], **params: JsonValue) -> AbstractContextManager[Rig]:
    return provider_deployment(gateway, respond, model=f"baseten/{BASETEN_BACKEND}", api_key=BASETEN_KEY, **params)


def _events(rig: Rig, **fields: JsonValue) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    response, text = rig.stream(rig.body(stream=True, **fields))
    assert response.status_code == 200, text
    assert response.headers["content-type"].startswith("text/event-stream"), response.headers
    return sse_events(text)


def _tool_stream(arguments: tuple[str, ...], *, leading_text: str | None = None) -> Reply:
    opening: Final[list[tuple[dict[str, JsonValue], str | None]]] = [({"role": "assistant", "content": ""}, None)]
    text_part: Final[list[tuple[dict[str, JsonValue], str | None]]] = (
        [({"content": leading_text}, None)] if leading_text is not None else []
    )
    first_call: Final[dict[str, JsonValue]] = {
        "tool_calls": [
            {
                "index": 0,
                "id": "call_stream01",
                "type": "function",
                "function": {"name": "get_weather", "arguments": ""},
            }
        ]
    }
    fragments: Final[list[tuple[dict[str, JsonValue], str | None]]] = [
        ({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}, None) for piece in arguments
    ]
    return chat_stream(HOSTED_VLLM_BACKEND, (*opening, *text_part, (first_call, None), *fragments, ({}, "tool_calls")))


# ---------------------------------------------------------------- Grammar


def test_a_text_stream_follows_the_anthropic_event_grammar(gateway: Gateway) -> None:
    """Chat chunks with role, two text pieces, a `stop` finish and a usage chunk become, with repeated deltas
    folded, exactly `message_start, content_block_start, content_block_delta, content_block_stop, message_delta,
    message_stop`. Anthropic SDKs raise on any other order."""
    with _vllm(gateway, lambda _r: _TEXT_STREAM) as rig:
        names: Final = event_names(_events(rig))
        assert collapsed(names) == (
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ), names


def test_message_start_has_an_empty_message_and_zeroed_usage(gateway: Gateway) -> None:
    """The first event is `message_start` whose `message` has `type: "message"`, `role: "assistant"`,
    `content: []`, `stop_reason: null`, `stop_sequence: null`, an `id` starting `msg_`, and usage
    `{input_tokens: 0, output_tokens: 0, cache_creation_input_tokens: 0, cache_read_input_tokens: 0}`: the
    bridge knows no usage until the provider's usage chunk, which lands on `message_delta`."""
    with _vllm(gateway, lambda _r: _TEXT_STREAM) as rig:
        events: Final = _events(rig)
        assert events[0][0] == "message_start"
        message: Final = object_value(events[0][1]["message"])
        assert (message["type"], message["role"], message["content"]) == ("message", "assistant", [])
        assert (message["stop_reason"], message["stop_sequence"]) == (None, None)
        assert str(message["id"]).startswith("msg_")
        assert message["usage"] == dict(INITIAL_STREAM_USAGE)


def test_text_arrives_as_text_deltas_on_block_zero(gateway: Gateway) -> None:
    """Block 0 starts as `{"type": "text", "text": ""}` and its `text_delta` pieces concatenate to
    `hello stream`; every delta on it has `type: "text_delta"`."""
    with _vllm(gateway, lambda _r: _TEXT_STREAM) as rig:
        events: Final = _events(rig)
        assert blocks_by_index(events) == {0: {"type": "text", "text": ""}}
        deltas: Final = deltas_of(events, 0)
        assert {delta["type"] for delta in deltas} == {"text_delta"}
        assert "".join(str(delta["text"]) for delta in deltas) == "hello stream"


def test_message_delta_carries_the_stop_reason_and_provider_usage(gateway: Gateway) -> None:
    """The single `message_delta` has `delta.stop_reason: "end_turn"` and usage with `input_tokens: 11` and
    `output_tokens: 7` from the provider's usage chunk, so a client that bills from the stream sees the real
    numbers, not the zeros of `message_start`."""
    with _vllm(gateway, lambda _r: _TEXT_STREAM) as rig:
        delta_event: Final = the_event(_events(rig), "message_delta")
        assert object_value(delta_event["delta"])["stop_reason"] == "end_turn"
        usage: Final = object_value(delta_event["usage"])
        assert (usage["input_tokens"], usage["output_tokens"]) == (11, 7)


def test_cached_prompt_tokens_appear_on_message_delta_as_cache_reads(gateway: Gateway) -> None:
    """A usage chunk with `prompt_tokens: 20` and `prompt_tokens_details.cached_tokens: 8` gives `message_delta`
    usage `input_tokens: 12` and `cache_read_input_tokens: 8`, the same split as the unary answer."""
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_tokens_details": {"cached_tokens": 8},
    }
    reply: Final = chat_stream(
        HOSTED_VLLM_BACKEND, (({"role": "assistant", "content": "cached"}, None), ({}, "stop")), usage=usage
    )
    with _vllm(gateway, lambda _r: reply) as rig:
        streamed: Final = object_value(the_event(_events(rig), "message_delta")["usage"])
        assert (streamed["input_tokens"], streamed["cache_read_input_tokens"], streamed["output_tokens"]) == (12, 8, 5)


def test_every_block_start_has_a_matching_stop_and_message_stop_is_last(gateway: Gateway) -> None:
    """For a text then tool call stream, each `content_block_start` index has exactly one `content_block_stop`
    with the same index, every stop comes before `message_delta`, and `message_stop` is the last event and
    appears once."""
    reply: Final = _tool_stream(('{"city": ', '"Lisbon"}'), leading_text="checking")
    with _vllm(gateway, lambda _r: reply) as rig:
        events: Final = _events(rig, tools=[TOOL])
        names: Final = event_names(events)
        starts: Final = sorted(int(str(data["index"])) for name, data in events if name == "content_block_start")
        stops: Final = sorted(int(str(data["index"])) for name, data in events if name == "content_block_stop")
        assert starts == stops == [0, 1], names
        last_stop: Final = max(i for i, name in enumerate(names) if name == "content_block_stop")
        assert last_stop < names.index("message_delta")
        assert names[-1] == "message_stop" and names.count("message_stop") == 1


# ---------------------------------------------------------------- Tool use, thinking and stop reasons


def test_a_streamed_tool_call_becomes_a_tool_use_block_with_json_deltas(gateway: Gateway) -> None:
    """A tool call streamed as an opening chunk (id, name, empty arguments) and two argument fragments becomes a
    `content_block_start` with `{"type": "tool_use", "id": "call_stream01", "name": "get_weather", "input": {}}`
    and `input_json_delta` events whose `partial_json` concatenates to `{"city": "Lisbon"}`; `message_delta` has
    `stop_reason: "tool_use"`."""
    reply: Final = _tool_stream(('{"city": ', '"Lisbon"}'))
    with _vllm(gateway, lambda _r: reply) as rig:
        events: Final = _events(rig, tools=[TOOL])
        tool_blocks: Final = {
            index: block for index, block in blocks_by_index(events).items() if block["type"] == "tool_use"
        }
        assert len(tool_blocks) == 1, events
        index, block = next(iter(tool_blocks.items()))
        assert block == {"type": "tool_use", "id": "call_stream01", "name": "get_weather", "input": {}}
        fragments: Final = deltas_of(events, index)
        assert {delta["type"] for delta in fragments} == {"input_json_delta"}
        assert json.loads("".join(str(delta["partial_json"]) for delta in fragments)) == {"city": "Lisbon"}
        assert object_value(the_event(events, "message_delta")["delta"])["stop_reason"] == "tool_use"


def test_text_before_a_tool_call_is_block_zero_and_the_tool_is_block_one(gateway: Gateway) -> None:
    """With `checking` streamed before the tool call, block 0 is the text block and block 1 is the `tool_use`
    block: indices follow arrival order and never reuse an index."""
    reply: Final = _tool_stream(('{"city": "Oslo"}',), leading_text="checking")
    with _vllm(gateway, lambda _r: reply) as rig:
        blocks: Final = blocks_by_index(_events(rig, tools=[TOOL]))
        assert (blocks[0]["type"], blocks[1]["type"]) == ("text", "tool_use")


def test_streamed_reasoning_becomes_a_thinking_block_before_the_text(gateway: Gateway) -> None:
    """Chunks carrying `reasoning_content: "step one"` then `content: "answer"` become block 0 of type `thinking`
    whose `thinking_delta` pieces give `step one`, then block 1 of type `text` with `answer`. UNVERIFIED: the
    exact `content_block_start` fields of the thinking block (`thinking: ""`, and whether `signature` is
    present), so only the type is pinned."""
    reply: Final = chat_stream(
        HOSTED_VLLM_BACKEND,
        (
            ({"role": "assistant", "content": ""}, None),
            ({"reasoning_content": "step "}, None),
            ({"reasoning_content": "one"}, None),
            ({"content": "answer"}, None),
            ({}, "stop"),
        ),
    )
    with _vllm(gateway, lambda _r: reply) as rig:
        events: Final = _events(rig)
        blocks: Final = blocks_by_index(events)
        assert (blocks[0]["type"], blocks[1]["type"]) == ("thinking", "text"), events
        thinking: Final = [delta for delta in deltas_of(events, 0) if delta["type"] == "thinking_delta"]
        assert "".join(str(delta["thinking"]) for delta in thinking) == "step one"
        assert "".join(str(delta.get("text", "")) for delta in deltas_of(events, 1)) == "answer"


@pytest.mark.parametrize(("finish", "stop_reason"), [("length", "max_tokens"), ("content_filter", "refusal")])
def test_the_stream_finish_reason_maps_on_message_delta(gateway: Gateway, finish: str, stop_reason: str) -> None:
    """A stream that ends with finish `length` gives `message_delta.delta.stop_reason: "max_tokens"`, and
    `content_filter` gives `"refusal"`, the same mapping as the unary answer."""
    reply: Final = chat_stream(HOSTED_VLLM_BACKEND, (({"role": "assistant", "content": "partial"}, None), ({}, finish)))
    with _vllm(gateway, lambda _r: reply) as rig:
        assert object_value(the_event(_events(rig), "message_delta")["delta"])["stop_reason"] == stop_reason


def test_a_baseten_stream_ends_with_message_delta_then_message_stop(gateway: Gateway) -> None:
    """A Baseten chat stream ends, on the caller side, with exactly `message_delta` then `message_stop`, and
    `message_delta.usage.output_tokens` is 7 from Baseten's usage chunk."""
    reply: Final = chat_stream(BASETEN_BACKEND, (({"role": "assistant", "content": "hi"}, None), ({}, "stop")))
    with _baseten(gateway, lambda _r: reply) as rig:
        events: Final = _events(rig)
        assert event_names(events)[-2:] == ("message_delta", "message_stop")
        assert object_value(events[-2][1]["usage"])["output_tokens"] == 7


# ---------------------------------------------------------------- Spend


def test_a_streamed_bridge_call_with_cached_tokens_is_billed_with_the_cache_rate(gateway: Gateway) -> None:
    """A streamed hosted vLLM call whose usage chunk reports 20 prompt tokens with 8 cached and 5 completion
    tokens writes one `anthropic_messages` row with `custom_llm_provider = "hosted_vllm"` and
    `spend = 12 * 0.001 + 8 * 0.0001 + 5 * 0.002 = 0.0228`."""
    usage: Final[dict[str, JsonValue]] = {
        "prompt_tokens": 20,
        "completion_tokens": 5,
        "total_tokens": 25,
        "prompt_tokens_details": {"cached_tokens": 8},
    }
    reply: Final = chat_stream(
        HOSTED_VLLM_BACKEND, (({"role": "assistant", "content": "ok"}, None), ({}, "stop")), usage=usage
    )
    with _vllm(gateway, lambda _r: reply, **PRICES) as rig:
        _events(rig)
        row: Final = only_spend_row(rig)
        assert (row["call_type"], row["custom_llm_provider"]) == ("anthropic_messages", "hosted_vllm")
        assert spend_of(row) == pytest.approx(12 * INPUT_COST + 8 * CACHE_READ_COST + 5 * OUTPUT_COST)


def test_a_streamed_tool_call_is_billed_once(gateway: Gateway) -> None:
    """A streamed tool call answer with the default usage chunk (11 and 7 tokens) writes exactly one row with
    spend 0.025, even though the bridge emits more Anthropic events than the provider sent chunks."""
    reply: Final = _tool_stream(('{"city": "Oslo"}',))
    with _vllm(gateway, lambda _r: reply, **PRICES) as rig:
        _events(rig, tools=[TOOL])
        assert spend_of(only_spend_row(rig)) == pytest.approx(11 * INPUT_COST + 7 * OUTPUT_COST)
