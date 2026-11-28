"""Contract: how an Anthropic SSE stream becomes OpenAI `chat.completion.chunk` events on `/v1/chat/completions`.

`ModelResponseIterator.chunk_parser` (`litellm/llms/anthropic/chat/handler.py:1006`) reads Anthropic's named events
(`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`,
`ping`, `error`) and emits OpenAI chunks: text deltas as `delta.content`, `tool_use` blocks as tool call fragments
keyed by tool index, thinking deltas as `delta.reasoning_content`, `message_delta.stop_reason` as `finish_reason`,
and usage from `message_start` plus `message_delta`. The caller's stream ends with `data: [DONE]`.

Gaps targeted here (the basic text, single tool call, thinking and cache read cases are already pinned in
`streaming/test_dec1_stream_provider_translation.py`): parallel tool calls, stop reasons other than end_turn, ping
frames, mid-stream provider errors, server tool use, JSON mode, the upstream request flags and the spend row.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from integration._support.client import Gateway, list_value, object_value, string_value
from integration.providers.dec1_anthropic_support import (
    CALL_COST,
    MESSAGE_STOP,
    PING,
    REPLY_TEXT,
    SIGNATURE,
    block_delta,
    block_start,
    block_stop,
    chat_stream_lines,
    data_events,
    deltas,
    deployment,
    finish_reasons,
    marker,
    message_delta,
    message_start,
    peer,
    spend_row_by_group,
    sse,
    streamed_text,
    streaming,
    tool,
    user,
)
from pydantic import JsonValue


def _body(model: str, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [user()], "stream": True, **extra}


def _tool_calls(events: tuple[dict[str, JsonValue], ...]) -> list[dict[str, JsonValue]]:
    return [object_value(call) for delta in deltas(events) for call in list_value(delta.get("tool_calls") or [])]


def test_a_streamed_request_is_sent_upstream_with_stream_true(gateway: Gateway) -> None:
    """The caller streams a chat call with `stream_options: {"include_usage": true}`.

    The upstream body must carry `stream: true` and must not carry `stream_options`, which Anthropic does not know.
    """
    with peer() as provider, gateway.scenario() as scenario:
        chat_stream_lines(gateway, _body(deployment(scenario, provider), stream_options={"include_usage": True}))
        body: Final = provider.only().body
        assert body["stream"] is True
        assert "stream_options" not in body


def test_text_deltas_concatenate_to_the_reply_and_the_stream_ends_with_done(gateway: Gateway) -> None:
    """The provider streams three text deltas `dec1 `, `anthropic `, `reply` and `end_turn`.

    The caller's `delta.content` pieces must join to `dec1 anthropic reply`, every chunk must have `object:
    chat.completion.chunk`, the only finish reason is `stop`, and the last line is `data: [DONE]`.
    """
    with peer() as provider, gateway.scenario() as scenario:
        headers, lines = chat_stream_lines(gateway, _body(deployment(scenario, provider)))
        events: Final = data_events(lines)
        assert headers["content-type"].startswith("text/event-stream")
        assert streamed_text(events) == REPLY_TEXT
        assert {event.get("object") for event in events if event.get("choices")} == {"chat.completion.chunk"}
        assert finish_reasons(events) == ("stop",)
        assert lines[-1].strip() == "data: [DONE]"


def test_include_usage_adds_the_input_and_output_tokens_from_start_and_delta(gateway: Gateway) -> None:
    """The provider reports `input_tokens: 11` in `message_start` and `output_tokens: 7` in `message_delta`.

    With `stream_options.include_usage` the caller must get a chunk whose usage is `prompt_tokens: 11`,
    `completion_tokens: 7`, `total_tokens: 18`: the input count from the start event survives the cumulative delta.
    """
    with peer() as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(
            gateway, _body(deployment(scenario, provider), stream_options={"include_usage": True})
        )
        usages: Final = [object_value(e["usage"]) for e in data_events(lines) if e.get("usage")]
        assert usages
        final: Final = usages[-1]
        assert (final["prompt_tokens"], final["completion_tokens"], final["total_tokens"]) == (11, 7, 18)


@pytest.mark.parametrize(("stop_reason", "finish"), [("max_tokens", "length"), ("stop_sequence", "stop")])
def test_the_streamed_stop_reason_is_mapped_on_the_finish_chunk(
    gateway: Gateway, stop_reason: str, finish: str
) -> None:
    """The provider streams one text delta and ends with `message_delta.stop_reason` max_tokens or stop_sequence.

    The caller's one finish reason must be `length` for max_tokens and `stop` for stop_sequence, as non-streaming.
    """
    chunks: Final = (
        message_start(),
        block_start(0, {"type": "text", "text": ""}),
        block_delta(0, {"type": "text_delta", "text": "partial"}),
        block_stop(0),
        message_delta(stop_reason, stop_sequence="END" if stop_reason == "stop_sequence" else None),
        MESSAGE_STOP,
    )
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(gateway, _body(deployment(scenario, provider)))
        events: Final = data_events(lines)
        assert finish_reasons(events) == (finish,)
        assert streamed_text(events) == "partial"


def test_ping_frames_are_ignored(gateway: Gateway) -> None:
    """The provider interleaves three `ping` events with a text stream.

    The caller's text must still be exactly `hello` and no chunk may carry ping content; a port that relays pings
    as empty chunks or fails on them breaks long thinking streams that ping while waiting.
    """
    chunks: Final = (
        PING,
        message_start(),
        PING,
        block_start(0, {"type": "text", "text": ""}),
        block_delta(0, {"type": "text_delta", "text": "hello"}),
        PING,
        block_stop(0),
        message_delta(),
        MESSAGE_STOP,
    )
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(gateway, _body(deployment(scenario, provider)))
        assert streamed_text(data_events(lines)) == "hello"
        assert not any("ping" in line for line in lines)


def test_parallel_streamed_tool_use_blocks_become_indexed_tool_call_fragments(gateway: Gateway) -> None:
    """The provider streams two `tool_use` blocks (`toolu_dec1a` get_weather at content index 0, `toolu_dec1b`
    get_time at content index 1), each input in two `input_json_delta` pieces, then `stop_reason: tool_use`.

    The caller's fragments must carry tool index 0 for the first and 1 for the second, the id and name on each
    tool's first fragment, arguments that concatenate per index to `{"city":"Paris"}` and `{"city":"Rome"}`, and
    the finish reason `tool_calls`.
    """
    chunks: Final = (
        message_start(),
        block_start(0, {"type": "tool_use", "id": "toolu_dec1a", "name": "get_weather", "input": {}}),
        block_delta(0, {"type": "input_json_delta", "partial_json": '{"city":'}),
        block_delta(0, {"type": "input_json_delta", "partial_json": '"Paris"}'}),
        block_stop(0),
        block_start(1, {"type": "tool_use", "id": "toolu_dec1b", "name": "get_time", "input": {}}),
        block_delta(1, {"type": "input_json_delta", "partial_json": '{"city":'}),
        block_delta(1, {"type": "input_json_delta", "partial_json": '"Rome"}'}),
        block_stop(1),
        message_delta("tool_use"),
        MESSAGE_STOP,
    )
    tools: Final[list[JsonValue]] = [
        tool("get_weather", city={"type": "string"}),
        tool("get_time", city={"type": "string"}),
    ]
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(gateway, _body(deployment(scenario, provider), tools=tools))
        events: Final = data_events(lines)
        fragments: Final = _tool_calls(events)
        by_index: Final[dict[int, list[dict[str, JsonValue]]]] = {}
        for fragment in fragments:
            by_index.setdefault(int(str(fragment["index"])), []).append(fragment)
        assert sorted(by_index) == [0, 1]
        assert [by_index[i][0].get("id") for i in (0, 1)] == ["toolu_dec1a", "toolu_dec1b"]
        assert [object_value(by_index[i][0]["function"]).get("name") for i in (0, 1)] == ["get_weather", "get_time"]
        arguments: Final = [
            "".join(str(object_value(f["function"]).get("arguments") or "") for f in by_index[i]) for i in (0, 1)
        ]
        assert [json.loads(a) for a in arguments] == [{"city": "Paris"}, {"city": "Rome"}]
        assert finish_reasons(events) == ("tool_calls",)


def test_a_text_block_before_a_streamed_tool_call_keeps_its_text(gateway: Gateway) -> None:
    """The provider streams a text block `checking` then a `tool_use` block. The caller must get `checking` as
    content and one tool call `toolu_dec1a`, so assistants that narrate before calling a tool keep their words.
    """
    chunks: Final = (
        message_start(),
        block_start(0, {"type": "text", "text": ""}),
        block_delta(0, {"type": "text_delta", "text": "checking"}),
        block_stop(0),
        block_start(1, {"type": "tool_use", "id": "toolu_dec1a", "name": "get_weather", "input": {}}),
        block_delta(1, {"type": "input_json_delta", "partial_json": "{}"}),
        block_stop(1),
        message_delta("tool_use"),
        MESSAGE_STOP,
    )
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(
            gateway, _body(deployment(scenario, provider), tools=[tool("get_weather", city={"type": "string"})])
        )
        events: Final = data_events(lines)
        assert streamed_text(events) == "checking"
        assert {c.get("id") for c in _tool_calls(events) if c.get("id")} == {"toolu_dec1a"}


def test_streamed_thinking_arrives_as_reasoning_content_with_the_signature_on_a_thinking_block(
    gateway: Gateway,
) -> None:
    """The provider streams a thinking block (`thinking_delta` `plan `, `thinking_delta` `ahead`, `signature_delta`)
    then a text block `done`.

    The caller's `reasoning_content` pieces must join to `plan ahead`, the text to `done`, and some delta must carry
    `thinking_blocks` with a `thinking` block whose `signature` is the provider's, so the caller can replay it.
    """
    chunks: Final = (
        message_start(),
        block_start(0, {"type": "thinking", "thinking": "", "signature": ""}),
        block_delta(0, {"type": "thinking_delta", "thinking": "plan "}),
        block_delta(0, {"type": "thinking_delta", "thinking": "ahead"}),
        block_delta(0, {"type": "signature_delta", "signature": SIGNATURE}),
        block_stop(0),
        block_start(1, {"type": "text", "text": ""}),
        block_delta(1, {"type": "text_delta", "text": "done"}),
        block_stop(1),
        message_delta(),
        MESSAGE_STOP,
    )
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(gateway, _body(deployment(scenario, provider), reasoning_effort="low"))
        events: Final = data_events(lines)
        all_deltas: Final = deltas(events)
        assert "".join(str(d.get("reasoning_content") or "") for d in all_deltas) == "plan ahead"
        assert streamed_text(events) == "done"
        signatures: Final = [
            object_value(block).get("signature")
            for d in all_deltas
            for block in list_value(d.get("thinking_blocks") or [])
        ]
        assert SIGNATURE in signatures


def test_a_streamed_server_web_search_is_not_relayed_as_a_client_tool_call(gateway: Gateway) -> None:
    """The provider streams a `server_tool_use` web_search block with its input, a `web_search_tool_result`, and a
    text answer.

    The search already ran on Anthropic's side, so no caller delta may carry `tool_calls`, and the answer text must
    arrive as content (`chunk_parser` builds a client tool chunk only for `tool_use`).
    """
    chunks: Final = (
        message_start(),
        block_start(0, {"type": "server_tool_use", "id": "srvtoolu_dec1", "name": "web_search", "input": {}}),
        block_delta(0, {"type": "input_json_delta", "partial_json": '{"query":"dec1"}'}),
        block_stop(0),
        block_start(
            1,
            {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_dec1",
                "content": [
                    {"type": "web_search_result", "url": "https://dec1.invalid", "title": "t", "encrypted_content": "x"}
                ],
            },
        ),
        block_stop(1),
        block_start(2, {"type": "text", "text": ""}),
        block_delta(2, {"type": "text_delta", "text": REPLY_TEXT}),
        block_stop(2),
        message_delta(usage={"output_tokens": 7, "server_tool_use": {"web_search_requests": 1}}),
        MESSAGE_STOP,
    )
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(
            gateway, _body(deployment(scenario, provider), web_search_options={"search_context_size": "low"})
        )
        events: Final = data_events(lines)
        assert _tool_calls(events) == []
        assert streamed_text(events) == REPLY_TEXT


def test_a_streamed_json_tool_call_is_relayed_as_content(gateway: Gateway) -> None:
    """The caller streams with an open-schema `response_format`, which takes the `json_tool_call` path; the provider
    streams a `json_tool_call` tool_use whose input arrives in two `input_json_delta` pieces.

    The caller must get content that decodes to `{"city": "Paris"}` and no tool call fragments
    (`_handle_json_mode_chunk`).
    """
    chunks: Final = (
        message_start(),
        block_start(0, {"type": "tool_use", "id": "toolu_dec1json", "name": "json_tool_call", "input": {}}),
        block_delta(0, {"type": "input_json_delta", "partial_json": '{"city":'}),
        block_delta(0, {"type": "input_json_delta", "partial_json": '"Paris"}'}),
        block_stop(0),
        message_delta("tool_use"),
        MESSAGE_STOP,
    )
    schema: Final[dict[str, JsonValue]] = {"type": "object", "properties": {"city": {"type": "string"}}}
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(
            gateway,
            _body(
                deployment(scenario, provider),
                response_format={"type": "json_schema", "json_schema": {"name": "place", "schema": schema}},
            ),
        )
        events: Final = data_events(lines)
        assert json.loads(streamed_text(events)) == {"city": "Paris"}
        assert _tool_calls(events) == []


def test_a_mid_stream_overloaded_error_event_reaches_the_caller_after_the_partial_text(gateway: Gateway) -> None:
    """The provider streams `message_start`, a text delta `partial`, then an `error` event
    `{"type": "overloaded_error", "message": "Overloaded"}` and closes.

    The caller already has a 200 stream, so it must first get `partial` as content and then a `data:` frame whose
    JSON has an `error` object mentioning `Overloaded` (`chunk_parser` raises AnthropicError 500 on an `error`
    event). UNVERIFIED: the exact error frame keys; the test reads only `error` and its text.
    """
    chunks: Final = (
        message_start(),
        block_start(0, {"type": "text", "text": ""}),
        block_delta(0, {"type": "text_delta", "text": "partial"}),
        sse("error", {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}),
    )
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(gateway, _body(deployment(scenario, provider)))
        events: Final = data_events(lines)
        assert streamed_text(events) == "partial"
        errors: Final = [object_value(e["error"]) for e in events if "error" in e]
        assert errors
        assert "Overloaded" in string_value(errors[-1].get("message") or "")


def test_a_streamed_call_writes_a_spend_row_priced_from_the_stream_usage(gateway: Gateway) -> None:
    """The caller streams a call the provider prices at 11 input and 7 output tokens.

    One SpendLogs row for the model group must land with `prompt_tokens: 11`, `completion_tokens: 7` and spend
    0.025, the same as the non-streaming call, even though the caller did not ask for usage in the stream.
    """
    with peer() as provider, gateway.scenario() as scenario:
        model: Final = deployment(scenario, provider)
        chat_stream_lines(gateway, _body(model))
        row: Final = spend_row_by_group(gateway, model)
        assert (row["prompt_tokens"], row["completion_tokens"]) == (11, 7)
        assert float(str(row["spend"])) == pytest.approx(CALL_COST)


def test_streamed_cache_creation_tokens_are_added_to_prompt_tokens(gateway: Gateway) -> None:
    """`message_start` reports `input_tokens: 4`, `cache_creation_input_tokens: 6`, `cache_read_input_tokens: 0`;
    `message_delta` reports `output_tokens: 7`.

    With `include_usage` the final usage must say `prompt_tokens: 10` and `cache_creation_input_tokens: 6`, the
    streaming counterpart of the cache write accounting (the cache read case is pinned in the streaming suite).
    """
    chunks: Final = (
        message_start(
            usage={
                "input_tokens": 4,
                "output_tokens": 1,
                "cache_creation_input_tokens": 6,
                "cache_read_input_tokens": 0,
            }
        ),
        block_start(0, {"type": "text", "text": ""}),
        block_delta(0, {"type": "text_delta", "text": marker()}),
        block_stop(0),
        message_delta(),
        MESSAGE_STOP,
    )
    with peer(streaming(*chunks)) as provider, gateway.scenario() as scenario:
        _, lines = chat_stream_lines(
            gateway, _body(deployment(scenario, provider), stream_options={"include_usage": True})
        )
        usages: Final = [object_value(e["usage"]) for e in data_events(lines) if e.get("usage")]
        assert usages
        assert usages[-1]["prompt_tokens"] == 10
        assert usages[-1].get("cache_creation_input_tokens") == 6
