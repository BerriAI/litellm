import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Final, cast

import anthropic
import httpx
import pytest

from litellm.llms.anthropic.messages.native.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)

_CLAUDE_CODE_SAFEGUARD_RESULTS: Final = [
    {
        "type": "dangerous_tool_use",
        "status": {
            "type": "available",
            "tool_uses": {
                "toolu_01": {"type": "evaluated", "outcome": "flagged", "explanation": "deletes the filesystem root"}
            },
        },
    }
]


def _decoded_event(chunk: object) -> tuple[str, dict[str, object]]:
    assert isinstance(chunk, bytes)
    event_line, data_line = chunk.decode().split("\n")[:2]
    assert event_line.startswith("event: ")
    return event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))


def _decoded_events(chunks: Iterable[object]) -> tuple[tuple[str, dict[str, object]], ...]:
    return tuple(_decoded_event(chunk) for chunk in chunks)


def _message_delta(response: Mapping[str, object]) -> object:
    decoded: Final = _decoded_events(FakeAnthropicMessagesStreamIterator(response=response))
    assert tuple(event_type for event_type, _ in decoded)[-2:] == ("message_delta", "message_stop")
    deltas: Final = tuple(payload["delta"] for event_type, payload in decoded if event_type == "message_delta")
    assert len(deltas) == 1
    return deltas[0]


def test_message_delta_carries_safeguard_results():
    """Websearch interception and the MCP loop render a complete response as
    this stream for a client that asked to stream (the guardrail path does too,
    after merging the upstream fields back in, which
    tests/unit/proxy/guardrails/test_anthropic_sse.py covers); a Claude Code
    auto-mode client reads the classifier verdict from message_delta.delta,
    where a real Anthropic stream puts it."""
    response: Final = {
        "id": "msg_1",
        "model": "claude-sonnet-4-6",
        "role": "assistant",
        "type": "message",
        "content": [{"type": "text", "text": "I won't run that."}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "safeguard_results": _CLAUDE_CODE_SAFEGUARD_RESULTS,
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    assert _message_delta(response) == {
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "safeguard_results": _CLAUDE_CODE_SAFEGUARD_RESULTS,
    }


def test_message_delta_carries_refusal_stop_details():
    stop_details: Final = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    response: Final = {
        "id": "msg_2",
        "content": [],
        "stop_reason": "refusal",
        "stop_sequence": None,
        "stop_details": stop_details,
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }
    assert _message_delta(response) == {
        "stop_reason": "refusal",
        "stop_sequence": None,
        "stop_details": stop_details,
    }


def test_message_delta_omits_absent_optional_fields():
    response: Final = {
        "id": "msg_3",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    assert _message_delta(response) == {"stop_reason": "end_turn", "stop_sequence": None}


_WEB_SEARCH_RESULT_BLOCK: Final = {
    "type": "web_search_tool_result",
    "tool_use_id": "srvtoolu_01",
    "content": [
        {
            "type": "web_search_result",
            "url": "https://example.com/a",
            "title": "Example A",
            "encrypted_content": "abc123",
        }
    ],
}


def _content_events(content: Sequence[Mapping[str, object]]) -> tuple[tuple[str, dict[str, object]], ...]:
    response: Final = {
        "id": "msg_ws",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-6",
        "content": list(content),
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    return tuple(
        event for event in _decoded_events(FakeAnthropicMessagesStreamIterator(response=response))
        if event[0].startswith("content_block_")
    )


def test_server_tool_use_streams_empty_start_then_input_json_delta():
    """A real Anthropic stream opens server_tool_use with an empty input and
    sends the query as an input_json_delta, which SDK accumulators read."""
    caller: Final = {"type": "code_execution_20260120", "tool_id": "srvtoolu_00"}
    events: Final = _content_events(
        [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {"query": "example search"},
                "caller": caller,
            }
        ]
    )
    assert [event_type for event_type, _ in events] == [
        "content_block_start",
        "content_block_delta",
        "content_block_stop",
    ]
    assert events[0][1]["content_block"] == {
        "type": "server_tool_use",
        "id": "srvtoolu_01",
        "name": "web_search",
        "input": {},
        "caller": caller,
    }
    delta: Final = events[1][1]["delta"]
    assert isinstance(delta, dict)
    assert delta["type"] == "input_json_delta"
    assert json.loads(delta["partial_json"]) == {"query": "example search"}


def test_web_search_tool_result_rides_whole_in_content_block_start():
    events: Final = _content_events([_WEB_SEARCH_RESULT_BLOCK])
    assert [event_type for event_type, _ in events] == ["content_block_start", "content_block_stop"]
    assert events[0][1]["content_block"] == _WEB_SEARCH_RESULT_BLOCK


def test_native_search_blocks_keep_indices_aligned_with_text():
    events: Final = _content_events(
        [
            {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {"query": "q"}},
            _WEB_SEARCH_RESULT_BLOCK,
            {"type": "text", "text": "Here is what I found."},
        ]
    )
    starts: Final = tuple(payload for event_type, payload in events if event_type == "content_block_start")
    stops: Final = tuple(payload["index"] for event_type, payload in events if event_type == "content_block_stop")
    assert [(start["index"], cast(dict[str, object], start["content_block"])["type"]) for start in starts] == [
        (0, "server_tool_use"),
        (1, "web_search_tool_result"),
        (2, "text"),
    ]
    assert stops == (0, 1, 2)


_CALLER: Final = {"type": "code_execution_20260120", "tool_id": "srvtoolu_00"}

_REPLAYED_BLOCKS: Final = (
    {"type": "thinking", "thinking": "Look up the release page.", "signature": "sig_01"},
    {"type": "redacted_thinking", "data": "EncryptedThinking=="},
    {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}, "caller": _CALLER},
    {"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_fetch", "input": {"url": "https://example.com/a"}, "caller": _CALLER},
    _WEB_SEARCH_RESULT_BLOCK,
    {
        "type": "text",
        "text": "Example A is the newest release.",
        "citations": [
            {
                "type": "web_search_result_location",
                "cited_text": "Example A was released today.",
                "url": "https://example.com/a",
                "title": "Example A",
                "encrypted_index": "idx_01",
            }
        ],
    },
)


def _sdk_rebuilt_content(block: Mapping[str, object]) -> list[dict[str, object]]:
    response: Final = {
        "id": "msg_replay",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [block],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    body: Final = b"".join(FakeAnthropicMessagesStreamIterator(response=response))
    transport: Final = httpx.MockTransport(
        lambda _request: httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})
    )
    client: Final = anthropic.Anthropic(api_key="sk-ant-test", http_client=httpx.Client(transport=transport))
    with client.messages.stream(
        model="claude-sonnet-5-5", max_tokens=64, messages=[{"role": "user", "content": "hi"}]
    ) as stream:
        return [content_block.model_dump(exclude_none=True) for content_block in stream.get_final_message().content]


@pytest.mark.parametrize("block", _REPLAYED_BLOCKS, ids=lambda block: str(block["type"]))
def test_anthropic_sdk_rebuilds_every_block_field_from_the_stream(block: Mapping[str, object]):
    assert _sdk_rebuilt_content(block) == [block]
