"""
Tests for FakeAnthropicMessagesStreamIterator

Source: litellm/llms/anthropic/experimental_pass_through/messages/fake_stream_iterator.py
"""

import json

from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)

WEB_SEARCH_RESULT_BLOCK = {
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


def _response_with_content(content: list) -> dict:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5",
        "content": content,
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }


def _events_by_type(iterator: FakeAnthropicMessagesStreamIterator) -> list[dict]:
    events = []
    for chunk in iterator.chunks:
        for line in chunk.decode().splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[6:]))
    return events


def test_server_tool_use_block_streams_start_and_input_delta():
    server_tool_use = {
        "type": "server_tool_use",
        "id": "srvtoolu_01",
        "name": "web_search",
        "input": {"query": "example search"},
    }
    iterator = FakeAnthropicMessagesStreamIterator(
        response=_response_with_content([server_tool_use])
    )

    events = _events_by_type(iterator)

    starts = [e for e in events if e.get("type") == "content_block_start"]
    assert starts[0]["content_block"]["type"] == "server_tool_use"
    assert starts[0]["content_block"]["id"] == "srvtoolu_01"
    assert starts[0]["content_block"]["name"] == "web_search"

    deltas = [e for e in events if e.get("type") == "content_block_delta"]
    assert deltas[0]["delta"]["type"] == "input_json_delta"
    assert json.loads(deltas[0]["delta"]["partial_json"]) == {
        "query": "example search"
    }

    stops = [e for e in events if e.get("type") == "content_block_stop"]
    assert len(stops) == 1


def test_web_search_tool_result_block_rides_in_content_block_start():
    iterator = FakeAnthropicMessagesStreamIterator(
        response=_response_with_content([WEB_SEARCH_RESULT_BLOCK])
    )

    events = _events_by_type(iterator)

    starts = [e for e in events if e.get("type") == "content_block_start"]
    assert len(starts) == 1
    assert starts[0]["content_block"] == WEB_SEARCH_RESULT_BLOCK

    stops = [e for e in events if e.get("type") == "content_block_stop"]
    assert len(stops) == 1


def test_mixed_content_keeps_block_indices_aligned():
    content = [
        {"type": "text", "text": "Here is what I found."},
        {
            "type": "server_tool_use",
            "id": "srvtoolu_02",
            "name": "web_search",
            "input": {"query": "q"},
        },
        WEB_SEARCH_RESULT_BLOCK,
    ]
    iterator = FakeAnthropicMessagesStreamIterator(
        response=_response_with_content(content)
    )

    events = _events_by_type(iterator)

    starts = [e for e in events if e.get("type") == "content_block_start"]
    assert [s["index"] for s in starts] == [0, 1, 2]
    assert [s["content_block"]["type"] for s in starts] == [
        "text",
        "server_tool_use",
        "web_search_tool_result",
    ]

    stops = [e for e in events if e.get("type") == "content_block_stop"]
    assert [s["index"] for s in stops] == [0, 1, 2]


def test_text_only_block_behavior_unchanged():
    iterator = FakeAnthropicMessagesStreamIterator(
        response=_response_with_content([{"type": "text", "text": "hello"}])
    )

    events = _events_by_type(iterator)

    starts = [e for e in events if e.get("type") == "content_block_start"]
    assert starts[0]["content_block"] == {"type": "text", "text": ""}
    deltas = [e for e in events if e.get("type") == "content_block_delta"]
    assert deltas[0]["delta"] == {"type": "text_delta", "text": "hello"}
