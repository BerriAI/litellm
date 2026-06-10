"""
Tests for FakeAnthropicMessagesStreamIterator.

Focus: the advisor blocks (server_tool_use + advisor_tool_result) must stream as
content_block_start events so clients render the advisor activity. Before the fix
these block types were silently dropped by the iterator.
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)


def _parse_events(chunks):
    events = []
    for raw in chunks:
        text = raw.decode()
        for block in text.strip().split("\n\n"):
            if not block.strip():
                continue
            data_line = next(
                line for line in block.splitlines() if line.startswith("data: ")
            )
            events.append(json.loads(data_line[len("data: ") :]))
    return events


def _advisor_response():
    return {
        "id": "msg_x",
        "type": "message",
        "role": "assistant",
        "model": "gpt-4o-mini",
        "content": [
            {"type": "server_tool_use", "id": "tid_01", "name": "advisor", "input": {}},
            {
                "type": "advisor_tool_result",
                "tool_use_id": "tid_01",
                "content": {"type": "advisor_result", "text": "Use a sieve."},
            },
            {"type": "text", "text": "Final answer."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


def test_advisor_blocks_emitted_as_content_block_start():
    iterator = FakeAnthropicMessagesStreamIterator(_advisor_response())
    events = _parse_events(list(iterator))

    starts = [e for e in events if e["type"] == "content_block_start"]
    by_type = {e["content_block"]["type"]: e["content_block"] for e in starts}

    assert "server_tool_use" in by_type, "advisor call must stream"
    assert by_type["server_tool_use"]["name"] == "advisor"

    assert "advisor_tool_result" in by_type, "advisor reply must stream"
    assert by_type["advisor_tool_result"]["tool_use_id"] == "tid_01"
    assert by_type["advisor_tool_result"]["content"] == {
        "type": "advisor_result",
        "text": "Use a sieve.",
    }


def test_server_tool_use_emits_input_json_delta():
    iterator = FakeAnthropicMessagesStreamIterator(_advisor_response())
    events = _parse_events(list(iterator))

    deltas = [
        e
        for e in events
        if e["type"] == "content_block_delta"
        and e["delta"].get("type") == "input_json_delta"
    ]
    assert any(d["delta"]["partial_json"] == "{}" for d in deltas)
