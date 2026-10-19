import json
from collections.abc import Iterable, Mapping
from typing import Final

from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
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
