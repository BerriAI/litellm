import json

from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)

_CLAUDE_CODE_SAFEGUARD_RESULTS = [
    {
        "type": "dangerous_tool_use",
        "status": {"type": "available", "tool_uses": {"toolu_01": {"verdict": "dangerous"}}},
    }
]


def _decode_sse_chunks(chunks: list[bytes]) -> list[tuple[str, dict]]:
    decoded = []
    for chunk in chunks:
        assert isinstance(chunk, bytes)
        lines = chunk.decode().split("\n")
        assert lines[0].startswith("event: ")
        decoded.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return decoded


def _message_delta(response: dict) -> dict:
    decoded = _decode_sse_chunks(list(FakeAnthropicMessagesStreamIterator(response=response)))
    assert [event_type for event_type, _ in decoded][-2:] == ["message_delta", "message_stop"]
    deltas = [payload["delta"] for event_type, payload in decoded if event_type == "message_delta"]
    assert len(deltas) == 1
    return deltas[0]


def test_message_delta_carries_safeguard_results():
    """Websearch interception, the MCP loop and the guardrail path render a
    complete response as this stream for a client that asked to stream; a
    Claude Code auto-mode client reads the classifier verdict from
    message_delta.delta, where a real Anthropic stream puts it."""
    response = {
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
    stop_details = {"type": "refusal", "category": "cyber", "explanation": "declined"}
    response = {
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
    response = {
        "id": "msg_3",
        "content": [{"type": "text", "text": "hello"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "stop_details": None,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    assert _message_delta(response) == {"stop_reason": "end_turn", "stop_sequence": None}
