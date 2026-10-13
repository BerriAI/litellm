import json
from typing import Final

import pytest

from litellm.proxy.guardrails.anthropic_sse import (
    anthropic_sse_chunks_from_response,
    assemble_anthropic_sse_stream,
)
from litellm.types.utils import Choices


def _frame(event: str, data: dict[str, object]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def _decode(chunks: tuple[bytes, ...]) -> list[tuple[str, dict[str, object]]]:
    decoded = []
    for chunk in chunks:
        lines = chunk.decode().split("\n")
        assert lines[0].startswith("event: ")
        decoded.append((lines[0].removeprefix("event: "), json.loads(lines[1].removeprefix("data: "))))
    return decoded


def _upstream_stream(delta: dict[str, object]) -> tuple[bytes, ...]:
    return (
        _frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_upstream_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-upstream",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 3, "output_tokens": 1},
                },
            },
        ),
        _frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
        _frame(
            "content_block_delta",
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "call 555-0100"}},
        ),
        _frame("content_block_stop", {"type": "content_block_stop", "index": 0}),
        _frame("message_delta", {"type": "message_delta", "delta": delta, "usage": {"output_tokens": 2}}),
        _frame("message_stop", {"type": "message_stop"}),
    )


SAFEGUARD_RESULTS: Final = [
    {"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": {"toolu_01": {"verdict": "dangerous"}}}}
]
REFUSAL_STOP_DETAILS: Final = {"type": "refusal", "category": "cyber", "explanation": "declined"}


@pytest.mark.parametrize(
    ("stop_reason", "delta_fields"),
    [
        ("end_turn", {"safeguard_results": SAFEGUARD_RESULTS}),
        ("refusal", {"stop_details": REFUSAL_STOP_DETAILS}),
    ],
    ids=["safeguard_results", "refusal_stop_details"],
)
def test_rewritten_stream_keeps_the_upstream_message_delta_fields(stop_reason: str, delta_fields: dict[str, object]):
    """A guardrail that rewrites a streamed response assembles it into the OpenAI shape, which has no
    slot for ``safeguard_results`` or ``stop_details``, so the re-emitted ``message_delta`` used to
    drop the classifier verdict a Claude Code auto-mode session reads from it."""
    upstream_delta: dict[str, object] = {"stop_reason": stop_reason, "stop_sequence": None, **delta_fields}
    upstream = _upstream_stream(upstream_delta)
    assembled = assemble_anthropic_sse_stream(upstream, restore_identity=True)
    assert assembled is not None
    choice = assembled.choices[0]
    assert isinstance(choice, Choices)
    choice.message.content = "call <PHONE>"

    decoded = _decode(anthropic_sse_chunks_from_response(assembled, upstream))
    payloads = {event: payload for event, payload in decoded}

    assert payloads["content_block_delta"]["delta"] == {"type": "text_delta", "text": "call <PHONE>"}
    assert payloads["message_delta"]["delta"] == upstream_delta
    assert not set(delta_fields) & set(payloads["message_start"]["message"])
