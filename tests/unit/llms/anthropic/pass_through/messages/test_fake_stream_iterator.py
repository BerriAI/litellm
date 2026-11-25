import json
from collections.abc import Iterable, Mapping
from typing import Final, cast

import pytest

from litellm.llms.anthropic.pass_through.messages.fake_stream_iterator import (
    FakeAnthropicMessagesStreamIterator,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)


def _decoded_event(chunk: object) -> tuple[str, dict[str, object]]:
    assert isinstance(chunk, bytes)
    event_line, data_line = chunk.decode().split("\n")[:2]
    assert event_line.startswith("event: ")
    return event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))


def _decoded_events(chunks: Iterable[object]) -> tuple[tuple[str, dict[str, object]], ...]:
    return tuple(_decoded_event(chunk) for chunk in chunks)


@pytest.mark.parametrize(
    "unsigned_block",
    [
        {"type": "thinking", "thinking": "Let me think.", "signature": ""},
        {"type": "thinking", "thinking": "Let me think.", "signature": None},
        {"type": "thinking", "thinking": "Let me think."},
    ],
    ids=["empty", "null", "missing"],
)
def test_unsigned_thinking_block_closes_with_an_empty_signature_delta(unsigned_block: Mapping[str, object]):
    response: Final = {
        "id": "msg_unsigned",
        "content": [unsigned_block, {"type": "text", "text": "Hi"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    thinking_events: Final = tuple(
        (event_type, payload.get("delta", payload.get("content_block")))
        for event_type, payload in _decoded_events(
            FakeAnthropicMessagesStreamIterator(response=cast(AnthropicMessagesResponse, response))
        )
        if payload.get("index") == 0
    )
    assert thinking_events == (
        ("content_block_start", {"type": "thinking", "thinking": "", "signature": ""}),
        ("content_block_delta", {"type": "thinking_delta", "thinking": "Let me think."}),
        ("content_block_delta", {"type": "signature_delta", "signature": ""}),
        ("content_block_stop", None),
    )
