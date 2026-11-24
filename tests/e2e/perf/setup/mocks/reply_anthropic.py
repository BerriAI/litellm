"""Anthropic wire format for a generated `Reply`: `/v1/messages` JSON and the SSE event sequence
(`message_start`, `ping`, content blocks of `thinking`, `text` or `tool_use` with their deltas,
`message_delta`, `message_stop`) with the usage fields, delta and chunk sizes api.anthropic.com uses."""

from __future__ import annotations

import json
from typing import Final

from reply_common import Reply, grouped, ident, pieces, sse

DELTA_CHARS: Final = 40
CHUNK_BYTES: Final = 1000


def _usage(reply: Reply) -> dict[str, object]:
    return {
        "input_tokens": reply.uncached_tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": reply.cached_tokens,
        "cache_creation": {"ephemeral_5m_input_tokens": 0, "ephemeral_1h_input_tokens": 0},
        "output_tokens": reply.output_tokens,
        "service_tier": "standard",
    }


def _signature(reply: Reply) -> str:
    return ident(reply.seed, "", 64, hex_digits=True)


def _blocks(reply: Reply) -> tuple[dict[str, object], ...]:
    thinking: Final[tuple[dict[str, object], ...]] = (
        ({"type": "thinking", "thinking": reply.reasoning, "signature": _signature(reply)},)
        if reply.reasoning_tokens
        else ()
    )
    if reply.tool_call is None:
        return (*thinking, _block(type="text", text=reply.text))
    tool: Final = _block(
        type="tool_use",
        id=ident(reply.seed, "toolu_", 24),
        name=reply.tool_call.name,
        input=reply.tool_call.parsed_arguments,
    )
    return (*thinking, tool)


def _block(**fields: object) -> dict[str, object]:
    return dict(fields)


def _stop_reason(reply: Reply) -> str:
    return "end_turn" if reply.tool_call is None else "tool_use"


def messages_json(reply: Reply) -> bytes:
    body: Final = {
        "model": reply.model,
        "id": ident(reply.seed, "msg_", 24),
        "type": "message",
        "role": "assistant",
        "content": list(_blocks(reply)),
        "stop_reason": _stop_reason(reply),
        "stop_sequence": None,
        "usage": _usage(reply),
    }
    return json.dumps(body, separators=(",", ":")).encode()


def _block_events(
    index: int, start: dict[str, object], deltas: tuple[dict[str, object], ...]
) -> tuple[dict[str, object], ...]:
    return (
        {"type": "content_block_start", "index": index, "content_block": start},
        *({"type": "content_block_delta", "index": index, "delta": delta} for delta in deltas),
        {"type": "content_block_stop", "index": index},
    )


def messages_events(reply: Reply) -> tuple[dict[str, object], ...]:
    """The stream as event payloads, `message_start` to `message_stop`, each one's `type` its SSE event name."""
    opening: Final[dict[str, object]] = {
        "model": reply.model,
        "id": ident(reply.seed, "msg_", 24),
        "type": "message",
        "role": "assistant",
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": {**_usage(reply), "output_tokens": 1},
    }
    thinking: Final = (
        _block_events(
            0,
            {"type": "thinking", "thinking": "", "signature": ""},
            (
                *({"type": "thinking_delta", "thinking": piece} for piece in pieces(reply.reasoning, DELTA_CHARS)),
                {"type": "signature_delta", "signature": _signature(reply)},
            ),
        )
        if reply.reasoning_tokens
        else ()
    )
    index: Final = 1 if reply.reasoning_tokens else 0
    content: Final = (
        _block_events(
            index,
            {"type": "text", "text": ""},
            tuple({"type": "text_delta", "text": p} for p in pieces(reply.text, DELTA_CHARS)),
        )
        if reply.tool_call is None
        else _block_events(
            index,
            {"type": "tool_use", "id": ident(reply.seed, "toolu_", 24), "name": reply.tool_call.name, "input": {}},
            tuple(
                {"type": "input_json_delta", "partial_json": p} for p in pieces(reply.tool_call.arguments, DELTA_CHARS)
            ),
        )
    )
    closing: Final[tuple[dict[str, object], ...]] = (
        {
            "type": "message_delta",
            "delta": {"stop_reason": _stop_reason(reply), "stop_sequence": None},
            "usage": {**_usage(reply), "output_tokens": reply.output_tokens},
        },
        {"type": "message_stop"},
    )
    return ({"type": "message_start", "message": opening}, {"type": "ping"}, *thinking, *content, *closing)


def messages_chunks(reply: Reply) -> tuple[bytes, ...]:
    return grouped(tuple(sse(event, str(event["type"])) for event in messages_events(reply)), CHUNK_BYTES)
