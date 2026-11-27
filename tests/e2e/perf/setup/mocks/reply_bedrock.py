"""Bedrock wire formats for a generated `Reply`: the `converse` JSON body and `converse-stream` frames, and
for Anthropic models the `invoke` body (Anthropic's own messages JSON) and `invoke-with-response-stream` frames
(each a `chunk` event carrying one base64 Anthropic stream event). Frames are `application/vnd.amazon.eventstream`
(prelude, string headers, JSON payload, CRC32s), what the gateway's botocore-based decoder reads, one per chunk."""

from __future__ import annotations

import base64
import json
import struct
import zlib
from collections.abc import Callable
from typing import Final

from reply_anthropic import messages_events
from reply_common import Reply, ident, pick, pieces

DELTA_CHARS: Final = 15
_PADDING: Final = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123"


def _usage(reply: Reply) -> dict[str, object]:
    return {
        "cacheReadInputTokenCount": reply.cached_tokens,
        "cacheReadInputTokens": reply.cached_tokens,
        "cacheWriteInputTokenCount": 0,
        "cacheWriteInputTokens": 0,
        "inputTokens": reply.uncached_tokens,
        "outputTokens": reply.output_tokens,
        "serverToolUsage": {},
        "totalTokens": reply.total_tokens,
    }


def _tool_use_id(reply: Reply) -> str:
    return ident(reply.seed, "tooluse_", 22)


def _stop_reason(reply: Reply) -> str:
    return "end_turn" if reply.tool_call is None else "tool_use"


def _text_block(text: str) -> dict[str, object]:
    return {"text": text}


def _content(reply: Reply) -> tuple[dict[str, object], ...]:
    reasoning: Final[tuple[dict[str, object], ...]] = (
        (
            {
                "reasoningContent": {
                    "reasoningText": {"text": reply.reasoning, "signature": ident(reply.seed, "", 64, hex_digits=True)}
                }
            },
        )
        if reply.reasoning_tokens
        else ()
    )
    if reply.tool_call is None:
        return (*reasoning, _text_block(reply.text))
    tool: Final[dict[str, object]] = {
        "toolUse": {
            "toolUseId": _tool_use_id(reply),
            "name": reply.tool_call.name,
            "input": reply.tool_call.parsed_arguments,
        }
    }
    return (*reasoning, tool)


def converse_json(reply: Reply) -> bytes:
    body: Final = {
        "metrics": {"latencyMs": 40 * reply.output_tokens},
        "output": {"message": {"role": "assistant", "content": list(_content(reply))}},
        "stopReason": _stop_reason(reply),
        "usage": _usage(reply),
    }
    return json.dumps(body, separators=(",", ":")).encode()


def _header(name: str, value: str) -> bytes:
    encoded: Final = value.encode()
    return struct.pack("!B", len(name)) + name.encode() + struct.pack("!BH", 7, len(encoded)) + encoded


def frame(event_type: str, payload: dict[str, object]) -> bytes:
    """One eventstream message: prelude (total length, headers length, CRC), the three string headers
    bedrock-runtime sets, the JSON payload, and the CRC of everything before it."""
    headers: Final = (
        _header(":event-type", event_type)
        + _header(":content-type", "application/json")
        + _header(":message-type", "event")
    )
    body: Final = json.dumps(payload, separators=(",", ":")).encode()
    total: Final = 4 + 4 + 4 + len(headers) + len(body) + 4
    prelude: Final = struct.pack("!II", total, len(headers))
    head: Final = prelude + struct.pack("!I", zlib.crc32(prelude))
    message: Final = head + headers + body
    return message + struct.pack("!I", zlib.crc32(message))


def _pad(seed: str, index: int) -> str:
    return _PADDING[: 2 + pick(f"{seed}:pad", index, 50)]


def _text_events(reply: Reply, index: int, event: Callable[[int, str, dict[str, object]], bytes]) -> tuple[bytes, ...]:
    return (
        *(
            event(i, "contentBlockDelta", {"contentBlockIndex": index, "delta": {"text": piece}})
            for i, piece in enumerate(pieces(reply.text, DELTA_CHARS), start=1)
        ),
        event(0, "contentBlockStop", {"contentBlockIndex": index}),
    )


def _tool_events(reply: Reply, index: int, event: Callable[[int, str, dict[str, object]], bytes]) -> tuple[bytes, ...]:
    assert reply.tool_call is not None
    start: Final = {"toolUse": {"name": reply.tool_call.name, "toolUseId": _tool_use_id(reply)}}
    return (
        event(0, "contentBlockStart", {"contentBlockIndex": index, "start": start}),
        *(
            event(i, "contentBlockDelta", {"contentBlockIndex": index, "delta": {"toolUse": {"input": piece}}})
            for i, piece in enumerate(pieces(reply.tool_call.arguments, DELTA_CHARS), start=1)
        ),
        event(0, "contentBlockStop", {"contentBlockIndex": index}),
    )


def converse_chunks(reply: Reply) -> tuple[bytes, ...]:
    def event(index: int, event_type: str, payload: dict[str, object]) -> bytes:
        return frame(event_type, {**payload, "p": _pad(reply.seed, index)})

    reasoning: Final = (
        (
            *(
                event(i, "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": piece}}})
                for i, piece in enumerate(pieces(reply.reasoning, DELTA_CHARS), start=1)
            ),
            event(
                0,
                "contentBlockDelta",
                {
                    "contentBlockIndex": 0,
                    "delta": {"reasoningContent": {"signature": ident(reply.seed, "", 64, hex_digits=True)}},
                },
            ),
            event(0, "contentBlockStop", {"contentBlockIndex": 0}),
        )
        if reply.reasoning_tokens
        else ()
    )
    index: Final = 1 if reply.reasoning_tokens else 0
    content: Final = _text_events(reply, index, event) if reply.tool_call is None else _tool_events(reply, index, event)
    return (
        event(0, "messageStart", {"role": "assistant"}),
        *reasoning,
        *content,
        event(0, "messageStop", {"stopReason": _stop_reason(reply)}),
        event(0, "metadata", {"metrics": {"latencyMs": 40 * reply.output_tokens}, "usage": _usage(reply)}),
    )


def invoke_chunks(reply: Reply) -> tuple[bytes, ...]:
    """The Anthropic stream events, each base64 in the `bytes` of one `chunk` frame."""
    return tuple(
        frame(
            "chunk",
            {
                "bytes": base64.b64encode(json.dumps(event, separators=(",", ":")).encode()).decode(),
                "p": _pad(reply.seed, index),
            },
        )
        for index, event in enumerate(messages_events(reply))
    )
