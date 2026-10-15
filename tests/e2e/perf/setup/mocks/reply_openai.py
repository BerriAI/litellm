"""OpenAI wire formats for a generated `Reply`: `/v1/chat/completions` (JSON and SSE chunks) and
`/v1/responses` (JSON and typed SSE events), with the field set, ids, event order, delta sizes and chunk
sizes of what api.openai.com sends gpt-6-luna replies as."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Final

from reply_common import Reply, grouped, ident, pieces, sse

CHAT_DELTA_CHARS: Final = 4
CHAT_CHUNK_BYTES: Final = 560
RESPONSES_DELTA_CHARS: Final = 3
RESPONSES_CHUNK_BYTES: Final = 250


def _chat_usage(reply: Reply) -> dict[str, object]:
    return {
        "prompt_tokens": reply.prompt_tokens,
        "completion_tokens": reply.output_tokens,
        "total_tokens": reply.total_tokens,
        "prompt_tokens_details": {"cached_tokens": reply.cached_tokens, "audio_tokens": 0},
        "completion_tokens_details": {
            "reasoning_tokens": reply.reasoning_tokens,
            "audio_tokens": 0,
            "accepted_prediction_tokens": 0,
            "rejected_prediction_tokens": 0,
        },
    }


def _chat_message(reply: Reply) -> dict[str, object]:
    if reply.tool_call is None:
        return {"role": "assistant", "content": reply.text, "refusal": None, "annotations": []}
    call: Final = {
        "id": ident(reply.seed, "call_", 24),
        "type": "function",
        "function": {"name": reply.tool_call.name, "arguments": reply.tool_call.arguments},
    }
    return {"role": "assistant", "content": None, "refusal": None, "annotations": [], "tool_calls": [call]}


def chat_json(reply: Reply) -> bytes:
    finish: Final = "stop" if reply.tool_call is None else "tool_calls"
    body: Final = {
        "id": ident(reply.seed, "chatcmpl-", 29),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": reply.model,
        "choices": [{"index": 0, "message": _chat_message(reply), "logprobs": None, "finish_reason": finish}],
        "usage": _chat_usage(reply),
        "service_tier": "default",
        "system_fingerprint": None,
    }
    return json.dumps(body, separators=(",", ":")).encode()


def _chat_deltas(reply: Reply, chunk: Callable[[dict[str, object]], bytes]) -> tuple[bytes, ...]:
    if reply.tool_call is None:
        return tuple(chunk({"content": piece}) for piece in pieces(reply.text, CHAT_DELTA_CHARS))
    opening: Final = {
        "index": 0,
        "id": ident(reply.seed, "call_", 24),
        "type": "function",
        "function": {"name": reply.tool_call.name, "arguments": ""},
    }
    argument_pieces: Final = pieces(reply.tool_call.arguments, CHAT_DELTA_CHARS)
    return (
        chunk({"tool_calls": [opening]}),
        *(chunk({"tool_calls": [{"index": 0, "function": {"arguments": piece}}]}) for piece in argument_pieces),
    )


def chat_chunks(reply: Reply) -> tuple[bytes, ...]:
    head: Final = {
        "id": ident(reply.seed, "chatcmpl-", 29),
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": reply.model,
        "service_tier": "default",
        "system_fingerprint": None,
    }

    def chunk(delta: dict[str, object], finish: str | None = None) -> bytes:
        choice: Final = {"index": 0, "delta": delta, "logprobs": None, "finish_reason": finish}
        return sse({**head, "choices": [choice], "usage": None})

    deltas: Final = _chat_deltas(reply, chunk)
    finish: Final = "stop" if reply.tool_call is None else "tool_calls"
    events: Final = (
        chunk({"role": "assistant", "content": "", "refusal": None}),
        *deltas,
        chunk({}, finish),
        sse({**head, "choices": [], "usage": _chat_usage(reply)}),
        b"data: [DONE]\n\n",
    )
    return grouped(events, CHAT_CHUNK_BYTES)


def _responses_usage(reply: Reply) -> dict[str, object]:
    return {
        "input_tokens": reply.prompt_tokens,
        "input_tokens_details": {"cached_tokens": reply.cached_tokens},
        "output_tokens": reply.output_tokens,
        "output_tokens_details": {"reasoning_tokens": reply.reasoning_tokens},
        "total_tokens": reply.total_tokens,
    }


def _reasoning_item(reply: Reply) -> dict[str, object]:
    return {"id": ident(reply.seed, "rs_", 50, hex_digits=True), "type": "reasoning", "summary": [], "content": []}


def _message_item(reply: Reply, text: str) -> dict[str, object]:
    content: Final[dict[str, object]] = {"type": "output_text", "annotations": [], "logprobs": [], "text": text}
    return {
        "id": ident(reply.seed, "msg_", 50, hex_digits=True),
        "type": "message",
        "status": "completed",
        "content": [content],
        "role": "assistant",
    }


def _function_item(reply: Reply, arguments: str, status: str) -> dict[str, object]:
    assert reply.tool_call is not None
    return {
        "id": ident(reply.seed, "fc_", 50, hex_digits=True),
        "type": "function_call",
        "status": status,
        "arguments": arguments,
        "call_id": ident(reply.seed, "call_", 24),
        "name": reply.tool_call.name,
    }


def _output(reply: Reply) -> tuple[dict[str, object], ...]:
    final: Final = (
        _message_item(reply, reply.text)
        if reply.tool_call is None
        else _function_item(reply, reply.tool_call.arguments, "completed")
    )
    return (_reasoning_item(reply), final) if reply.reasoning_tokens else (final,)


def _response(reply: Reply, status: str, output: tuple[dict[str, object], ...]) -> dict[str, object]:
    created: Final = int(time.time())
    return {
        "id": ident(reply.seed, "resp_", 50, hex_digits=True),
        "object": "response",
        "created_at": created,
        "status": status,
        "background": False,
        "completed_at": created if status == "completed" else None,
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "model": reply.model,
        "output": list(output),
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "reasoning": {"effort": "medium", "summary": None},
        "service_tier": "default",
        "store": True,
        "temperature": 1.0,
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "truncation": "disabled",
        "usage": _responses_usage(reply) if status == "completed" else None,
        "user": None,
        "metadata": {},
    }


def responses_json(reply: Reply) -> bytes:
    return json.dumps(_response(reply, "completed", _output(reply)), separators=(",", ":")).encode()


def _message_events(reply: Reply, index: int, start: int) -> tuple[bytes, ...]:
    item_id: Final = ident(reply.seed, "msg_", 50, hex_digits=True)
    at: Final = {"item_id": item_id, "output_index": index, "content_index": 0}
    deltas: Final = pieces(reply.text, RESPONSES_DELTA_CHARS)
    part: Final[dict[str, object]] = {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}
    added: Final[dict[str, object]] = {
        "id": item_id,
        "type": "message",
        "status": "in_progress",
        "content": [],
        "role": "assistant",
    }
    middle: Final = tuple(
        sse(
            {"type": "response.output_text.delta", **at, "delta": delta, "sequence_number": start + 2 + i},
            "response.output_text.delta",
        )
        for i, delta in enumerate(deltas)
    )
    end: Final = start + 2 + len(deltas)
    return (
        sse(
            {"type": "response.output_item.added", "output_index": index, "item": added, "sequence_number": start},
            "response.output_item.added",
        ),
        sse(
            {"type": "response.content_part.added", **at, "part": part, "sequence_number": start + 1},
            "response.content_part.added",
        ),
        *middle,
        sse(
            {"type": "response.output_text.done", **at, "text": reply.text, "sequence_number": end},
            "response.output_text.done",
        ),
        sse(
            {
                "type": "response.content_part.done",
                **at,
                "part": {**part, "text": reply.text},
                "sequence_number": end + 1,
            },
            "response.content_part.done",
        ),
        sse(
            {
                "type": "response.output_item.done",
                "output_index": index,
                "item": _message_item(reply, reply.text),
                "sequence_number": end + 2,
            },
            "response.output_item.done",
        ),
    )


def _function_events(reply: Reply, index: int, start: int) -> tuple[bytes, ...]:
    assert reply.tool_call is not None
    item_id: Final = ident(reply.seed, "fc_", 50, hex_digits=True)
    deltas: Final = pieces(reply.tool_call.arguments, RESPONSES_DELTA_CHARS)
    middle: Final = tuple(
        sse(
            {
                "type": "response.function_call_arguments.delta",
                "item_id": item_id,
                "output_index": index,
                "delta": delta,
                "sequence_number": start + 1 + i,
            },
            "response.function_call_arguments.delta",
        )
        for i, delta in enumerate(deltas)
    )
    end: Final = start + 1 + len(deltas)
    return (
        sse(
            {
                "type": "response.output_item.added",
                "output_index": index,
                "item": _function_item(reply, "", "in_progress"),
                "sequence_number": start,
            },
            "response.output_item.added",
        ),
        *middle,
        sse(
            {
                "type": "response.function_call_arguments.done",
                "item_id": item_id,
                "output_index": index,
                "arguments": reply.tool_call.arguments,
                "sequence_number": end,
            },
            "response.function_call_arguments.done",
        ),
        sse(
            {
                "type": "response.output_item.done",
                "output_index": index,
                "item": _function_item(reply, reply.tool_call.arguments, "completed"),
                "sequence_number": end + 1,
            },
            "response.output_item.done",
        ),
    )


def responses_chunks(reply: Reply) -> tuple[bytes, ...]:
    opening: Final = (
        sse(
            {"type": "response.created", "response": _response(reply, "in_progress", ()), "sequence_number": 0},
            "response.created",
        ),
        sse(
            {"type": "response.in_progress", "response": _response(reply, "in_progress", ()), "sequence_number": 1},
            "response.in_progress",
        ),
    )
    reasoning: Final = (
        (
            sse(
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": _reasoning_item(reply),
                    "sequence_number": 2,
                },
                "response.output_item.added",
            ),
            sse(
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": _reasoning_item(reply),
                    "sequence_number": 3,
                },
                "response.output_item.done",
            ),
        )
        if reply.reasoning_tokens
        else ()
    )
    index: Final = len(reasoning) // 2
    start: Final = 2 + len(reasoning)
    body: Final = (
        _message_events(reply, index, start) if reply.tool_call is None else _function_events(reply, index, start)
    )
    sequence: Final = start + len(body)
    closing: Final = sse(
        {
            "type": "response.completed",
            "response": _response(reply, "completed", _output(reply)),
            "sequence_number": sequence,
        },
        "response.completed",
    )
    return grouped((*opening, *reasoning, *body, closing), RESPONSES_CHUNK_BYTES)
