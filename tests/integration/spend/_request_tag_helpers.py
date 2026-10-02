import json
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import yaml
from integration._support.database import read_rows
from integration._support.wire import Reply, Request

ANTHROPIC_MODEL: Final = "claude-sonnet-4-5-20250929"
OPENAI_MODEL: Final = "gpt-4o-mini"
GEMINI_MODEL: Final = "gemini-2.5-flash"
HEADERS: Final = {"user-agent": "claude-cli/2.0.0", "x-tenant-id": "tenant-a"}
T3: Final = ["User-Agent: claude-cli", "User-Agent: claude-cli/2.0.0", "x-tenant-id: tenant-a"]
ANTHROPIC_SONNET_BODY: Final = {
    "id": "msg_synthetic",
    "type": "message",
    "role": "assistant",
    "model": ANTHROPIC_MODEL,
    "content": [{"type": "text", "text": "tagged"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 10, "output_tokens": 2},
}
CHAT_COMPLETION_BODY: Final = {
    "id": "chatcmpl-synthetic",
    "object": "chat.completion",
    "created": 1,
    "model": OPENAI_MODEL,
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "tagged"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
}
RESPONSES_BODY: Final = {
    "id": "resp_synthetic",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": OPENAI_MODEL,
    "output": [
        {
            "type": "message",
            "id": "msg_synthetic",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "tagged", "annotations": []}],
        }
    ],
    "usage": {
        "input_tokens": 10,
        "output_tokens": 2,
        "total_tokens": 12,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens_details": {"reasoning_tokens": 0},
    },
}
GEMINI_BODY: Final = {
    "candidates": [
        {
            "content": {"parts": [{"text": "tagged"}], "role": "model"},
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 2, "totalTokenCount": 12},
}
ANTHROPIC_STREAM_EVENTS: Final = (
    {
        "type": "message_start",
        "message": {**ANTHROPIC_SONNET_BODY, "content": [], "usage": {"input_tokens": 10, "output_tokens": 1}},
    },
    {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "tagged"}},
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn", "stop_sequence": None},
        "usage": {"output_tokens": 2},
    },
    {"type": "message_stop"},
)
OPENAI_STREAM_CHUNKS: Final = (
    {
        "id": CHAT_COMPLETION_BODY["id"],
        "object": "chat.completion.chunk",
        "created": 1,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "tagged"}, "finish_reason": None}],
    },
    {
        "id": CHAT_COMPLETION_BODY["id"],
        "object": "chat.completion.chunk",
        "created": 1,
        "model": OPENAI_MODEL,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    },
    {
        "id": CHAT_COMPLETION_BODY["id"],
        "object": "chat.completion.chunk",
        "created": 1,
        "model": OPENAI_MODEL,
        "choices": [],
        "usage": CHAT_COMPLETION_BODY["usage"],
    },
)


def _message_id() -> str:
    return "msg_" + uuid.uuid4().hex


def _sse_frames(events: tuple[dict, ...]) -> tuple[bytes, ...]:
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


def provider_reply(request: Request) -> Reply:
    """Scripted edge for every route the audit drives: anthropic messages, openai chat completions and
    responses, gemini generateContent. Error bodies keyed off the sentinel model name."""
    body: Final = json.loads(request.body) if request.body else {}
    if body.get("model") == "claude-nonexistent-model":
        return Reply(
            status=400,
            body=json.dumps(
                {"error": {"type": "invalid_request_error", "message": "model: claude-nonexistent-model"}}
            ).encode(),
        )
    if request.target == "/v1/messages":
        identity: Final = _message_id()
        if body.get("stream") is True:
            events: Final = tuple(
                {**event, "message": {**event.get("message", {}), "id": identity}} if "message" in event else event
                for event in ANTHROPIC_STREAM_EVENTS
            )
            return Reply(content_type="text/event-stream", chunks=_sse_frames(events))
        return Reply(body=json.dumps({**ANTHROPIC_SONNET_BODY, "id": identity}).encode())
    if request.target == "/v1/chat/completions":
        identity = "chatcmpl_" + uuid.uuid4().hex
        if body.get("stream") is True:
            frames: Final = tuple(
                f"data: {json.dumps({**chunk, 'id': identity})}\n\n".encode() for chunk in OPENAI_STREAM_CHUNKS
            ) + (b"data: [DONE]\n\n",)
            return Reply(content_type="text/event-stream", chunks=frames)
        return Reply(body=json.dumps({**CHAT_COMPLETION_BODY, "id": identity}).encode())
    if request.target == "/v1/responses":
        identity = "resp_" + uuid.uuid4().hex
        message: Final = _message_id()
        completed_body: Final = {
            **RESPONSES_BODY,
            "id": identity,
            "output": [{**RESPONSES_BODY["output"][0], "id": message}],
        }
        if body.get("stream") is True:
            created: Final = {
                "type": "response.created",
                "response": {**completed_body, "status": "in_progress", "output": [], "usage": None},
            }
            delta: Final = {
                "type": "response.output_text.delta",
                "item_id": message,
                "output_index": 0,
                "content_index": 0,
                "delta": "tagged",
            }
            completed: Final = {"type": "response.completed", "response": completed_body}
            return Reply(content_type="text/event-stream", chunks=_sse_frames((created, delta, completed)))
        return Reply(body=json.dumps(completed_body).encode())
    if request.target.endswith(":generateContent") or request.target.endswith(":streamGenerateContent"):
        return Reply(body=json.dumps(GEMINI_BODY).encode())
    raise AssertionError(f"unexpected upstream target {request.target}")


def provider_env(url: str) -> dict[str, str]:
    return {
        "ANTHROPIC_API_BASE": url,
        "ANTHROPIC_API_KEY": "synthetic-anthropic-key",
        "OPENAI_API_BASE": url,
        "OPENAI_API_KEY": "synthetic-openai-key",
        "GEMINI_API_BASE": url,
        "GEMINI_API_KEY": "synthetic-gemini-key",
    }


def write_config(directory: Path, mutations: dict, name: str = "spend-tag-headers.yaml") -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    for section, values in mutations.items():
        if isinstance(values, dict) and isinstance(config.get(section), dict):
            config[section].update(values)
        else:
            config[section] = values
    path: Final = directory / name
    path.write_text(yaml.safe_dump(config))
    return path


def tags_of(row: dict) -> list:
    value: Final = row["request_tags"]
    return json.loads(value) if isinstance(value, str) else value


def tags_by_key(key: str) -> list[list]:
    rows: Final = read_rows(
        'SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (sha256(key.encode()).hexdigest(),)
    )
    return [tags_of(row) for row in rows]


def tags_by_id(request_id: str) -> list[list]:
    rows: Final = read_rows('SELECT request_tags FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,))
    return [tags_of(row) for row in rows]


def row_by_id(request_id: str) -> list[dict]:
    return read_rows(
        'SELECT request_id, call_type, request_tags, status, spend FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
        (request_id,),
    )


def unique_marker() -> str:
    return "tagprobe" + uuid.uuid4().hex[:12]
