"""Shared rig for the `test_dec1_anthropic2_*` files: newer Anthropic API features through the proxy.

Round 2 of the Anthropic contracts reuses the round 1 peer (`dec1_anthropic_support.peer`), which records every
request the proxy sends to an owned HTTP server playing api.anthropic.com, and adds what the newer features need:
a `/v1/messages` caller (the Rust-native route since #51119), a `/v1/messages/count_tokens` caller, an Anthropic
event-stream reader, canned bodies for the Files API, and the synthetic MCP server, container and skill ids the
tests share. Nothing here patches the product; every expected value in the tests is a literal read from the code.

Every prompt carries a fresh `marker()` because the shared proxy has a Redis response cache on.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from typing import Final

import httpx
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value
from integration._support.wire import Reply
from integration.providers.dec1_anthropic_support import (
    LANDING_SECONDS,
    Call,
    Respond,
    default_respond,
    json_reply,
    marker,
    message_json,
)
from pydantic import JsonValue

MCP_URL: Final = "https://mcp.dec1.invalid/sse"
MCP_NAME: Final = "dec1-docs"
MCP_TOKEN: Final = "dec1-synthetic-mcp-token"
CONTAINER_ID: Final = "container_dec1reuse"
SKILL_ID: Final = "skill_dec1report"
FILE_ID: Final = "file_011dec1synthetic"
FILE_BYTES: Final = b"dec1 synthetic file body\n"
FILES_BETA: Final = "files-api-2025-04-14"
MCP_BETA: Final = "mcp-client-2025-04-04"
SKILLS_BETA: Final = "skills-2025-10-02"
CODE_EXECUTION_BETA: Final = "code-execution-2025-08-25"
CONTEXT_MANAGEMENT_BETA: Final = "context-management-2025-06-27"
COMPACT_BETA: Final = "compact-2026-01-12"
WEB_FETCH_BETA: Final = "web-fetch-2025-09-10"
TOKEN_COUNTING_BETA: Final = "token-counting-2024-11-01"
TOKEN_EFFICIENT_BETA: Final = "token-efficient-tools-2025-02-19"
CONTEXT_1M_BETA: Final = "context-1m-2025-08-07"
EXTENDED_CACHE_TTL_BETA: Final = "extended-cache-ttl-2025-04-11"
CLEAR_TOOL_USES: Final[Mapping[str, JsonValue]] = {
    "type": "clear_tool_uses_20250919",
    "trigger": {"type": "input_tokens", "value": 30000},
    "keep": {"type": "tool_uses", "value": 3},
}
MEMORY_TOOL: Final[Mapping[str, JsonValue]] = {"type": "memory_20250818", "name": "memory"}
WEB_FETCH_TOOL: Final[Mapping[str, JsonValue]] = {"type": "web_fetch_20250910", "name": "web_fetch", "max_uses": 2}
CODE_EXECUTION_TOOL: Final[Mapping[str, JsonValue]] = {"type": "code_execution_20250825", "name": "code_execution"}
SEARCH_RESULT: Final[Mapping[str, JsonValue]] = {
    "type": "search_result",
    "source": "https://kb.dec1.invalid/articles/7",
    "title": "Synthetic article seven",
    "content": [{"type": "text", "text": "The synthetic widget ships in blue."}],
    "citations": {"enabled": True},
}
CITED_DOCUMENT: Final[Mapping[str, JsonValue]] = {
    "type": "document",
    "source": {"type": "text", "media_type": "text/plain", "data": "The synthetic widget ships in blue."},
    "title": "Widget sheet",
    "context": "Synthetic catalogue entry",
    "citations": {"enabled": True},
}


def mcp_server(**extra: JsonValue) -> dict[str, JsonValue]:
    """An Anthropic `mcp_servers` entry for the synthetic MCP server."""
    return {"type": "url", "url": MCP_URL, "name": MCP_NAME, **extra}


def messages_body(model: str, text: str | None = None, **fields: JsonValue) -> dict[str, JsonValue]:
    """A minimal Anthropic Messages request for `model`; `fields` override or add top-level keys."""
    return {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": text or marker()}],
        **fields,
    }


def native(
    gateway: Gateway,
    body: Mapping[str, JsonValue],
    *,
    headers: Mapping[str, str] | None = None,
    key: str | None = None,
) -> httpx.Response:
    """POST `/v1/messages`, the Rust-native route for `anthropic/` deployments."""
    return gateway.request("POST", "/v1/messages", body, key=key, headers=headers)


def count(
    gateway: Gateway,
    body: Mapping[str, JsonValue],
    *,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """POST `/v1/messages/count_tokens`."""
    return gateway.request("POST", "/v1/messages/count_tokens", body, headers=headers)


def native_stream(
    gateway: Gateway, body: Mapping[str, JsonValue], *, headers: Mapping[str, str] | None = None
) -> tuple[httpx.Headers, str]:
    """POST a streaming `/v1/messages` request and return the response headers and the whole body text."""
    with gateway.client.stream(
        "POST",
        "/v1/messages",
        json={**body, "stream": True},
        headers={"Authorization": f"Bearer {gateway.key}", **(headers or {})},
    ) as response:
        text: Final = "".join(response.iter_text())
        assert response.status_code == 200, text
        return response.headers, text


def sse_events(text: str) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    """Every named Anthropic event of an SSE body except pings, as (event name, decoded data)."""
    frames: Final = tuple(frame for frame in text.replace("\r\n", "\n").split("\n\n") if frame.strip())
    named: Final = tuple(
        (
            next((line.removeprefix("event:").strip() for line in frame.splitlines() if line.startswith("event:")), ""),
            next((line.removeprefix("data:").strip() for line in frame.splitlines() if line.startswith("data:")), ""),
        )
        for frame in frames
    )
    return tuple((event, JSON_OBJECT.validate_json(data)) for event, data in named if event != "ping" and data)


def body_of(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def provider_fields(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """`choices[0].message.provider_specific_fields` of a chat completion."""
    choices: Final = body["choices"]
    assert isinstance(choices, list) and choices, body
    return object_value(object_value(object_value(choices[0])["message"])["provider_specific_fields"])


def usage_of(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(body["usage"])


def replying(
    content: Sequence[JsonValue] | None = None, *, headers: Mapping[str, str] | None = None, **fields: JsonValue
) -> Respond:
    """A responder answering every POST with one message (optional `content`, top-level `fields`, headers)."""

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return default_respond(call)
        if call.path.endswith("/count_tokens"):
            return default_respond(call)
        return json_reply(message_json(call.body, content=content, **fields), headers=headers)  # type: ignore[arg-type]  # test-local keyword passthrough

    return respond


def file_object(file_id: str = FILE_ID, **extra: JsonValue) -> dict[str, JsonValue]:
    """An Anthropic Files API file object, as `POST /v1/files` and `GET /v1/files/{id}` answer it."""
    return {
        "id": file_id,
        "type": "file",
        "filename": "dec1-notes.txt",
        "mime_type": "text/plain",
        "size_bytes": len(FILE_BYTES),
        "created_at": "2026-11-17T08:30:00Z",
        "downloadable": True,
        **extra,
    }


def files_respond(call: Call) -> Reply:
    """The Files API surface of the peer, falling back to the round 1 canned answers."""
    if call.path == "/v1/files" and call.method == "POST":
        return json_reply(file_object())
    if call.path == "/v1/files" and call.method == "GET":
        return json_reply({"data": [file_object()], "has_more": False, "first_id": FILE_ID, "last_id": FILE_ID})
    if call.path == f"/v1/files/{FILE_ID}/content" and call.method == "GET":
        return Reply(body=FILE_BYTES, content_type="text/plain")
    if call.path == f"/v1/files/{FILE_ID}" and call.method == "GET":
        return json_reply(file_object())
    if call.path == f"/v1/files/{FILE_ID}" and call.method == "DELETE":
        return json_reply({"id": FILE_ID, "type": "file_deleted"})
    if call.path.startswith("/v1/skills") and call.method == "GET":
        return json_reply({"data": [{"id": SKILL_ID, "type": "skill", "display_title": "Report"}], "has_more": False})
    return default_respond(call)


def multipart_field(raw: bytes, name: str) -> bytes | None:
    """The value of one form field in a multipart body, or None. Good enough for the small bodies tests send."""
    marker_bytes: Final = f'name="{name}"'.encode()
    start: Final = raw.find(marker_bytes)
    if start < 0:
        return None
    value_start: Final = raw.find(b"\r\n\r\n", start) + 4
    value_end: Final = raw.find(b"\r\n--", value_start)
    return raw[value_start:value_end]


def spend_by_group(gateway: Gateway, model_group: str, *, seconds: float = LANDING_SECONDS) -> float:
    """The `spend` of the one SpendLogs row of a model group created for one test."""
    rows: Final = eventually(
        lambda: gateway.rows('SELECT spend FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model_group,)),
        lambda found: len(found) == 1,
        seconds=seconds,
    )
    value: Final = rows[0]["spend"]
    assert isinstance(value, (int, float, str)), rows
    return float(value)


def dumps(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True)


def fresh_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:16]}"
