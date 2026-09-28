"""Traffic matrix and sink doubles for the callback credential slots.

- ``upstream(request)``: provider double for every endpoint in ``ENDPOINTS``: OpenAI chat (plain
  and SSE) and OpenAI Responses (``/v1/messages`` reaches it as chat). A body carrying
  ``PROVIDER_4XX`` gets HTTP 400 and one carrying ``PROVIDER_5XX`` gets HTTP 500. The sensitivity
  marker found in the body is echoed back.
- ``langfuse_sink`` / ``datadog_sink``: Langfuse OTLP ingest and Datadog intake doubles.
- ``send(gateway, key, endpoint, model, text, extra)``: one client call per endpoint.
- ``spend_request_id(marker)``: the spend row written for the request carrying ``marker``.
- ``wait_for_sink(recorder, marker)``: bounded wait until a sink received the marker (gzip aware).
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Mapping
from typing import Final

import httpx
from integration._support.client import Gateway, eventually, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request
from integration.security._canary import Canary, find_canary
from integration.security._sinks import PROVIDER_4XX, Recorder
from pydantic import JsonValue

PROVIDER_5XX: Final = "canary-provider-5xx"
ENDPOINTS: Final = ("chat", "chat_stream", "messages", "responses")
OUTCOMES: Final = ("success", "provider_4xx", "provider_5xx")
EXPECTED_STATUS: Final = {"success": 200, "provider_4xx": 400, "provider_5xx": 500}
LANGFUSE_PUBLIC_KEY: Final = "pk-lf-canary-public"
_MARKER: Final = re.compile(rb"lkc-M0-[0-9a-f]{32}")


def _echo(body: bytes) -> str:
    found: Final = _MARKER.search(body)
    return "echo " + (found.group().decode() if found else "none")


def _failure(body: bytes) -> Reply | None:
    if PROVIDER_4XX.encode() in body:
        return Reply(
            status=400,
            body=b'{"error":{"type":"invalid_request_error","code":"canary_rejected","message":"rejected"}}',
        )
    if PROVIDER_5XX.encode() in body:
        return Reply(status=500, body=b'{"error":{"type":"server_error","message":"upstream exploded"}}')
    return None


def _chat(body: Mapping[str, JsonValue], text: str) -> Reply:
    identity: Final = f"chatcmpl-{uuid.uuid4().hex}"
    usage: Final = {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}
    if body.get("stream") is True:
        chunks: Final = (
            {"choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}]},
            {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": usage},
        )
        events: Final = b"".join(
            b"data: "
            + json.dumps(
                {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": "gpt-4o-mini", **chunk}
            ).encode()
            + b"\n\n"
            for chunk in chunks
        )
        return Reply(body=events + b"data: [DONE]\n\n", content_type="text/event-stream")
    return Reply(
        body=json.dumps(
            {
                "id": identity,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": usage,
            }
        ).encode()
    )


def _responses(text: str) -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": f"resp_{uuid.uuid4().hex}",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-4o-mini",
                "output": [
                    {
                        "type": "message",
                        "id": f"msg_{uuid.uuid4().hex}",
                        "status": "completed",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "usage": {"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
            }
        ).encode()
    )


def upstream(request: Request) -> Reply:
    failure: Final = _failure(request.body)
    if failure is not None:
        return failure
    text: Final = _echo(request.body)
    if request.target.split("?", 1)[0].endswith("/responses"):
        return _responses(text)
    return _chat(json.loads(request.body or b"{}"), text)


def langfuse_sink(request: Request) -> Reply:
    if request.method == "GET" and request.target.startswith("/api/public/projects"):
        return Reply(body=b'{"data":[{"id":"canary-project","name":"canary"}]}')
    return Reply(body=b"", content_type="application/x-protobuf")


def datadog_sink(request: Request) -> Reply:
    return Reply(status=202, body=b"{}")


def body_for(endpoint: str, model: str, text: str) -> dict[str, JsonValue]:
    if endpoint == "responses":
        return {"model": model, "input": text}
    if endpoint == "messages":
        return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}]}
    return {
        "model": model,
        "messages": [{"role": "user", "content": text}],
        **({"stream": True, "stream_options": {"include_usage": True}} if endpoint == "chat_stream" else {}),
    }


def send(
    gateway: Gateway, key: str, endpoint: str, model: str, text: str, extra: Mapping[str, JsonValue] | None = None
) -> httpx.Response:
    path: Final = {"responses": "/v1/responses", "messages": "/v1/messages"}.get(endpoint, "/v1/chat/completions")
    return gateway.request("POST", path, {**body_for(endpoint, model, text), **(extra or {})}, key=key)


def outcome_text(slot: str, marker: Canary, outcome: str) -> str:
    trigger: Final = {"success": "", "provider_4xx": f" {PROVIDER_4XX}", "provider_5xx": f" {PROVIDER_5XX}"}[outcome]
    return f"slot {slot} {marker.value}{trigger}"


def spend_request_id(marker: Canary) -> str:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT request_id FROM "LiteLLM_SpendLogs" WHERE proxy_server_request::text LIKE %s',
            (f"%{marker.core}%",),
        ),
        lambda found: len(found) >= 1,
        seconds=70,
    )
    return string_value(rows[0]["request_id"])


def wait_for_sink(recorder: Recorder, marker: Canary, seconds: float = 40) -> tuple[Request, ...]:
    return eventually(
        lambda: tuple(request for request in recorder.requests() if find_canary(request.body, (marker,))),
        bool,
        seconds=seconds,
    )
