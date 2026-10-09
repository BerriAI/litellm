from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlsplit

import yaml
from integration._support import responses_vendor as rv
from integration._support.wire import Reply, Request
from pydantic import JsonValue

TOKEN: Final = "synthetic-chatgpt-token"
ACCOUNT: Final = "acct-synthetic"
PROBE_CALLBACK: Final = "integration._support.agentic_probe.probe"
INPUT_MUST_BE_A_LIST: Final = "Input must be a list"
UNAUTHORIZED: Final = "Unauthorized"
UNAUTHORIZED_DIRECTIVE: Final = "codex-unauthorized"
FAILED_DIRECTIVE: Final = "codex-failed"
INCOMPLETE_DIRECTIVE: Final = "codex-incomplete"
INCOMPLETE_PAUSE_SECONDS: Final = 0.5
INCOMPLETE_CHUNKS: Final = 4
USAGE: Final[Mapping[str, JsonValue]] = {
    "input_tokens": 30,
    "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
    "output_tokens": 5,
    "output_tokens_details": {"reasoning_tokens": 0},
    "total_tokens": 35,
}
_TOTAL_KEYS: Final = ("input_tokens", "output_tokens", "total_tokens")
FORWARDED_KEYS: Final = frozenset(
    {
        "model",
        "input",
        "instructions",
        "stream",
        "store",
        "include",
        "tools",
        "tool_choice",
        "reasoning",
        "previous_response_id",
        "truncation",
    }
)


def login(directory: Path) -> Path:
    chatgpt: Final = directory / "chatgpt"
    chatgpt.mkdir()
    (chatgpt / "auth.json").write_text(
        json.dumps({"access_token": TOKEN, "account_id": ACCOUNT, "expires_at": time.time() + 3600})
    )
    return chatgpt


def proxy_config(directory: Path, *, probe: bool) -> Path:
    stock: Final = rv.JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    litellm_settings: Final = rv.JSON_OBJECT.validate_python(stock["litellm_settings"])
    router_settings: Final = rv.JSON_OBJECT.validate_python(stock.get("router_settings") or {})
    config: Final[Mapping[str, JsonValue]] = {
        **stock,
        "litellm_settings": {**litellm_settings, "callbacks": [PROBE_CALLBACK]} if probe else litellm_settings,
        "router_settings": {**router_settings, "num_retries": 0},
    }
    path: Final = directory / "chatgpt-codex-rig.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def totals(usage: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    return {key: usage[key] for key in _TOTAL_KEYS}


def failure_message(marker: str | None) -> str:
    return f"codex failed marker-{marker}"


def forwarded(request: Request, marker: str, *, stream: bool = True) -> Mapping[str, JsonValue]:
    assert request.method == "POST", request.method
    assert urlsplit(request.target).path.endswith("/responses"), request.target
    assert request.headers.get("authorization") == f"Bearer {TOKEN}", request.headers
    assert request.headers.get("chatgpt-account-id") == ACCOUNT, request.headers
    body: Final = rv.JSON_OBJECT.validate_json(request.body)
    assert set(body) <= FORWARDED_KEYS, sorted(body)
    assert (body["stream"], body["store"]) == (stream, False), body
    assert isinstance(body["instructions"], str) and body["instructions"], body
    assert rv.newest_marker(json.dumps(body["input"])) == marker, body["input"]
    return body


def _detail(status: int, detail: str) -> Reply:
    return Reply(status=status, body=json.dumps({"detail": detail}).encode())


def _stream(
    frames: Sequence[Mapping[str, JsonValue]],
    *,
    abort_after: int | None = None,
    pause: float = 0,
    gate: threading.Event | None = None,
) -> Reply:
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(rv.sse(frame) for frame in frames),
        abort_after=abort_after,
        pause_between_chunks=pause,
        gate_after_first=gate,
    )


def _response(model: str, tag: str) -> Mapping[str, JsonValue]:
    return {
        "id": f"resp_{tag}",
        "object": "response",
        "created_at": 1,
        "status": "in_progress",
        "model": model,
        "output": [],
        "instructions": "You are a coding agent.",
        "metadata": {},
        "parallel_tool_calls": True,
        "temperature": 1.0,
        "tool_choice": "auto",
        "tools": [],
        "top_p": 1.0,
        "reasoning": {"effort": "medium", "summary": None},
        "text": {"format": {"type": "text"}, "verbosity": "medium"},
        "truncation": "disabled",
        "store": False,
        "background": False,
        "service_tier": "default",
    }


def _message(tag: str, text: str) -> Mapping[str, JsonValue]:
    return {
        "id": f"msg_{tag}",
        "type": "message",
        "status": "completed",
        "role": "assistant",
        "phase": "final_answer",
        "content": [{"type": "output_text", "annotations": [], "logprobs": [], "text": text}],
    }


def _frames(model: str, tag: str, text: str) -> tuple[Mapping[str, JsonValue], ...]:
    response: Final = _response(model, tag)
    message: Final = _message(tag, text)
    part: Final[Mapping[str, JsonValue]] = {"type": "output_text", "annotations": [], "logprobs": [], "text": ""}
    position: Final[Mapping[str, JsonValue]] = {"item_id": f"msg_{tag}", "output_index": 0, "content_index": 0}
    return (
        {"type": "response.created", "sequence_number": 0, "model": model, "response": dict(response)},
        {"type": "response.in_progress", "sequence_number": 1, "model": model, "response": dict(response)},
        {
            "type": "response.output_item.added",
            "sequence_number": 2,
            "output_index": 0,
            "model": model,
            "item": {**message, "status": "in_progress", "content": []},
        },
        {"type": "response.content_part.added", "sequence_number": 3, "model": model, **position, "part": dict(part)},
        {
            "type": "response.output_text.delta",
            "sequence_number": 4,
            "model": model,
            **position,
            "delta": text,
            "logprobs": [],
            "obfuscation": "",
        },
        {
            "type": "response.output_text.done",
            "sequence_number": 5,
            "model": model,
            **position,
            "text": text,
            "logprobs": [],
        },
        {
            "type": "response.content_part.done",
            "sequence_number": 6,
            "model": model,
            **position,
            "part": {**part, "text": text},
        },
        {
            "type": "response.output_item.done",
            "sequence_number": 7,
            "output_index": 0,
            "model": model,
            "item": dict(message),
        },
        {
            "type": "response.completed",
            "sequence_number": 8,
            "model": model,
            "response": {**response, "status": "completed", "usage": dict(USAGE), "completed_at": 2},
        },
    )


def _failed_frames(model: str, tag: str, marker: str | None) -> tuple[Mapping[str, JsonValue], ...]:
    response: Final = _response(model, tag)
    return (
        {"type": "response.created", "sequence_number": 0, "model": model, "response": dict(response)},
        {
            "type": "response.failed",
            "sequence_number": 1,
            "model": model,
            "response": {
                **response,
                "status": "failed",
                "error": {"code": "server_error", "message": failure_message(marker)},
            },
        },
    )


@dataclass(frozen=True, slots=True)
class CodexVendor:
    """The ChatGPT Codex backend as the proxy sees it: SSE only, and `input` must be a list of items."""

    pause_between_chunks: float = 0
    incomplete_gate: threading.Event | None = None

    def respond(self, request: Request) -> Reply:
        if request.method == "GET":
            return Reply(body=json.dumps({"object": "list", "data": [{"id": "gpt-5.5", "object": "model"}]}).encode())
        assert urlsplit(request.target).path.endswith("/responses"), request.target
        body: Final = rv.JSON_OBJECT.validate_json(request.body)
        items: Final = body.get("input")
        if not isinstance(items, list) or any(not isinstance(item, dict) for item in items):
            return _detail(400, INPUT_MUST_BE_A_LIST)
        text: Final = json.dumps(items)
        marker: Final = rv.newest_marker(text)
        model: Final = str(body["model"])
        tag: Final = uuid.uuid4().hex
        if UNAUTHORIZED_DIRECTIVE in text:
            return _detail(401, UNAUTHORIZED)
        if FAILED_DIRECTIVE in text:
            return _stream(_failed_frames(model, tag, marker))
        if INCOMPLETE_DIRECTIVE in text:
            return _stream(
                _frames(model, tag, rv.answer(marker)),
                abort_after=INCOMPLETE_CHUNKS,
                pause=INCOMPLETE_PAUSE_SECONDS,
                gate=self.incomplete_gate,
            )
        return _stream(_frames(model, tag, rv.answer(marker)), pause=self.pause_between_chunks)
