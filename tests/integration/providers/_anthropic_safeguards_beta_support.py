from __future__ import annotations

import json
import re
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from queue import SimpleQueue
from types import MappingProxyType
from typing import Final

import yaml
from integration._support.anthropic_sse import (
    CONTENT_BLOCK_START,
    error_body,
    message_start,
    sse,
    text_delta,
)
from integration._support.client import Gateway, object_value
from integration._support.vertex import service_account_json
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue, TypeAdapter

DANGEROUS_TOOL_USE: Final = "dangerous-tool-use-2026-09-03"
WEB_SEARCH: Final = "web-search-2025-03-05"
CLIENT_BETAS: Final = "interleaved-thinking-2025-05-14,claude-code-20250219,interleaved-thinking-2025-05-14"
MERGED_BETAS: Final = f"claude-code-20250219,{DANGEROUS_TOOL_USE},interleaved-thinking-2025-05-14"
BACKEND: Final = "claude-sonnet-4-6"
ANTHROPIC_DEPLOYMENT: Final = "safeguarded-claude"
VERTEX_DEPLOYMENT: Final = "safeguarded-vertex-claude"
DEEPSEEK_DEPLOYMENT: Final = "safeguarded-deepseek"
ANTHROPIC_KEY: Final = "synthetic-anthropic-key"
DEEPSEEK_KEY: Final = "synthetic-deepseek-key"
PROJECT: Final = "scripted-project"
LOCATION: Final = "us-east5"
_VERTEX_MODEL_PATH: Final = f"/v1/projects/{PROJECT}/locations/{LOCATION}/publishers/anthropic/models/{BACKEND}"
VERTEX_TARGET: Final = f"{_VERTEX_MODEL_PATH}:rawPredict"
VERTEX_STREAM_TARGET: Final = f"{_VERTEX_MODEL_PATH}:streamRawPredict?alt=sse"
ANTHROPIC_TARGET: Final = "/v1/messages"
DEEPSEEK_TARGET: Final = "/anthropic/v1/messages"
SCRIPTED_TARGETS: Final = frozenset({ANTHROPIC_TARGET, DEEPSEEK_TARGET, VERTEX_TARGET, VERTEX_STREAM_TARGET})
REPLY_TEXT: Final = "safeguards beta control"
EXTRA_INPUTS_MESSAGE: Final = "safeguards: Extra inputs are not permitted"
NOT_A_LIST_MESSAGE: Final = "safeguards: Input should be a valid array"
OVERLOADED_MESSAGE: Final = "Overloaded"
SAFEGUARDS: Final[tuple[JsonValue, ...]] = (
    {
        "type": "dangerous_tool_use",
        "classifier_context": {"v": 1, "permission_mode": "auto"},
    },
)
SAFEGUARD_RESULTS: Final[tuple[JsonValue, ...]] = (
    {"type": "dangerous_tool_use", "status": {"type": "available", "tool_uses": {}}},
)
BASH_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {
        "name": "Bash",
        "description": "Run a shell command",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
)
WEB_SEARCH_TOOL: Final[Mapping[str, JsonValue]] = MappingProxyType(
    {"type": "web_search_20250305", "name": "web_search", "max_uses": 1}
)
RUST_ROUTE_HEADER: Final = "x-litellm-rust"
_MARKER: Final = re.compile(rb"marker-[0-9a-f]{32}")
_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_STARTED_WORKER: Final = "Started server process ["


@dataclass(frozen=True, slots=True)
class Delivery:
    target: str
    beta: str | None
    safeguards: JsonValue
    marker: str
    authorization: str | None
    api_key: str | None


def new_marker() -> str:
    return f"marker-{uuid.uuid4().hex}"


def marker_of(request: Request) -> str:
    found: Final = tuple(match.group(0) for match in _MARKER.finditer(request.body))
    return found[-1].decode() if found else "unmarked"


def body_of(request: Request) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(request.body)


def json_object(text: str) -> dict[str, JsonValue]:
    return _JSON_OBJECT.validate_json(text)


def betas_of(request: Request) -> frozenset[str]:
    return frozenset(part.strip() for part in request.headers.get("anthropic-beta", "").split(",") if part.strip())


def started_workers(log: Path) -> tuple[int, ...]:
    return tuple(
        int(line.rsplit("[", 1)[1].strip(" ]\r")) for line in log.read_text().splitlines() if _STARTED_WORKER in line
    )


def _streams(request: Request) -> bool:
    return request.target == VERTEX_STREAM_TARGET or body_of(request).get("stream") is True


def _results(safeguards: JsonValue) -> list[JsonValue] | None:
    if safeguards is None:
        return None
    return list(SAFEGUARD_RESULTS) if safeguards else []


def _message(identity: str, results: list[JsonValue] | None) -> dict[str, JsonValue]:
    verdict: Final[dict[str, JsonValue]] = {} if results is None else {"safeguard_results": results}
    return {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": BACKEND,
        "content": [{"type": "text", "text": REPLY_TEXT}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 5, "output_tokens": 3},
        **verdict,
    }


def _stream(identity: str, results: list[JsonValue] | None) -> tuple[bytes, ...]:
    verdict: Final[dict[str, JsonValue]] = {} if results is None else {"safeguard_results": results}
    return (
        message_start(identity, BACKEND),
        CONTENT_BLOCK_START,
        text_delta(REPLY_TEXT),
        sse("content_block_stop", {"type": "content_block_stop", "index": 0}),
        sse(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None, **verdict},
                "usage": {"output_tokens": 3},
            },
        ),
        sse("message_stop", {"type": "message_stop"}),
    )


def anthropic_peer(request: Request) -> Reply:
    if request.target not in SCRIPTED_TARGETS:
        return Reply(
            status=404,
            body=json.dumps({"error": f"unscripted target {request.target}"}).encode(),
        )
    if request.target == DEEPSEEK_TARGET:
        return Reply(body=json.dumps(_message(f"msg_{marker_of(request)}", None)).encode())
    body: Final = body_of(request)
    safeguards: Final = body.get("safeguards")
    if "safeguards" in body and DANGEROUS_TOOL_USE not in betas_of(request):
        return Reply(status=400, body=error_body(400, EXTRA_INPUTS_MESSAGE))
    if safeguards is not None and not isinstance(safeguards, list):
        return Reply(status=400, body=error_body(400, NOT_A_LIST_MESSAGE))
    identity: Final = f"msg_{marker_of(request)}"
    results: Final = _results(safeguards)
    if _streams(request):
        return Reply(content_type="text/event-stream", chunks=_stream(identity, results))
    return Reply(body=json.dumps(_message(identity, results)).encode())


def outage_peer(outage: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if outage.is_set():
            return Reply(status=529, body=error_body(529, OVERLOADED_MESSAGE))
        return anthropic_peer(request)

    return respond


def holding_peer(held: SimpleQueue[str], release: threading.Event) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        held.put(marker_of(request))
        assert release.wait(timeout=60), "Held request was never released"
        return anthropic_peer(request)

    return respond


def deliveries(wire: Wire) -> tuple[Delivery, ...]:
    return tuple(
        Delivery(
            request.target,
            request.headers.get("anthropic-beta"),
            body_of(request).get("safeguards"),
            marker_of(request),
            request.headers.get("authorization"),
            request.headers.get("x-api-key"),
        )
        for request in wire.drain()
    )


def one_delivery(wire: Wire, marker: str) -> Delivery:
    (sent,) = deliveries(wire)
    assert sent.marker == marker, sent
    return sent


def rust_routed(headers: Mapping[str, str]) -> bool:
    return headers.get(RUST_ROUTE_HEADER) == "true"


def messages_body(model: str, marker: str, *, stream: bool = False) -> dict[str, JsonValue]:
    return {
        "model": model,
        "max_tokens": 64,
        "stream": stream,
        "messages": [{"role": "user", "content": marker}],
        "tools": [dict(BASH_TOOL)],
    }


def safeguarded(body: Mapping[str, JsonValue], safeguards: JsonValue) -> dict[str, JsonValue]:
    return {**body, "safeguards": safeguards}


def chat_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": marker}]}


def responses_body(model: str, marker: str) -> dict[str, JsonValue]:
    return {"model": model, "input": marker}


def owned_config(directory: Path, gateway: Gateway, api_base: str) -> Path:
    config: Final = _JSON_OBJECT.validate_python(
        yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    )
    merged: Final = {
        **config,
        "model_list": [
            {
                "model_name": ANTHROPIC_DEPLOYMENT,
                "litellm_params": {
                    "model": f"anthropic/{BACKEND}",
                    "api_base": api_base,
                    "api_key": ANTHROPIC_KEY,
                },
            },
            {
                "model_name": VERTEX_DEPLOYMENT,
                "litellm_params": {
                    "model": f"vertex_ai/{BACKEND}",
                    "api_base": api_base,
                    "vertex_project": PROJECT,
                    "vertex_location": LOCATION,
                    "vertex_credentials": service_account_json(PROJECT, gateway.upstream_url.rstrip("/")),
                },
            },
            {
                "model_name": DEEPSEEK_DEPLOYMENT,
                "litellm_params": {
                    "model": "deepseek/deepseek-chat",
                    "api_base": api_base,
                    "api_key": DEEPSEEK_KEY,
                },
            },
        ],
        "router_settings": {
            **object_value(config["router_settings"]),
            "num_retries": 0,
        },
    }
    path: Final = directory / f"safeguards-beta-{uuid.uuid4().hex}.yaml"
    path.write_text(yaml.safe_dump(merged))
    return path
