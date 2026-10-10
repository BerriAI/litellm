"""Shared payloads and wire helpers for the status-code follow-up audit cells.

Every cell drives a real proxy against the scripted upstream and reads the outbound provider body back from the
upstream's observation feed, so a cell asserts both what the caller got and what the provider received."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
from integration._support.client import JSON_OBJECT, Gateway, Scenario, gateway_from_environment, object_value
from integration._support.process import owned_proxy_process
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from pydantic import JsonValue

from tests.integration.cost_calculation.cost_tracking_case import (
    JsonResponse,
    SseResponse,
    StoredResponse,
)

ROUTER_DEFAULTS: Final[dict[str, JsonValue]] = {
    "prompt": "router default prompt",
    "input": "router default input",
    "max_tokens": 32,
}
CHAT: Final[dict[str, JsonValue]] = {
    "id": "chatcmpl-$UNIQUE_ID",
    "object": "chat.completion",
    "created": 1,
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "scripted"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}
RESPONSE: Final[dict[str, JsonValue]] = {
    "id": "resp_$UNIQUE_ID",
    "object": "response",
    "created_at": 1,
    "status": "completed",
    "model": "gpt-4o-mini",
    "output": [
        {
            "type": "message",
            "id": "msg_$UNIQUE_ID",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "scripted", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
}
MESSAGE: Final[dict[str, JsonValue]] = {
    "id": "msg_$UNIQUE_ID",
    "type": "message",
    "role": "assistant",
    "model": "claude-haiku-4-5",
    "content": [{"type": "text", "text": "scripted"}],
    "stop_reason": "end_turn",
    "stop_sequence": None,
    "usage": {"input_tokens": 1, "output_tokens": 1},
}
EMBEDDING: Final[dict[str, JsonValue]] = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}
MODERATION: Final[dict[str, JsonValue]] = {
    "id": "modr-$UNIQUE_ID",
    "model": "omni-moderation-latest",
    "results": [{"flagged": False, "categories": {}, "category_scores": {}}],
}
CHAT_FRAMES: Final = (
    'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":"streamed "},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    '"choices":[{"index":0,"delta":{"content":"response"},"finish_reason":null}]}',
    'data: {"id":"chatcmpl-$UNIQUE_ID","object":"chat.completion.chunk","created":1,"model":"gpt-4o-mini",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    "data: [DONE]",
)
RESPONSES_FRAMES: Final = (
    "event: response.created\n"
    'data: {"type":"response.created","sequence_number":0,"response":{"id":"resp_$UNIQUE_ID","object":"response",'
    '"created_at":1,"status":"in_progress","model":"gpt-4o-mini","output":[],"usage":null}}',
    "event: response.output_text.delta\n"
    'data: {"type":"response.output_text.delta","sequence_number":1,"item_id":"msg_$UNIQUE_ID","output_index":0,'
    '"content_index":0,"delta":"streamed response"}',
    "event: response.completed\n"
    'data: {"type":"response.completed","sequence_number":2,"response":{"id":"resp_$UNIQUE_ID","object":"response",'
    '"created_at":1,"status":"completed","model":"gpt-4o-mini","output":[{"type":"message","id":"msg_$UNIQUE_ID",'
    '"status":"completed","role":"assistant","content":[{"type":"output_text","text":"streamed response",'
    '"annotations":[]}]}],"usage":{"input_tokens":1,"output_tokens":2,"total_tokens":3}}}',
)
MESSAGE_FRAMES: Final = (
    "event: message_start\n"
    'data: {"type":"message_start","message":{"id":"msg_$UNIQUE_ID","type":"message","role":"assistant",'
    '"model":"claude-haiku-4-5","content":[],"stop_reason":null,"stop_sequence":null,'
    '"usage":{"input_tokens":1,"output_tokens":0}}}',
    "event: content_block_start\n"
    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
    "event: content_block_delta\n"
    'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"streamed response"}}',
    'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}',
    "event: message_delta\n"
    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":2}}',
    'event: message_stop\ndata: {"type":"message_stop"}',
)
UNSET: Final = object()
USER_MESSAGES: Final[list[JsonValue]] = [{"role": "user", "content": "Hello"}]
STREAM_TERMINALS: Final[Mapping[str, str]] = {
    "chat": "data: [DONE]",
    "responses": '"type":"response.completed"',
    "messages": "event: message_stop",
}
ENDPOINT_PATHS: Final[Mapping[str, str]] = {
    "chat": "/v1/chat/completions",
    "responses": "/v1/responses",
    "messages": "/v1/messages",
}
ENDPOINT_BODIES: Final[Mapping[str, dict[str, JsonValue]]] = {
    "chat": {"messages": USER_MESSAGES},
    "responses": {"input": "Hello", "store": False},
    "messages": {"messages": USER_MESSAGES, "max_tokens": 16},
}


def json_response(body: dict[str, JsonValue], status: int = 200) -> JsonResponse:
    return JsonResponse(content_type="application/json", body=body, status=status)


def sse_response(frames: tuple[str, ...]) -> SseResponse:
    return SseResponse(content_type="text/event-stream", frames=frames)


def openai_error(message: str, param: str) -> JsonResponse:
    return json_response(
        {"error": {"message": message, "type": "invalid_request_error", "param": param, "code": None}}, 400
    )


class Upstream:
    """Reads the scripted upstream's destructive observation feed and keeps every record it drained."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.items: tuple[dict[str, JsonValue], ...] = ()

    def drain(self) -> None:
        with httpx.Client(timeout=10, trust_env=False) as client:
            payload: Final = JSON_OBJECT.validate_python(
                client.get(f"{self.url}/__observations?include_method=true").json()
            )
        requests: Final = payload.get("requests")
        assert isinstance(requests, list)
        self.items = (*self.items, *(object_value(item) for item in requests if isinstance(item, dict)))

    def calls(self, identity: str) -> tuple[dict[str, JsonValue], ...]:
        self.drain()
        return tuple(
            item
            for item in self.items
            if f"/{identity}/" in str(item.get("path")) and not str(item.get("path", "")).endswith("/models")
        )


def drain_rig_upstream() -> None:
    """Drops every observation earlier cells left on the shared upstream, so a cell reads only its own calls."""
    Upstream(os.environ["INTEGRATION_UPSTREAM_URL"]).drain()


def assert_no_provider_call(gateway: Gateway, *identities: str) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    assert {identity: upstream.calls(identity) for identity in identities} == dict.fromkeys(identities, ())


def one_outbound(gateway: Gateway, identity: str) -> dict[str, JsonValue]:
    calls: Final = Upstream(gateway.upstream_url).calls(identity)
    assert len(calls) == 1, calls
    return object_value(calls[0]["body"])


def post(gateway: Gateway, path: str, body: Mapping[str, JsonValue], *, key: str | None = None) -> httpx.Response:
    return gateway.client.post(
        path,
        json=dict(body),
        headers={"Authorization": f"Bearer {gateway.key if key is None else key}"},
        timeout=30,
    )


def stream_lines(gateway: Gateway, path: str, body: Mapping[str, JsonValue]) -> tuple[int, tuple[str, ...]]:
    with gateway.client.stream(
        "POST", path, json=dict(body), headers={"Authorization": f"Bearer {gateway.key}"}, timeout=30
    ) as response:
        return response.status_code, tuple(response.iter_lines())


def error_body(response: httpx.Response) -> dict[str, JsonValue]:
    """The error object of an OpenAI-shaped or Anthropic-shaped error response."""
    return object_value(JSON_OBJECT.validate_python(response.json())["error"])


def invalid_request(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 400, response.text
    error: Final = error_body(response)
    assert error.get("type") == "invalid_request_error", response.text
    return error


def register(scenario: Scenario, prefix: str, response: StoredResponse) -> tuple[str, ScenarioHandle]:
    identity: Final = f"{prefix}-{uuid.uuid4().hex}"
    handle: Final = register_scenario(identity, response)
    scenario.cleanups.callback(delete_scenario, handle)
    return identity, handle


def deployment(
    scenario: Scenario, name: str, model: str, response: StoredResponse, **extra: JsonValue
) -> tuple[str, dict[str, JsonValue]]:
    identity, handle = register(scenario, name, response)
    return identity, {
        "model_name": name,
        "litellm_params": {"model": model, "api_base": handle.api_base(), "api_key": identity, **extra},
    }


@contextmanager
def owned_gateway(
    directory: Path,
    config: Mapping[str, JsonValue],
    *,
    workers: int = 1,
    environment: Mapping[str, str] | None = None,
) -> Iterator[Gateway]:
    """A proxy booted from this checkout with `config` and extra `environment`, sharing the rig's database, Redis
    and upstream."""
    path: Final = directory / "config.yaml"
    path.write_text(json.dumps(dict(config)), encoding="utf-8")
    with (
        gateway_from_environment() as gateway,
        owned_proxy_process(gateway, directory, environment or {}, config=path, workers=workers) as owned,
    ):
        yield Gateway(owned.gateway.client, owned.gateway.key, gateway.upstream_url)


def stream_finished(endpoint: str, lines: Sequence[str]) -> bool:
    """Whether the stream reached the endpoint's terminal frame."""
    return any(STREAM_TERMINALS[endpoint] in line for line in lines)


def _frame_text(frame: Mapping[str, JsonValue]) -> str:
    """The text one chat, responses or messages delta frame carries, or "" for any other frame."""
    choices: Final = frame.get("choices")
    first: Final = choices[0] if isinstance(choices, list) and choices else None
    chat_delta: Final = first.get("delta") if isinstance(first, dict) else None
    if isinstance(chat_delta, dict) and isinstance(chat_delta.get("content"), str):
        return str(chat_delta["content"])
    delta: Final = frame.get("delta")
    if frame.get("type") == "response.output_text.delta" and isinstance(delta, str):
        return delta
    if frame.get("type") == "content_block_delta" and isinstance(delta, dict) and delta.get("type") == "text_delta":
        return str(delta.get("text"))
    return ""


def assembled_text(lines: Sequence[str]) -> str:
    """The text a chat, responses or messages stream carried, joined across its delta frames."""
    return "".join(
        _frame_text(JSON_OBJECT.validate_json(line.removeprefix("data: ")))
        for line in lines
        if line.startswith("data: ") and line != "data: [DONE]"
    )
