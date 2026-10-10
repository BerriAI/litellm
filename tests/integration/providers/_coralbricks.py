from __future__ import annotations

import json
import re
import socket
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import anthropic
import httpx
import openai
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, Wire
from pydantic import JsonValue

PROVIDER: Final = "coralbricks"
MODEL: Final = "deepseek-v4.1-flash-fast"
MODELS: Final = (MODEL, "glm-5.3-fast")
API_KEY: Final = "cb_synthetic_integration_key"
ANSWER: Final = "coral ok"
NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
PROMPT_TOKENS: Final = 4357
CACHED_TOKENS: Final = 4352
CACHE_WRITE_TOKENS: Final = 5
COMPLETION_TOKENS: Final = 20
REASONING_TOKENS: Final = 12
COST_MAP: Final = Path(__file__).resolve().parents[3] / "model_prices_and_context_window.json"
LEAK_FIELDS: Final = frozenset(
    {"litellm_params", "litellm_logging_obj", "litellm_call_id", "litellm_metadata", "proxy_server_request"}
)
READINESS_TEXT: Final = "model readiness"
MARKER: Final = re.compile(r"marker-([0-9a-f]{32})")
USER_MESSAGES: Final[list[dict[str, JsonValue]]] = [{"role": "user", "content": "Say hello to the reef"}]
SPEND_COLUMNS: Final = (
    "SELECT request_id, model, custom_llm_provider, model_group, call_type, status, spend, prompt_tokens, "
    'completion_tokens, api_base FROM "LiteLLM_SpendLogs"'
)


@dataclass(frozen=True, slots=True)
class Rates:
    input: float
    output: float
    cache_write: float
    cache_read: float


def number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)), value
    return float(value)


def rates_of(model: str) -> Rates:
    row: Final = object_value(JSON_OBJECT.validate_json(COST_MAP.read_bytes())[f"{PROVIDER}/{model}"])
    assert row["litellm_provider"] == PROVIDER, row
    return Rates(
        input=number(row["input_cost_per_token"]),
        output=number(row["output_cost_per_token"]),
        cache_write=number(row["cache_creation_input_token_cost"]),
        cache_read=number(row["cache_read_input_token_cost"]),
    )


def expected_spend(model: str) -> float:
    rates: Final = rates_of(model)
    return (
        (PROMPT_TOKENS - CACHED_TOKENS - CACHE_WRITE_TOKENS) * rates.input
        + CACHED_TOKENS * rates.cache_read
        + CACHE_WRITE_TOKENS * rates.cache_write
        + COMPLETION_TOKENS * rates.output
    )


def chat_usage() -> dict[str, JsonValue]:
    return {
        "prompt_tokens": PROMPT_TOKENS,
        "completion_tokens": COMPLETION_TOKENS,
        "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
        "prompt_tokens_details": {
            "cached_tokens": CACHED_TOKENS,
            "cache_write_tokens": CACHE_WRITE_TOKENS,
            "billable_cache_write_tokens": CACHE_WRITE_TOKENS,
            "cache_write_blocks": 3,
        },
        "completion_tokens_details": {"reasoning_tokens": REASONING_TOKENS},
    }


def sse(payload: Mapping[str, JsonValue], event: str | None = None) -> bytes:
    prefix: Final = f"event: {event}\n" if event is not None else ""
    return f"{prefix}data: {json.dumps(payload)}\n\n".encode()


def chat_reply(identity: str, model: str, text: str = ANSWER) -> Reply:
    body: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": chat_usage(),
    }
    return Reply(body=json.dumps(body).encode())


def chat_stream_reply(identity: str, model: str, text: str = ANSWER, pause: float = 0) -> Reply:
    chunk: Final[dict[str, JsonValue]] = {"id": identity, "object": "chat.completion.chunk", "created": 1, "model": model}
    frames: Final[tuple[dict[str, JsonValue], ...]] = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}]},
        {**chunk, "choices": [{"index": 0, "delta": {"content": text}}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {**chunk, "choices": [], "usage": chat_usage()},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=(*(sse(frame) for frame in frames), b"data: [DONE]\n\n"),
        pause_between_chunks=pause,
    )


def responses_body(identity: str, model: str, text: str = ANSWER) -> dict[str, JsonValue]:
    return {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": model,
        "output": [
            {
                "id": f"msg_{identity}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {
            "input_tokens": PROMPT_TOKENS,
            "output_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            "input_tokens_details": {
                "cached_tokens": CACHED_TOKENS,
                "cache_write_tokens": CACHE_WRITE_TOKENS,
                "billable_cache_write_tokens": CACHE_WRITE_TOKENS,
            },
            "output_tokens_details": {"reasoning_tokens": REASONING_TOKENS},
        },
    }


def responses_reply(identity: str, model: str, stream: bool, text: str = ANSWER, pause: float = 0) -> Reply:
    body: Final = responses_body(identity, model, text)
    if not stream:
        return Reply(body=json.dumps(body).encode())
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "response.created", "sequence_number": 0, "response": {**body, "status": "in_progress", "output": []}},
        {
            "type": "response.output_text.delta",
            "sequence_number": 1,
            "item_id": f"msg_{identity}",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {"type": "response.completed", "sequence_number": 2, "response": body},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(sse(event, string_value(event["type"])) for event in events),
        pause_between_chunks=pause,
    )


def messages_usage(output_tokens: int) -> dict[str, JsonValue]:
    return {
        "input_tokens": PROMPT_TOKENS - CACHED_TOKENS - CACHE_WRITE_TOKENS,
        "cache_read_input_tokens": CACHED_TOKENS,
        "cache_creation_input_tokens": CACHE_WRITE_TOKENS,
        "output_tokens": output_tokens,
    }


def messages_reply(identity: str, model: str, text: str = ANSWER) -> Reply:
    body: Final[dict[str, JsonValue]] = {
        "id": identity,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": messages_usage(COMPLETION_TOKENS),
    }
    return Reply(body=json.dumps(body).encode())


def messages_stream_reply(identity: str, model: str, text: str = ANSWER, pause: float = 0) -> Reply:
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {
            "type": "message_start",
            "message": {
                "id": identity,
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": messages_usage(1),
            },
        },
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": COMPLETION_TOKENS},
        },
        {"type": "message_stop"},
    )
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(sse(event, string_value(event["type"])) for event in events),
        pause_between_chunks=pause,
    )


def error_reply(status: int, message: str) -> Reply:
    body: Final[dict[str, JsonValue]] = {"error": {"message": message, "type": "scripted_error", "code": str(status)}}
    return Reply(status=status, body=json.dumps(body).encode())


def is_readiness_probe(request: Request) -> bool:
    return bool(request.body) and READINESS_TEXT in request.body.decode(errors="replace")


def is_streaming(request: Request) -> bool:
    return bool(request.body) and JSON_OBJECT.validate_json(request.body).get("stream") is True


def marker_of(request: Request) -> str | None:
    found: Final = MARKER.search(request.body.decode(errors="replace"))
    return found.group(1) if found is not None else None


def route(request: Request, identity: str, model: str, text: str = ANSWER, pause: float = 0) -> Reply:
    match request.target:
        case "/v1/chat/completions":
            return (
                chat_stream_reply(identity, model, text, pause)
                if is_streaming(request)
                else chat_reply(identity, model, text)
            )
        case "/v1/responses":
            return responses_reply(identity, model, is_streaming(request), text, pause)
        case "/v1/messages":
            return (
                messages_stream_reply(identity, model, text, pause)
                if is_streaming(request)
                else messages_reply(identity, model, text)
            )
        case _:
            return error_reply(404, f"unknown route {request.target}")


def provider(identity: str, model: str = MODEL, stream_pause: float = 0) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        if is_readiness_probe(request):
            return chat_reply(f"probe_{uuid.uuid4().hex}", model)
        return route(request, identity, model, pause=stream_pause)

    return respond


def marker_identity(request: Request, marker: str) -> str:
    match request.target:
        case "/v1/responses":
            return f"resp_{marker}"
        case "/v1/messages":
            return f"msg_{marker}"
        case _:
            return f"req_{marker}"


def marker_provider(model: str = MODEL, stream_pause: float = 0) -> Callable[[Request], Reply]:
    def respond(request: Request) -> Reply:
        marker: Final = marker_of(request)
        if marker is None:
            return chat_reply(f"probe_{uuid.uuid4().hex}", model)
        return route(request, marker_identity(request, marker), model, f"answer marker-{marker}", stream_pause)

    return respond


def failing_provider(status: int, message: str) -> Callable[[Request], Reply]:
    return lambda _request: error_reply(status, message)


def gateway_v1(gateway: Gateway) -> str:
    return f"{str(gateway.client.base_url).rstrip('/')}/v1"


def openai_client(gateway: Gateway) -> openai.OpenAI:
    return openai.OpenAI(base_url=gateway_v1(gateway), api_key=gateway.key, max_retries=0)


def async_openai_client(gateway: Gateway) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(base_url=gateway_v1(gateway), api_key=gateway.key, max_retries=0)


def anthropic_client(gateway: Gateway) -> anthropic.Anthropic:
    return anthropic.Anthropic(base_url=str(gateway.client.base_url).rstrip("/"), api_key=gateway.key, max_retries=0)


def async_anthropic_client(gateway: Gateway) -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(
        base_url=str(gateway.client.base_url).rstrip("/"), api_key=gateway.key, max_retries=0
    )


def not_yet_routable(response: object) -> bool:
    status: Final = getattr(response, "status_code", None)
    text: Final = getattr(response, "text", "")
    return status == 400 and "Invalid model name" in text


def ready(gateway: Gateway, model: str, seconds: float = 60) -> None:
    def probe() -> tuple[int, str]:
        try:
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": READINESS_TEXT}], "cache": NO_CACHE},
            )
        except httpx.TransportError as lost:
            return 0, repr(lost)
        return response.status_code, response.text

    def routable(status: int, text: str) -> bool:
        return status != 0 and not (status == 400 and "Invalid model name" in text)

    eventually(
        lambda: tuple(probe() for _ in range(2)),
        lambda probes: all(routable(status, text) for status, text in probes),
        seconds=seconds,
    )


def deployment(scenario: Scenario, wire: Wire, model: str = MODEL, **overrides: JsonValue) -> str:
    alias: Final = scenario.model(model=f"{PROVIDER}/{model}", api_base=f"{wire.url}/v1", api_key=API_KEY, **overrides)
    ready(scenario.gateway, alias)
    return alias


def raw_deployment(scenario: Scenario, model_name: str, litellm_params: Mapping[str, JsonValue]) -> str:
    created: Final = scenario.gateway.post(
        "/model/new", {"model_name": model_name, "litellm_params": dict(litellm_params), "model_info": {}}
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return model_name


def closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def provider_calls(wire: Wire) -> tuple[Request, ...]:
    return tuple(request for request in wire.drain() if not is_readiness_probe(request))


def only_call(wire: Wire, target: str, model: str = MODEL, key: str = API_KEY) -> dict[str, JsonValue]:
    received: Final = provider_calls(wire)
    assert [(request.method, request.target) for request in received] == [("POST", target)], [
        (request.method, request.target) for request in received
    ]
    assert received[0].headers.get("authorization") == f"Bearer {key}", received[0].headers
    if target == "/v1/messages":
        assert received[0].headers.get("anthropic-version"), received[0].headers
    body: Final = JSON_OBJECT.validate_json(received[0].body)
    assert body["model"] == model, body
    assert not LEAK_FIELDS & body.keys(), sorted(LEAK_FIELDS & body.keys())
    return body


def spend_row(identity: str, seconds: float = 70) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows(f"{SPEND_COLUMNS} WHERE request_id=%s", (identity,)),
        lambda found: len(found) == 1,
        seconds=seconds,
    )
    return rows[0]


def group_rows(model_group: str, count: int, seconds: float = 70) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: read_rows(f"{SPEND_COLUMNS} WHERE model_group=%s AND request_id NOT LIKE 'probe_%%'", (model_group,)),
        lambda found: len(found) == count,
        seconds=seconds,
    )


def assert_billed(row: Mapping[str, JsonValue], alias: str, call_type: str, model: str = MODEL) -> None:
    assert row["model_group"] == alias, row
    assert row["model"] == f"{PROVIDER}/{model}", row
    assert row["custom_llm_provider"] == PROVIDER, row
    assert row["status"] == "success", row
    assert row["call_type"] == call_type, row
    assert row["prompt_tokens"] == PROMPT_TOKENS and row["completion_tokens"] == COMPLETION_TOKENS, row
    assert number(row["spend"]) == pytest.approx(expected_spend(model)), (row, expected_spend(model))


def assert_failed(row: Mapping[str, JsonValue], alias: str, model: str = MODEL) -> None:
    assert row["model_group"] == alias, row
    assert row["model"] == f"{PROVIDER}/{model}", row
    assert row["custom_llm_provider"] == PROVIDER, row
    assert row["status"] == "failure", row
    assert number(row["spend"]) == 0, row
