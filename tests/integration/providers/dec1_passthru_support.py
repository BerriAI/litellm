"""Shared rig for the `test_dec1_passthru_*` contracts: provider pass-through routes under keys, budgets and limits.

The built-in provider routes (`/anthropic/*`, `/cohere/*`, `/mistral/*`, `/gemini/*`, `/openai/*`, `/azure/*`)
read their upstream base URL and operator credential from the environment (`ANTHROPIC_API_BASE`,
`COHERE_API_BASE`, ... and `<PROVIDER>_API_KEY`, `passthrough_endpoint_router.py:428`). The shared suite proxy
sets none of them, so these tests boot one proxy of their own per module with every base pointed at a single
owned peer (`wire_server`). Each provider gets its own path prefix on that peer (`/an`, `/co`, `/mi`, `/ge`,
`/oa`, `/az`), which both tells the scripted responder which provider it is playing and pins that a base URL
with a path keeps that path when the route joins the caller's endpoint onto it
(`HttpPassThroughEndpointHelpers.join_base_and_endpoint_path`).

The owned proxy boots on the shared database and Redis (`OwnedState.SHARED_WITH_WORKER`), so keys, teams and
spend rows created or written through it can be read with the usual SQL helpers. The Bedrock router-model
contracts need no environment and run on the shared `gateway`.

Nothing here asserts a contract on its own; it builds replies, boots proxies and reads rows.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Final
from urllib.parse import parse_qsl, urlsplit

import httpx
from integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from integration._support.upstream import _aws_event_frame
from integration._support.wire import Reply, Request, Wire
from integration.authorization.dec1_limits_support import limits_proxy
from pydantic import JsonValue

SPEND_SECONDS: Final = 70
PASS_THROUGH_CALL_TYPE: Final = "pass_through_endpoint"
ROUTER_PASS_THROUGH_CALL_TYPE: Final = "allm_passthrough_route"

ANTHROPIC_PREFIX: Final = "/an"
COHERE_PREFIX: Final = "/co"
MISTRAL_PREFIX: Final = "/mi"
GEMINI_PREFIX: Final = "/ge"
OPENAI_PREFIX: Final = "/oa"
AZURE_PREFIX: Final = "/az"

ANTHROPIC_OPERATOR_KEY: Final = "dec1-passthru-operator-anthropic"
COHERE_OPERATOR_KEY: Final = "dec1-passthru-operator-cohere"
MISTRAL_OPERATOR_KEY: Final = "dec1-passthru-operator-mistral"
GEMINI_OPERATOR_KEY: Final = "dec1-passthru-operator-gemini"
OPENAI_OPERATOR_KEY: Final = "dec1-passthru-operator-openai"
AZURE_OPERATOR_KEY: Final = "dec1-passthru-operator-azure"

ANTHROPIC_MODEL: Final = "claude-sonnet-4-5"
ANTHROPIC_INPUT_RATE: Final = 3e-06
ANTHROPIC_OUTPUT_RATE: Final = 1.5e-05
ANTHROPIC_INPUT_TOKENS: Final = 7
ANTHROPIC_OUTPUT_TOKENS: Final = 3
COHERE_MODEL: Final = "command-a-03-2025"
COHERE_INPUT_RATE: Final = 2.5e-06
COHERE_OUTPUT_RATE: Final = 1e-05
COHERE_TOKENS: Final = (13, 5)
"""`usage.tokens`; the reply also carries different `usage.billed_units` so a test can tell which one is read."""
COHERE_BILLED_UNITS: Final = (2, 1)
MISTRAL_MODEL: Final = "mistral-small-latest"
OPENAI_MODEL: Final = "gpt-4o-mini"
GEMINI_MODEL: Final = "gemini-2.5-flash"
OPENAI_USAGE: Final = (5, 3)
GEMINI_USAGE: Final = (11, 7)

FAIL_MARKER: Final = "dec1-passthru-fail"
"""A prompt containing this marker makes the scripted provider answer 500 with `PROVIDER_FAILURE`."""
PROVIDER_FAILURE: Final = b'{"error":{"message":"dec1 scripted provider failure","type":"server_error"}}'
HOLD_MARKER: Final = "dec1-passthru-hold"
"""A prompt containing this marker parks the reply until `HOLD.set()`; used to keep one request in flight."""
HOLD: Final = threading.Event()


def digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def marker(label: str = "dec1-passthru") -> str:
    return f"{label}-{uuid.uuid4().hex}"


def path_of(request: Request) -> str:
    return urlsplit(request.target).path


def query_of(request: Request) -> dict[str, str]:
    return dict(parse_qsl(urlsplit(request.target).query, keep_blank_values=True))


def body_of(request: Request) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(request.body) if request.body else {}


def posts(wire: Wire) -> tuple[Request, ...]:
    """Every non-GET request the peer received since the last drain (the proxy may probe with GET)."""
    return tuple(request for request in wire.drain() if request.method != "GET")


def only_post(wire: Wire) -> Request:
    received: Final = posts(wire)
    assert len(received) == 1, [(request.method, request.target) for request in received]
    return received[0]


def assert_key_absent(request: Request, key: str) -> None:
    """The caller's virtual key must not reach the provider in any header, in the URL or in the body."""
    leaked: Final = {name: value for name, value in request.headers.items() if key in value}
    assert leaked == {}, leaked
    assert key not in request.target, request.target
    assert key.encode() not in request.body, request.body


def _sse(payload: Mapping[str, JsonValue], event: str | None = None) -> bytes:
    head: Final = f"event: {event}\n" if event is not None else ""
    return f"{head}data: {json.dumps(payload)}\n\n".encode()


def anthropic_message(text: str, model: str = ANTHROPIC_MODEL) -> dict[str, JsonValue]:
    return {
        "id": f"msg_{uuid.uuid4().hex[:12]}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": ANTHROPIC_INPUT_TOKENS, "output_tokens": ANTHROPIC_OUTPUT_TOKENS},
    }


def anthropic_frames(text: str, model: str = ANTHROPIC_MODEL) -> tuple[bytes, ...]:
    """Six Anthropic SSE frames: input tokens ride `message_start`, output tokens ride `message_delta`."""
    start: Final = {**anthropic_message("", model), "content": [], "stop_reason": None}
    start["usage"] = {"input_tokens": ANTHROPIC_INPUT_TOKENS, "output_tokens": 1}
    events: Final[tuple[dict[str, JsonValue], ...]] = (
        {"type": "message_start", "message": start},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": ANTHROPIC_OUTPUT_TOKENS},
        },
        {"type": "message_stop"},
    )
    return tuple(_sse(event, str(event["type"])) for event in events)


def cohere_chat(text: str) -> dict[str, JsonValue]:
    return {
        "id": f"co-{uuid.uuid4().hex[:12]}",
        "finish_reason": "COMPLETE",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        "usage": {
            "billed_units": {"input_tokens": COHERE_BILLED_UNITS[0], "output_tokens": COHERE_BILLED_UNITS[1]},
            "tokens": {"input_tokens": COHERE_TOKENS[0], "output_tokens": COHERE_TOKENS[1]},
        },
    }


def openai_chat(text: str, model: str) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": OPENAI_USAGE[0],
            "completion_tokens": OPENAI_USAGE[1],
            "total_tokens": sum(OPENAI_USAGE),
        },
    }


def openai_frames(text: str, model: str) -> tuple[bytes, ...]:
    chunk: Final[dict[str, JsonValue]] = {"id": "chatcmpl-dec1", "object": "chat.completion.chunk", "created": 1}
    return (
        _sse({**chunk, "model": model, "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}}]}),
        _sse(
            {
                **chunk,
                "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": openai_chat(text, model)["usage"],
            }
        ),
        b"data: [DONE]\n\n",
    )


def gemini_reply(text: str) -> dict[str, JsonValue]:
    return {
        "candidates": [{"content": {"parts": [{"text": text}], "role": "model"}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {
            "promptTokenCount": GEMINI_USAGE[0],
            "candidatesTokenCount": GEMINI_USAGE[1],
            "totalTokenCount": sum(GEMINI_USAGE),
        },
        "modelVersion": GEMINI_MODEL,
    }


def _held(reply: Reply, request: Request) -> Reply:
    if HOLD_MARKER.encode() not in request.body:
        return reply
    chunks: Final = reply.chunks if reply.chunks is not None else (reply.body,)
    return Reply(
        status=reply.status,
        chunks=chunks,
        content_type=reply.content_type,
        gate_before_first=HOLD,
        headers=reply.headers,
    )


def _scripted(request: Request) -> Reply:
    path: Final = path_of(request)
    if request.method == "GET":
        return Reply(body=json.dumps({"object": "list", "data": [], "path": path}).encode())
    if FAIL_MARKER.encode() in request.body:
        return Reply(status=500, body=PROVIDER_FAILURE)
    body: Final = body_of(request)
    streaming: Final = body.get("stream") is True
    if path.startswith(f"{ANTHROPIC_PREFIX}/"):
        model = str(body.get("model", ANTHROPIC_MODEL))
        if streaming:
            return Reply(chunks=anthropic_frames("scripted anthropic", model), content_type="text/event-stream")
        return Reply(body=json.dumps(anthropic_message("scripted anthropic", model)).encode())
    if path.startswith(f"{COHERE_PREFIX}/"):
        return Reply(body=json.dumps(cohere_chat("scripted cohere")).encode())
    if path.startswith(f"{GEMINI_PREFIX}/"):
        if ":streamGenerateContent" in path:
            return Reply(chunks=(_sse(gemini_reply("scripted gemini")),), content_type="text/event-stream")
        return Reply(body=json.dumps(gemini_reply("scripted gemini")).encode())
    model: Final = str(body.get("model", OPENAI_MODEL))
    if streaming:
        return Reply(chunks=openai_frames("scripted openai", model), content_type="text/event-stream")
    return Reply(body=json.dumps(openai_chat("scripted openai", model)).encode())


def scripted_provider(request: Request) -> Reply:
    """One responder that plays every provider, chosen by the path prefix its base URL carries."""
    return _held(_scripted(request), request)


def provider_environment(wire: Wire) -> dict[str, str]:
    return {
        "ANTHROPIC_API_BASE": f"{wire.url}{ANTHROPIC_PREFIX}",
        "ANTHROPIC_API_KEY": ANTHROPIC_OPERATOR_KEY,
        "COHERE_API_BASE": f"{wire.url}{COHERE_PREFIX}",
        "COHERE_API_KEY": COHERE_OPERATOR_KEY,
        "MISTRAL_API_BASE": f"{wire.url}{MISTRAL_PREFIX}",
        "MISTRAL_API_KEY": MISTRAL_OPERATOR_KEY,
        "GEMINI_API_BASE": f"{wire.url}{GEMINI_PREFIX}",
        "GEMINI_API_KEY": GEMINI_OPERATOR_KEY,
        "OPENAI_API_BASE": f"{wire.url}{OPENAI_PREFIX}",
        "OPENAI_API_KEY": OPENAI_OPERATOR_KEY,
        "AZURE_API_BASE": f"{wire.url}{AZURE_PREFIX}",
        "AZURE_API_KEY": AZURE_OPERATOR_KEY,
    }


@contextmanager
def provider_proxy(
    gateway: Gateway,
    directory: Path,
    wire: Wire,
    *,
    environment: Mapping[str, str] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
) -> Iterator[Gateway]:
    """An owned proxy on the shared database and Redis whose provider routes all reach `wire`."""
    with limits_proxy(
        gateway,
        directory,
        environment={**provider_environment(wire), **(environment or {})},
        general_settings=general_settings,
        litellm_settings=litellm_settings,
        shared=True,
    ) as candidate:
        yield candidate


def anthropic_body(text: str, *, model: str = ANTHROPIC_MODEL, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": text}], **extra}


def cohere_body(text: str, *, model: str = COHERE_MODEL, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": text}], **extra}


def openai_body(text: str, *, model: str = OPENAI_MODEL, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": text}], **extra}


def gemini_body(text: str) -> dict[str, JsonValue]:
    return {"contents": [{"role": "user", "parts": [{"text": text}]}]}


def gemini_path(model: str = GEMINI_MODEL, action: str = "generateContent") -> str:
    return f"/gemini/v1beta/models/{model}:{action}"


def send(
    proxy: Gateway,
    path: str,
    body: Mapping[str, JsonValue] | None,
    *,
    headers: Mapping[str, str],
    method: str = "POST",
    params: Mapping[str, str] | None = None,
) -> httpx.Response:
    """A raw call with exactly the headers given, so a test controls which header carries the key."""
    return proxy.client.request(method, path, json=body, headers=dict(headers), params=params)


def bearer(key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {key}"}


def anthropic_error(response: httpx.Response, status: int) -> dict[str, JsonValue]:
    """The Anthropic error envelope `/anthropic/*` routes answer with (`anthropic_error_payload.py:50`)."""
    assert response.status_code == status, f"{response.status_code} {response.text}"
    parsed: Final = JSON_OBJECT.validate_json(response.content)
    assert parsed["type"] == "error", response.text
    return object_value(parsed["error"])


def openai_error(response: httpx.Response, status: int) -> dict[str, JsonValue]:
    """The OpenAI error envelope every other proxy route answers with."""
    assert response.status_code == status, f"{response.status_code} {response.text}"
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


_SPEND_COLUMNS: Final = (
    "request_id, call_type, model, model_group, custom_llm_provider, spend, prompt_tokens, completion_tokens, "
    "total_tokens, status, api_base, team_id, end_user, request_tags, metadata"
)


def spend_rows(gateway: Gateway, key: str) -> list[dict[str, JsonValue]]:
    return gateway.rows(f'SELECT {_SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE api_key=%s', (digest(key),))


def spend_row(gateway: Gateway, key: str) -> dict[str, JsonValue]:
    """The single spend row this key wrote; rows land asynchronously, so this polls."""
    return eventually(lambda: spend_rows(gateway, key), lambda found: len(found) == 1, seconds=SPEND_SECONDS)[0]


def spend_row_by_id(gateway: Gateway, request_id: str) -> dict[str, JsonValue]:
    return eventually(
        lambda: gateway.rows(f'SELECT {_SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE request_id=%s', (request_id,)),
        lambda found: len(found) == 1,
        seconds=SPEND_SECONDS,
    )[0]


def number(value: JsonValue) -> float:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), value
    return float(value)


def key_spend(gateway: Gateway, key: str) -> float:
    return number(object_value(gateway.get("/key/info", {"key": key})["info"])["spend"])


def refused_until(call: Callable[[], httpx.Response], status: int, seconds: float = SPEND_SECONDS) -> httpx.Response:
    """Repeat `call` until it answers `status`; spend lands asynchronously, so a budget closes a little later."""
    return eventually(call, lambda response: response.status_code == status, seconds=seconds)


def message_of(error: Mapping[str, JsonValue]) -> str:
    return string_value(error["message"])


BEDROCK_MODEL_ID: Final = "anthropic.claude-sonnet-4-5-20250929-v1:0"
BEDROCK_INPUT_TOKENS: Final = 11
BEDROCK_OUTPUT_TOKENS: Final = 4
BEDROCK_CONVERSE_BODY: Final[dict[str, JsonValue]] = {
    "messages": [{"role": "user", "content": [{"text": "dec1 passthru bedrock"}]}]
}
BEDROCK_CONVERSE_REPLY: Final = json.dumps(
    {
        "output": {"message": {"role": "assistant", "content": [{"text": "scripted bedrock"}]}},
        "stopReason": "end_turn",
        "usage": {
            "inputTokens": BEDROCK_INPUT_TOKENS,
            "outputTokens": BEDROCK_OUTPUT_TOKENS,
            "totalTokens": BEDROCK_INPUT_TOKENS + BEDROCK_OUTPUT_TOKENS,
        },
        "metrics": {"latencyMs": 1},
    }
).encode()
BEDROCK_STREAM_BYTES: Final = b"".join(
    _aws_event_frame(kind, payload, "sc", "u")
    for kind, payload in (
        ("messageStart", {"role": "assistant"}),
        ("contentBlockDelta", {"delta": {"text": "scripted bedrock"}, "contentBlockIndex": 0}),
        ("messageStop", {"stopReason": "end_turn"}),
        (
            "metadata",
            {
                "usage": {
                    "inputTokens": BEDROCK_INPUT_TOKENS,
                    "outputTokens": BEDROCK_OUTPUT_TOKENS,
                    "totalTokens": BEDROCK_INPUT_TOKENS + BEDROCK_OUTPUT_TOKENS,
                },
                "metrics": {"latencyMs": 1},
            },
        ),
    )
)


def bedrock_peer(request: Request) -> Reply:
    """Bedrock runtime: event-stream bytes for the streaming actions, a Converse JSON body otherwise."""
    if FAIL_MARKER.encode() in request.body:
        return Reply(status=500, body=PROVIDER_FAILURE)
    if request.target.split("?")[0].endswith(("-stream", "invoke-with-response-stream")):
        return Reply(body=BEDROCK_STREAM_BYTES, content_type="application/vnd.amazon.eventstream")
    return Reply(body=BEDROCK_CONVERSE_REPLY)


def bedrock_deployment(wire: Wire, **parameters: JsonValue) -> dict[str, JsonValue]:
    """`litellm_params` for a SigV4 Bedrock deployment whose runtime endpoint is the owned peer."""
    return {
        "model": f"bedrock/{BEDROCK_MODEL_ID}",
        "api_base": wire.url,
        "aws_access_key_id": "AKIADEC1PASSTHRU",
        "aws_secret_access_key": "dec1-passthru-secret",
        "aws_region_name": "us-east-1",
        **parameters,
    }


@dataclass(frozen=True, slots=True)
class Held:
    """A request parked at the provider by `HOLD_MARKER`, released by `release()`."""

    thread: threading.Thread
    results: list[httpx.Response]

    def release(self) -> httpx.Response:
        HOLD.set()
        self.thread.join(timeout=30)
        assert not self.thread.is_alive(), "held request never finished"
        HOLD.clear()
        assert len(self.results) == 1, self.results
        return self.results[0]


def hold_in_background(call: Callable[[], httpx.Response]) -> Held:
    HOLD.clear()
    results: Final[list[httpx.Response]] = []
    thread: Final = threading.Thread(target=lambda: results.append(call()), daemon=True)
    thread.start()
    return Held(thread, results)


def wait_for_posts(wire: Wire, count: int, seconds: float = 30) -> tuple[Request, ...]:
    """Collect drained requests until `count` POSTs have arrived (a held request is recorded before its reply)."""
    seen: Final[list[Request]] = []

    def collect() -> int:
        seen.extend(request for request in wire.drain() if request.method != "GET")
        return len(seen)

    eventually(collect, lambda total: total >= count, seconds=seconds)
    return tuple(seen)
