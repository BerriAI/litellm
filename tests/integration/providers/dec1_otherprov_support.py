"""Shared rig for the `test_dec1_otherprov_*` files: the widely used providers beyond the big six, through the proxy.

Every test talks to the shared proxy (`gateway` fixture) and to one owned HTTP peer on loopback that plays the
provider. Every deployment points its `api_base` at that peer, so no request can reach a real provider host. The
peer records each request (method, path, query, headers, decoded JSON body) and answers with whatever the test's
`respond` callable returns; the default answers an OpenAI-shaped chat completion with `REPLY_TEXT` and 11 prompt
plus 7 completion tokens.

The shared proxy runs a Redis response cache, so every prompt carries a fresh `marker()`. Deployments set
`num_retries: 0` and `max_retries: 0` so one scripted failure is sent upstream exactly once.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final
from urllib.parse import parse_qs

import httpx
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    object_value,
    string_value,
)
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

PROVIDER_KEY: Final = "dec1-otherprov-provider-key"
PROMPT_TOKENS: Final = 11
COMPLETION_TOKENS: Final = 7
REPLY_TEXT: Final = "dec1 otherprov reply"
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
CALL_COST: Final = 0.025
LANDING_SECONDS: Final = 70
NO_RETRIES: Final[Mapping[str, JsonValue]] = {"num_retries": 0, "max_retries": 0}
PRICES: Final[Mapping[str, JsonValue]] = {"input_cost_per_token": INPUT_RATE, "output_cost_per_token": OUTPUT_RATE}
SPEND_COLUMNS: Final = (
    "request_id, call_type, spend, total_tokens, prompt_tokens, completion_tokens, model, model_group, "
    "custom_llm_provider, api_base, status"
)
TOOL: Final[dict[str, JsonValue]] = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Weather for a city",
        "strict": True,
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True, slots=True)
class Call:
    """One request the proxy sent to the provider peer."""

    method: str
    path: str
    query: Mapping[str, Sequence[str]]
    headers: Mapping[str, str]
    body: Mapping[str, JsonValue]
    raw: bytes


Respond = Callable[[Call], Reply]


@dataclass(frozen=True, slots=True)
class Peer:
    url: str
    seen: list[Call] = field(default_factory=list)  # mutable-ok: appended by the peer's handler threads
    lock: threading.Lock = field(default_factory=threading.Lock)

    def posts(self) -> tuple[Call, ...]:
        """Every inference request received so far; discovery GETs are left out."""
        with self.lock:
            return tuple(call for call in self.seen if call.method == "POST")

    def only(self) -> Call:
        calls: Final = self.posts()
        assert len(calls) == 1, [(call.method, call.path) for call in calls]
        return calls[0]


def marker() -> str:
    """A unique string, so the shared gateway's response cache never answers a request by accident."""
    return f"dec1-otherprov-{uuid.uuid4().hex}"


def json_reply(body: Mapping[str, JsonValue], status: int = 200, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode(), headers=headers or {})


def text_reply(text: str, status: int) -> Reply:
    return Reply(status=status, body=text.encode(), content_type="text/plain")


def sse(payload: Mapping[str, JsonValue], event: str | None = None) -> bytes:
    prefix: Final = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def ndjson(*payloads: Mapping[str, JsonValue]) -> tuple[bytes, ...]:
    """One JSON document per line, the framing Ollama and the Cohere iterators read."""
    return tuple(f"{json.dumps(payload, separators=(',', ':'))}\n".encode() for payload in payloads)


def usage(prompt: int = PROMPT_TOKENS, completion: int = COMPLETION_TOKENS) -> dict[str, JsonValue]:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def chat_json(
    body: Mapping[str, JsonValue],
    *,
    content: JsonValue = REPLY_TEXT,
    finish_reason: str = "stop",
    extra_message: Mapping[str, JsonValue] | None = None,
    extra_body: Mapping[str, JsonValue] | None = None,
    usage_block: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-dec1-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": body.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, **(extra_message or {})},
                "finish_reason": finish_reason,
            }
        ],
        "usage": dict(usage_block) if usage_block is not None else usage(),
        **(extra_body or {}),
    }


def chat_chunk(body: Mapping[str, JsonValue], identity: str, **choice_or_usage: JsonValue) -> dict[str, JsonValue]:
    """One OpenAI chat chunk; pass `delta=...` (and `finish_reason=...`) for a choice, or `usage=...` alone."""
    base: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": body.get("model"),
    }
    if "usage" in choice_or_usage and "delta" not in choice_or_usage:
        return {**base, "choices": [], "usage": choice_or_usage["usage"]}
    return {
        **base,
        "choices": [
            {
                "index": 0,
                "delta": choice_or_usage.get("delta", {}),
                "finish_reason": choice_or_usage.get("finish_reason"),
            }
        ],
    }


def chat_stream_reply(body: Mapping[str, JsonValue], deltas: Sequence[Mapping[str, JsonValue]]) -> Reply:
    """An OpenAI SSE stream: each delta in order, a stop chunk, a usage chunk with `choices: []`, and `[DONE]`."""
    identity: Final = f"chatcmpl-dec1-{uuid.uuid4().hex}"
    return Reply(
        content_type="text/event-stream",
        chunks=(
            *(sse(chat_chunk(body, identity, delta=dict(delta))) for delta in deltas),
            sse(chat_chunk(body, identity, delta={}, finish_reason="stop")),
            sse(chat_chunk(body, identity, usage=usage())),
            b"data: [DONE]\n\n",
        ),
    )


def default_respond(call: Call) -> Reply:
    if call.method != "POST":
        return json_reply({"object": "list", "data": []})
    if call.body.get("stream") is True:
        return chat_stream_reply(call.body, ({"role": "assistant", "content": REPLY_TEXT},))
    return json_reply(chat_json(call.body))


def _decoded(request: Request) -> Mapping[str, JsonValue]:
    """The JSON object body, whatever the content-type says. The Ollama chat call carries no JSON content-type (its
    `validate_environment` sets only `Authorization`, `ollama/chat/transformation.py:234-246`), so gating on the
    header left every Ollama body empty in run 44176682961; a multipart or binary body still decodes to `{}`."""
    if not request.body:
        return {}
    try:
        return JSON_OBJECT.validate_json(request.body)
    except ValueError:
        return {}


@contextmanager
def peer(respond: Respond = default_respond) -> Iterator[Peer]:
    """An owned loopback provider peer that records every request and answers with `respond`."""
    holder: Final[list[Peer]] = []  # mutable-ok: the handler needs the Peer built from its own wire

    def handle(request: Request) -> Reply:
        path, _, query = request.target.partition("?")
        call: Final = Call(request.method, path, parse_qs(query), request.headers, _decoded(request), request.body)
        with holder[0].lock:
            holder[0].seen.append(call)
        return respond(call)

    with wire_server(handle) as wire:
        holder.append(Peer(wire.url))
        yield holder[0]


def deployment(
    scenario: Scenario,
    provider: Peer,
    model: str,
    *,
    api_base_suffix: str = "",
    priced: bool = True,
    with_key: bool = True,
    **parameters: JsonValue,
) -> str:
    """A deployment of `model` (with its provider prefix) whose `api_base` is the peer plus `api_base_suffix`."""
    return scenario.model(
        **{
            "model": model,
            "api_base": f"{provider.url}{api_base_suffix}",
            "api_key": PROVIDER_KEY if with_key else None,
            **NO_RETRIES,
            **(PRICES if priced else {}),
            **parameters,
        }
    )


def chat(
    gateway: Gateway,
    model: str,
    *,
    text: str | None = None,
    messages: list[JsonValue] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": messages if messages is not None else [{"role": "user", "content": text or marker()}],
        **extra,
    }
    return gateway.request("POST", "/v1/chat/completions", body)


def stream_lines(gateway: Gateway, body: Mapping[str, JsonValue]) -> tuple[str, ...]:
    """Send a streaming chat request and return every non-empty line of the answer."""
    with gateway.client.stream(
        "POST", "/v1/chat/completions", json=dict(body), headers={"Authorization": f"Bearer {gateway.key}"}
    ) as response:
        assert response.status_code == 200, response.read().decode()
        return tuple(line for line in response.iter_lines() if line)


def stream_chunks(gateway: Gateway, model: str, **extra: JsonValue) -> tuple[dict[str, JsonValue], ...]:
    """The decoded `data:` chunks of a streamed chat call (without `[DONE]`)."""
    lines: Final = stream_lines(
        gateway,
        {"model": model, "messages": [{"role": "user", "content": marker()}], "stream": True, **extra},
    )
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data:").strip())
        for line in lines
        if line.startswith("data:") and not line.endswith("[DONE]")
    )


def joined_delta(chunks: Sequence[Mapping[str, JsonValue]], name: str) -> str:
    """Concatenate one delta field (`content`, `reasoning_content`) over every chunk's first choice."""
    pieces: Final[list[str]] = []
    for chunk in chunks:
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            value = object_value(object_value(choices[0]).get("delta") or {}).get(name)
            if isinstance(value, str):
                pieces.append(value)
    return "".join(pieces)


def final_usage(chunks: Sequence[Mapping[str, JsonValue]]) -> dict[str, JsonValue]:
    """The last `usage` object any chunk carried."""
    found: Final = [object_value(chunk["usage"]) for chunk in chunks if isinstance(chunk.get("usage"), dict)]
    assert found, chunks
    return found[-1]


def finish_reasons(chunks: Sequence[Mapping[str, JsonValue]]) -> tuple[str, ...]:
    reasons: Final[list[str]] = []
    for chunk in chunks:
        choices = chunk.get("choices")
        if isinstance(choices, list) and choices:
            reason = object_value(choices[0]).get("finish_reason")
            if isinstance(reason, str):
                reasons.append(reason)
    return tuple(reasons)


def served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def error_message(response: httpx.Response) -> str:
    return string_value(object_value(JSON_OBJECT.validate_json(response.content)["error"])["message"])


def first_choice(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    choices: Final = body["choices"]
    assert isinstance(choices, list) and choices, body
    return object_value(choices[0])


def first_message(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(first_choice(body)["message"])


def spend_row_by_group(gateway: Gateway, model_group: str, *, seconds: float = LANDING_SECONDS) -> dict[str, JsonValue]:
    """The one SpendLogs row of a model group created for one test (the deployment name is unique per test)."""
    rows: Final = eventually(
        lambda: gateway.rows(f'SELECT {SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model_group,)),
        lambda found: len(found) == 1,
        seconds=seconds,
    )
    return rows[0]
