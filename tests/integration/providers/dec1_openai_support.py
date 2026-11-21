"""Shared rig for the `test_dec1_openai_*` files: OpenAI and the large OpenAI-compatible providers through the proxy.

Every test talks to the shared proxy (`gateway` fixture) and to one owned HTTP peer that plays the provider. The
peer records each request the proxy sends (method, path, query, headers, decoded JSON body or raw multipart
bytes) and answers with a canned reply chosen from the request path, unless the test passes its own `respond`:

    .../chat/completions   a chat completion whose content is `REPLY_TEXT` with usage 11 prompt + 7 completion
                           tokens, or the same as SSE chunks (three content pieces, a finish chunk, a usage chunk
                           with `choices: []` and `[DONE]`) when the body has `stream: true`
    .../responses          a completed Responses API object whose one message says `REPLY_TEXT`, usage 11 input
                           + 7 output tokens, or the canonical event sequence when `stream: true`
    .../embeddings         an OpenAI list with one `[0.25, 0.5, 0.75]` vector per input, 5 prompt tokens
    .../moderations        one unflagged result
    .../audio/transcriptions  `{"text": TRANSCRIPT}`
    .../v1/messages        an Anthropic message (used by DeepSeek's Anthropic-compatible endpoint)

Deployments are priced through `litellm_params` at `INPUT_RATE` 0.001 per input token and `OUTPUT_RATE` 0.002
per output token, so every expected spend is a literal: one canned chat or responses call costs
11 * 0.001 + 7 * 0.002 = 0.025.

The shared proxy has a Redis response cache on, so every prompt carries a fresh `marker()`. Every deployment
sets `num_retries: 0` and `max_retries: 0` so a scripted failure is sent upstream exactly once.
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
    list_value,
    object_value,
    string_value,
)
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

PROVIDER_KEY: Final = "dec1-openai-provider-key"
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
PROMPT_TOKENS: Final = 11
COMPLETION_TOKENS: Final = 7
CALL_COST: Final = 0.025
EMBEDDING_TOKENS: Final = 5
VECTOR: Final = (0.25, 0.5, 0.75)
REPLY_TEXT: Final = "dec1 openai reply"
STREAM_PIECES: Final = ("dec1 ", "openai ", "reply")
TRANSCRIPT: Final = "dec1 synthetic transcript"
LANDING_SECONDS: Final = 70
NO_RETRIES: Final[Mapping[str, JsonValue]] = {"num_retries": 0, "max_retries": 0}
PRICES: Final[Mapping[str, JsonValue]] = {"input_cost_per_token": INPUT_RATE, "output_cost_per_token": OUTPUT_RATE}
SPEND_COLUMNS: Final = (
    "request_id, call_type, spend, total_tokens, prompt_tokens, completion_tokens, model, model_group, "
    "custom_llm_provider, api_base, status"
)


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
        """Every inference request received so far; model discovery GETs are left out."""
        with self.lock:
            return tuple(call for call in self.seen if call.method == "POST")

    def only(self) -> Call:
        calls: Final = self.posts()
        assert len(calls) == 1, [(call.method, call.path) for call in calls]
        return calls[0]

    def all(self) -> tuple[Call, ...]:
        with self.lock:
            return tuple(self.seen)


def marker() -> str:
    """A unique string, so the shared gateway's response cache never answers a request by accident."""
    return f"dec1-openai-{uuid.uuid4().hex}"


def json_reply(body: Mapping[str, JsonValue], status: int = 200, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode(), headers=headers or {})


def openai_error(
    status: int,
    message: str,
    *,
    kind: str | None = None,
    param: str | None = None,
    code: str | None = None,
    headers: Mapping[str, str] | None = None,
) -> Reply:
    """An OpenAI-shaped provider error body `{"error": {"message", "type", "param", "code"}}`."""
    default_kind: Final = (
        "rate_limit_error" if status == 429 else "invalid_request_error" if status < 500 else "server_error"
    )
    return json_reply(
        {"error": {"message": message, "type": kind or default_kind, "param": param, "code": code}}, status, headers
    )


def sse(payload: Mapping[str, JsonValue], event: str | None = None) -> bytes:
    prefix: Final = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def usage(prompt: int = PROMPT_TOKENS, completion: int = COMPLETION_TOKENS) -> dict[str, JsonValue]:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}


def chat_json(
    body: Mapping[str, JsonValue],
    *,
    content: str | None = REPLY_TEXT,
    finish_reason: str = "stop",
    extra_message: Mapping[str, JsonValue] | None = None,
    extra_choice: Mapping[str, JsonValue] | None = None,
    extra_body: Mapping[str, JsonValue] | None = None,
    choices: int = 1,
) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-dec1-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": body.get("model"),
        "choices": [
            {
                "index": index,
                "message": {"role": "assistant", "content": content, **(extra_message or {})},
                "finish_reason": finish_reason,
                **(extra_choice or {}),
            }
            for index in range(choices)
        ],
        "usage": usage(),
        **(extra_body or {}),
    }


def chat_stream(body: Mapping[str, JsonValue], *, with_usage: bool = True) -> Reply:
    identity: Final = f"chatcmpl-dec1-{uuid.uuid4().hex}"
    base: Final[dict[str, JsonValue]] = {
        "id": identity,
        "object": "chat.completion.chunk",
        "created": 1,
        "model": body.get("model"),
    }
    pieces: Final = tuple(
        sse({**base, "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
        for piece in STREAM_PIECES
    )
    tail: Final = (sse({**base, "choices": [], "usage": usage()}),) if with_usage else ()
    return Reply(
        content_type="text/event-stream",
        chunks=(
            sse(
                {
                    **base,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}],
                }
            ),
            *pieces,
            sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            *tail,
            b"data: [DONE]\n\n",
        ),
    )


def chat_reply(body: Mapping[str, JsonValue]) -> Reply:
    return chat_stream(body) if body.get("stream") is True else json_reply(chat_json(body))


def responses_object(
    body: Mapping[str, JsonValue], *, identity: str | None = None, status: str = "completed", **extra: JsonValue
) -> dict[str, JsonValue]:
    response_id: Final = identity or f"resp_dec1{uuid.uuid4().hex}"
    return {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": body.get("model"),
        "output": []
        if status in ("queued", "in_progress")
        else [
            {
                "id": f"msg_{response_id}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": REPLY_TEXT, "annotations": []}],
            }
        ],
        "usage": None
        if status in ("queued", "in_progress")
        else {
            "input_tokens": PROMPT_TOKENS,
            "output_tokens": COMPLETION_TOKENS,
            "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
        **extra,
    }


RESPONSES_EVENT_ORDER: Final = (
    "response.created",
    "response.in_progress",
    "response.output_item.added",
    "response.content_part.added",
    "response.output_text.delta",
    "response.output_text.delta",
    "response.output_text.delta",
    "response.output_text.done",
    "response.content_part.done",
    "response.output_item.done",
    "response.completed",
)


def responses_stream(body: Mapping[str, JsonValue], identity: str | None = None) -> Reply:
    """The OpenAI Responses SSE sequence in `RESPONSES_EVENT_ORDER`, with `sequence_number` 0..10."""
    response_id: Final = identity or f"resp_dec1{uuid.uuid4().hex}"
    item_id: Final = f"msg_{response_id}"
    pending: Final = responses_object(body, identity=response_id, status="in_progress")
    done: Final = responses_object(body, identity=response_id)
    item: Final[dict[str, JsonValue]] = {
        "id": item_id,
        "type": "message",
        "status": "in_progress",
        "role": "assistant",
        "content": [],
    }
    part: Final[dict[str, JsonValue]] = {"type": "output_text", "text": "", "annotations": []}
    events: Final[list[dict[str, JsonValue]]] = [
        {"type": "response.created", "response": pending},
        {"type": "response.in_progress", "response": pending},
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {
            "type": "response.content_part.added",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": part,
        },
        *(
            {
                "type": "response.output_text.delta",
                "item_id": item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": piece,
            }
            for piece in STREAM_PIECES
        ),
        {
            "type": "response.output_text.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "text": REPLY_TEXT,
        },
        {
            "type": "response.content_part.done",
            "item_id": item_id,
            "output_index": 0,
            "content_index": 0,
            "part": {**part, "text": REPLY_TEXT},
        },
        {"type": "response.output_item.done", "output_index": 0, "item": list_value(done["output"])[0]},
        {"type": "response.completed", "response": done},
    ]
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(
            sse({**event, "sequence_number": index}, event=string_value(event["type"]))
            for index, event in enumerate(events)
        ),
    )


def responses_reply(body: Mapping[str, JsonValue]) -> Reply:
    return responses_stream(body) if body.get("stream") is True else json_reply(responses_object(body))


def anthropic_message(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {
        "id": f"msg_dec1{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model"),
        "content": [{"type": "text", "text": REPLY_TEXT}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS},
    }


def default_respond(call: Call) -> Reply:
    """The canned answer for a provider path, as listed in the module docstring."""
    if call.method != "POST":
        return json_reply({"object": "list", "data": []})
    if call.path.endswith("/chat/completions"):
        return chat_reply(call.body)
    if call.path.endswith("/responses"):
        return responses_reply(call.body)
    if call.path.endswith("/embeddings"):
        count: Final = (
            len(value)
            if isinstance(value := call.body.get("input"), list) and value and isinstance(value[0], str)
            else 1
        )
        return json_reply(
            {
                "object": "list",
                "data": [{"object": "embedding", "index": index, "embedding": list(VECTOR)} for index in range(count)],
                "model": call.body.get("model"),
                "usage": {"prompt_tokens": EMBEDDING_TOKENS, "total_tokens": EMBEDDING_TOKENS},
            }
        )
    if call.path.endswith("/moderations"):
        return json_reply(
            {
                "id": f"modr-dec1-{uuid.uuid4().hex}",
                "model": call.body.get("model") or "omni-moderation-latest",
                "results": [
                    {"flagged": False, "categories": {"violence": False}, "category_scores": {"violence": 0.01}}
                ],
            }
        )
    if call.path.endswith("/audio/transcriptions"):
        return json_reply({"text": TRANSCRIPT})
    if call.path.endswith("/v1/messages"):
        return json_reply(anthropic_message(call.body))
    return openai_error(404, f"dec1 openai peer has no route {call.path}")


def _decoded(request: Request) -> Mapping[str, JsonValue]:
    if not request.body or "json" not in request.headers.get("content-type", "json"):
        return {}
    try:
        return JSON_OBJECT.validate_json(request.body)
    except ValueError:
        return {}


@contextmanager
def peer(respond: Respond = default_respond) -> Iterator[Peer]:
    """An owned provider peer that records every request and answers with `respond`."""
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
    api_base_suffix: str = "/v1",
    priced: bool = True,
    model_info: Mapping[str, JsonValue] | None = None,
    **parameters: JsonValue,
) -> str:
    """A deployment of `model` (with its provider prefix) whose `api_base` is the peer plus `api_base_suffix`."""
    return scenario.model(
        model_info=model_info,
        **{
            "model": model,
            "api_base": f"{provider.url}{api_base_suffix}",
            "api_key": PROVIDER_KEY,
            **NO_RETRIES,
            **(PRICES if priced else {}),
            **parameters,
        },
    )


def chat(
    gateway: Gateway,
    model: str,
    *,
    text: str | None = None,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    messages: list[JsonValue] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "messages": messages if messages is not None else [{"role": "user", "content": text or marker()}],
        **extra,
    }
    return gateway.request("POST", "/v1/chat/completions", body, key=key, headers=headers)


def responses(
    gateway: Gateway,
    model: str,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {"model": model, "input": extra.pop("input", None) or marker(), **extra}
    return gateway.request("POST", "/v1/responses", body, key=key, headers=headers)


def messages(
    gateway: Gateway,
    model: str,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "max_tokens": 64,
        "messages": [{"role": "user", "content": marker()}],
        **extra,
    }
    return gateway.request(
        "POST", "/v1/messages", body, key=key, headers={"anthropic-version": "2023-06-01", **(headers or {})}
    )


def stream_lines(
    gateway: Gateway, path: str, body: Mapping[str, JsonValue], *, key: str | None = None
) -> tuple[httpx.Headers, tuple[str, ...]]:
    """Send a streaming request and return the response headers and every non-empty line."""
    with gateway.client.stream(
        "POST",
        path,
        json=dict(body),
        headers={"Authorization": f"Bearer {key or gateway.key}", "anthropic-version": "2023-06-01"},
    ) as response:
        assert response.status_code == 200, response.read().decode()
        return response.headers, tuple(line for line in response.iter_lines() if line)


def data_events(lines: Sequence[str]) -> tuple[dict[str, JsonValue], ...]:
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data:").strip())
        for line in lines
        if line.startswith("data:") and not line.endswith("[DONE]")
    )


def served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def error_message(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def first_message(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    choices: Final = body["choices"]
    assert isinstance(choices, list) and choices, body
    return object_value(object_value(choices[0])["message"])


def spend_row(gateway: Gateway, request_id: str, *, seconds: float = LANDING_SECONDS) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: gateway.rows(f'SELECT {SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (request_id,)),
        lambda found: len(found) == 1,
        seconds=seconds,
    )
    return rows[0]


def spend_row_by_group(gateway: Gateway, model_group: str, *, seconds: float = LANDING_SECONDS) -> dict[str, JsonValue]:
    """The one SpendLogs row of a model group created for one test (the deployment name is unique per test)."""
    rows: Final = eventually(
        lambda: gateway.rows(f'SELECT {SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model_group,)),
        lambda found: len(found) == 1,
        seconds=seconds,
    )
    return rows[0]
