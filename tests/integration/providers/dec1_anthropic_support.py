"""Shared rig for the `test_dec1_anthropic_*` files: Anthropic as a provider end to end with a scripted upstream.

Every test talks to the shared proxy (`gateway` fixture) and to one owned HTTP peer that plays api.anthropic.com.
The peer records each request the proxy sends (method, path, query, lowercased headers, decoded JSON body) and
answers with a canned reply chosen from the request, unless the test passes its own `respond`:

    POST .../v1/messages               an Anthropic message whose one text block is `REPLY_TEXT`, usage
                                       11 input + 7 output tokens, or the six event text stream when
                                       the body has `stream: true`
    POST .../v1/messages/count_tokens  `{"input_tokens": 42}`
    GET  .../image.png                 a 1x1 PNG, so an `http://` image URL the proxy downloads resolves locally
    anything else                      404 in Anthropic's error envelope

Deployments point `api_base` at the peer root (the Anthropic config appends `/v1/messages`), carry
`PROVIDER_KEY` as the deployment key, and set `num_retries: 0` and `max_retries: 0` so a scripted failure reaches
the peer exactly once. Priced deployments carry `PRICES` through `litellm_params`, so every expected spend is a
literal: one canned call costs 11 * 0.001 + 7 * 0.002 = 0.025.

The shared proxy has a Redis response cache on, so every prompt carries a fresh `marker()`. Only POST requests
count as provider calls; the proxy may probe a new deployment with a GET.
"""

from __future__ import annotations

import base64
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

BACKEND: Final = "claude-sonnet-4-5-20250929"
"""A dated Claude id present in the bundled cost map with `max_output_tokens: 64000` and no adaptive thinking."""
MODEL: Final = f"anthropic/{BACKEND}"
ADAPTIVE_BACKEND: Final = "claude-opus-4-6"
"""Cost map row with `supports_adaptive_thinking: true` and `supports_legacy_thinking: true`, `max_output_tokens: 128000`."""
PROVIDER_KEY: Final = "dec1-anthropic-provider-key"
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
CACHE_READ_RATE: Final = 0.0001
CACHE_CREATION_RATE: Final = 0.003
PROMPT_TOKENS: Final = 11
COMPLETION_TOKENS: Final = 7
CALL_COST: Final = 0.025
COUNTED_TOKENS: Final = 42
REPLY_TEXT: Final = "dec1 anthropic reply"
STREAM_PIECES: Final = ("dec1 ", "anthropic ", "reply")
SIGNATURE: Final = "EqQBCkgIARABGAIiQL2dec1syntheticsignature"
LANDING_SECONDS: Final = 70
NO_RETRIES: Final[Mapping[str, JsonValue]] = {"num_retries": 0, "max_retries": 0}
PRICES: Final[Mapping[str, JsonValue]] = {
    "input_cost_per_token": INPUT_RATE,
    "output_cost_per_token": OUTPUT_RATE,
    "cache_read_input_token_cost": CACHE_READ_RATE,
    "cache_creation_input_token_cost": CACHE_CREATION_RATE,
}
SPEND_COLUMNS: Final = (
    "request_id, call_type, spend, total_tokens, prompt_tokens, completion_tokens, model, model_group, "
    "custom_llm_provider, api_base, status"
)
PNG_BYTES: Final = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_BASE64: Final = base64.b64encode(PNG_BYTES).decode()
PDF_BASE64: Final = base64.b64encode(b"%PDF-1.4\n% dec1 synthetic pdf\n%%EOF\n").decode()


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
        """Every inference request received so far; discovery and image GETs are left out."""
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
    return f"dec1-anthropic-{uuid.uuid4().hex}"


def json_reply(body: Mapping[str, JsonValue], status: int = 200, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode(), headers=headers or {})


def anthropic_error(status: int, kind: str, message: str, headers: Mapping[str, str] | None = None) -> Reply:
    """An error in Anthropic's own envelope `{"type": "error", "error": {"type", "message"}}`."""
    return json_reply({"type": "error", "error": {"type": kind, "message": message}}, status, headers)


def usage_block(
    input_tokens: int = PROMPT_TOKENS, output_tokens: int = COMPLETION_TOKENS, **extra: JsonValue
) -> dict[str, JsonValue]:
    return {"input_tokens": input_tokens, "output_tokens": output_tokens, **extra}


def message_json(
    body: Mapping[str, JsonValue] | None = None,
    *,
    content: Sequence[JsonValue] | None = None,
    stop_reason: str = "end_turn",
    stop_sequence: str | None = None,
    usage: Mapping[str, JsonValue] | None = None,
    **extra: JsonValue,
) -> dict[str, JsonValue]:
    """A complete Anthropic Messages API response body."""
    return {
        "id": f"msg_dec1{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": (body or {}).get("model") or BACKEND,
        "content": list(content) if content is not None else [{"type": "text", "text": REPLY_TEXT}],
        "stop_reason": stop_reason,
        "stop_sequence": stop_sequence,
        "usage": dict(usage) if usage is not None else usage_block(),
        **extra,
    }


def sse(event: str, payload: Mapping[str, JsonValue]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def message_start(
    body: Mapping[str, JsonValue] | None = None, *, usage: Mapping[str, JsonValue] | None = None
) -> bytes:
    return sse(
        "message_start",
        {
            "type": "message_start",
            "message": {
                "id": f"msg_dec1{uuid.uuid4().hex}",
                "type": "message",
                "role": "assistant",
                "model": (body or {}).get("model") or BACKEND,
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": dict(usage) if usage is not None else {"input_tokens": PROMPT_TOKENS, "output_tokens": 1},
            },
        },
    )


def block_start(index: int, block: Mapping[str, JsonValue]) -> bytes:
    return sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": dict(block)})


def block_delta(index: int, delta: Mapping[str, JsonValue]) -> bytes:
    return sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": dict(delta)})


def block_stop(index: int) -> bytes:
    return sse("content_block_stop", {"type": "content_block_stop", "index": index})


def message_delta(
    stop_reason: str = "end_turn",
    *,
    output_tokens: int = COMPLETION_TOKENS,
    stop_sequence: str | None = None,
    usage: Mapping[str, JsonValue] | None = None,
) -> bytes:
    return sse(
        "message_delta",
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence},
            "usage": dict(usage) if usage is not None else {"output_tokens": output_tokens},
        },
    )


MESSAGE_STOP: Final = sse("message_stop", {"type": "message_stop"})
PING: Final = sse("ping", {"type": "ping"})


def text_stream_chunks(body: Mapping[str, JsonValue] | None = None) -> tuple[bytes, ...]:
    """Anthropic's text stream: start, block start, three text deltas, block stop, delta, stop."""
    return (
        message_start(body),
        block_start(0, {"type": "text", "text": ""}),
        *(block_delta(0, {"type": "text_delta", "text": piece}) for piece in STREAM_PIECES),
        block_stop(0),
        message_delta(),
        MESSAGE_STOP,
    )


def stream(*chunks: bytes) -> Reply:
    return Reply(content_type="text/event-stream", chunks=tuple(chunks))


def message_reply(body: Mapping[str, JsonValue]) -> Reply:
    return stream(*text_stream_chunks(body)) if body.get("stream") is True else json_reply(message_json(body))


def default_respond(call: Call) -> Reply:
    """The canned answer for a provider path, as listed in the module docstring."""
    if call.method == "GET" and call.path.endswith(".png"):
        return Reply(body=PNG_BYTES, content_type="image/png")
    if call.method != "POST":
        return json_reply({"data": [], "has_more": False, "first_id": None, "last_id": None})
    if call.path.endswith("/v1/messages/count_tokens"):
        return json_reply({"input_tokens": COUNTED_TOKENS})
    if call.path.endswith("/v1/messages"):
        return message_reply(call.body)
    return anthropic_error(404, "not_found_error", f"dec1 anthropic peer has no route {call.path}")


def answering(content: Sequence[JsonValue], **fields: JsonValue) -> Respond:
    """A responder that answers every POST with one message carrying `content` and `fields`."""

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return default_respond(call)
        return json_reply(message_json(call.body, content=content, **fields))  # type: ignore[arg-type]  # test-local keyword passthrough

    return respond


def streaming(*chunks: bytes) -> Respond:
    """A responder that answers every POST with the given SSE chunks."""

    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return default_respond(call)
        return stream(*chunks)

    return respond


def always(reply: Reply) -> Respond:
    def respond(call: Call) -> Reply:
        if call.method != "POST":
            return default_respond(call)
        return reply

    return respond


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
    model: str = MODEL,
    *,
    priced: bool = True,
    model_info: Mapping[str, JsonValue] | None = None,
    **parameters: JsonValue,
) -> str:
    """A deployment of `model` whose `api_base` is the peer root and whose key is `PROVIDER_KEY`."""
    return scenario.model(
        model_info=model_info,
        **{
            "model": model,
            "api_base": provider.url,
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


def user(text: str | None = None) -> dict[str, JsonValue]:
    return {"role": "user", "content": text or marker()}


def chat_stream_lines(
    gateway: Gateway, body: Mapping[str, JsonValue], *, key: str | None = None
) -> tuple[httpx.Headers, tuple[str, ...]]:
    """Send a streaming chat request and return the response headers and every non-empty line."""
    with gateway.client.stream(
        "POST",
        "/v1/chat/completions",
        json=dict(body),
        headers={"Authorization": f"Bearer {key or gateway.key}"},
    ) as response:
        assert response.status_code == 200, response.read().decode()
        return response.headers, tuple(line for line in response.iter_lines() if line)


def data_events(lines: Sequence[str]) -> tuple[dict[str, JsonValue], ...]:
    """Every `data:` payload except `[DONE]`, decoded."""
    return tuple(
        JSON_OBJECT.validate_json(line.removeprefix("data:").strip())
        for line in lines
        if line.startswith("data:") and not line.endswith("[DONE]")
    )


def deltas(events: Sequence[Mapping[str, JsonValue]]) -> tuple[dict[str, JsonValue], ...]:
    """The `choices[0].delta` of every chunk that has a choice."""
    return tuple(
        object_value(object_value(list_value(event["choices"])[0])["delta"])
        for event in events
        if isinstance(event.get("choices"), list) and event["choices"]
    )


def finish_reasons(events: Sequence[Mapping[str, JsonValue]]) -> tuple[JsonValue, ...]:
    return tuple(
        object_value(list_value(event["choices"])[0]).get("finish_reason")
        for event in events
        if isinstance(event.get("choices"), list)
        and event["choices"]
        and object_value(list_value(event["choices"])[0]).get("finish_reason")
    )


def streamed_text(events: Sequence[Mapping[str, JsonValue]]) -> str:
    return "".join(str(delta.get("content") or "") for delta in deltas(events))


def served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def error_message(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def first_choice(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    choices: Final = body["choices"]
    assert isinstance(choices, list) and choices, body
    return object_value(choices[0])


def first_message(body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return object_value(first_choice(body)["message"])


def sent_messages(call: Call) -> list[JsonValue]:
    return list(list_value(call.body["messages"]))


def sent_message(call: Call, index: int) -> dict[str, JsonValue]:
    return object_value(sent_messages(call)[index])


def content_blocks(message: Mapping[str, JsonValue]) -> list[dict[str, JsonValue]]:
    """A sent message's content as a list of blocks; a string content counts as one text block."""
    content: Final = message["content"]
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [object_value(block) for block in list_value(content)]


def betas(call: Call) -> frozenset[str]:
    """The `anthropic-beta` values the proxy sent, split on commas and trimmed."""
    return frozenset(beta.strip() for beta in call.headers.get("anthropic-beta", "").split(",") if beta.strip())


def tool(name: str, description: str | None = None, **properties: JsonValue) -> dict[str, JsonValue]:
    """An OpenAI function tool with a string property per keyword, all required."""
    function: Final[dict[str, JsonValue]] = {
        "name": name,
        "parameters": {
            "type": "object",
            "properties": {key: value for key, value in properties.items()},
            "required": list(properties),
        },
    }
    if description is not None:
        function["description"] = description
    return {"type": "function", "function": function}


def spend_row_by_group(gateway: Gateway, model_group: str, *, seconds: float = LANDING_SECONDS) -> dict[str, JsonValue]:
    """The one SpendLogs row of a model group created for one test (the deployment name is unique per test)."""
    rows: Final = eventually(
        lambda: gateway.rows(f'SELECT {SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE model_group = %s', (model_group,)),
        lambda found: len(found) == 1,
        seconds=seconds,
    )
    return rows[0]
