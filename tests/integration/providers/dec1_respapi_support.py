"""Shared rig for the `test_dec1_respapi_*` files: `/v1/responses` served through the Responses to chat bridge.

A deployment whose provider has no native Responses API (Anthropic, DeepSeek, Gemini, Bedrock Converse) or that
opts out of it (`use_chat_completions_api: true` on an `openai/` deployment) is served by
`litellm/responses/litellm_completion_transformation/`: the Responses request is rewritten into a chat completion
(`transformation.py:transform_responses_api_request_to_chat_completion_request`), run through `litellm.acompletion`,
and the chat answer is rewritten back into a Responses object or a Responses event stream. The tests use three
carriers so the contract is read provider-agnostically:

    CHAT_BRIDGE   `openai/gpt-4o-mini` with `use_chat_completions_api: true`: the peer sees the exact chat body
                  the bridge built at `POST /v1/chat/completions`, with no provider rewrite on top.
    DEEPSEEK      `deepseek/deepseek-chat`: a chat only provider that bridges without any opt in.
    ANTHROPIC     `anthropic/claude-sonnet-4-5-20250929`: the bridge output goes through the Anthropic chat
                  transform, so the peer sees `POST /v1/messages`.

Every test talks to one owned HTTP peer that records each request (method, path, headers, decoded JSON body) and
answers with a canned reply chosen from the path, unless the test passes its own `respond`. Canned chat answers
say `REPLY_TEXT` with usage 11 prompt and 7 completion tokens; canned Anthropic answers say the same with
`input_tokens` 11 and `output_tokens` 7. Deployments are priced at 0.001 per input token and 0.002 per output
token, so one canned call costs 11 * 0.001 + 7 * 0.002 = 0.025.

The shared proxy has a Redis response cache, so every prompt carries a fresh `marker()` and every request sends
`cache: {"no-cache": true}`. Every deployment sets `num_retries: 0` and `max_retries: 0`, so a scripted failure
reaches the peer exactly once.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

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

PROVIDER_KEY: Final = "dec1-respapi-provider-key"
INPUT_RATE: Final = 0.001
OUTPUT_RATE: Final = 0.002
PROMPT_TOKENS: Final = 11
COMPLETION_TOKENS: Final = 7
CALL_COST: Final = 0.025
REPLY_TEXT: Final = "dec1 bridged reply"
STREAM_PIECES: Final = ("dec1 ", "bridged ", "reply")
LANDING_SECONDS: Final = 70
NO_CACHE: Final[dict[str, JsonValue]] = {"no-cache": True}
NO_RETRIES: Final[Mapping[str, JsonValue]] = {"num_retries": 0, "max_retries": 0}
PRICES: Final[Mapping[str, JsonValue]] = {"input_cost_per_token": INPUT_RATE, "output_cost_per_token": OUTPUT_RATE}

CHAT_BRIDGE: Final = "openai/gpt-4o-mini"
"""Bridged only because the deployment sets `use_chat_completions_api: true` (`responses/main.py:1280`)."""
REASONING_CHAT_BRIDGE: Final = "openai/o3-mini"
"""A chat mode reasoning model, so `reasoning_effort` is accepted on the chat wire and nothing bridges back."""
DEEPSEEK: Final = "deepseek/deepseek-chat"
ANTHROPIC_BACKEND: Final = "claude-sonnet-4-5-20250929"
ANTHROPIC: Final = f"anthropic/{ANTHROPIC_BACKEND}"
CHAT_PATH: Final = "/v1/chat/completions"
MESSAGES_PATH: Final = "/v1/messages"
SPEND_COLUMNS: Final = (
    "request_id, call_type, spend, total_tokens, prompt_tokens, completion_tokens, model, model_group, "
    "custom_llm_provider, api_base, status, session_id"
)


@dataclass(frozen=True, slots=True)
class Call:
    """One request the proxy sent to the provider peer."""

    method: str
    path: str
    headers: Mapping[str, str]
    body: Mapping[str, JsonValue]


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

    def last(self) -> Call:
        calls: Final = self.posts()
        assert calls, "the peer received no POST"
        return calls[-1]

    def all(self) -> tuple[Call, ...]:
        with self.lock:
            return tuple(self.seen)


def marker() -> str:
    """A unique string, so the shared gateway's response cache never answers a request by accident."""
    return f"dec1-respapi-{uuid.uuid4().hex}"


def json_reply(body: Mapping[str, JsonValue], status: int = 200, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(body).encode(), headers=headers or {})


def openai_error(status: int, message: str, *, kind: str | None = None, code: str | None = None) -> Reply:
    default_kind: Final = (
        "rate_limit_error" if status == 429 else "invalid_request_error" if status < 500 else "server_error"
    )
    return json_reply(
        {"error": {"message": message, "type": kind or default_kind, "param": None, "code": code}}, status
    )


def sse(payload: Mapping[str, JsonValue]) -> bytes:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


def usage(prompt: int = PROMPT_TOKENS, completion: int = COMPLETION_TOKENS, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion, **extra}


def chat_json(
    body: Mapping[str, JsonValue],
    *,
    content: str | None = REPLY_TEXT,
    finish_reason: str = "stop",
    message: Mapping[str, JsonValue] | None = None,
    identity: str | None = None,
    reported_usage: Mapping[str, JsonValue] | None = None,
) -> dict[str, JsonValue]:
    """A chat completion whose only choice says `content` with any extra `message` fields."""
    return {
        "id": identity or f"chatcmpl-dec1-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": body.get("model"),
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, **(message or {})},
                "finish_reason": finish_reason,
            }
        ],
        "usage": dict(reported_usage) if reported_usage is not None else usage(),
    }


def tool_call(call_id: str, name: str, arguments: str) -> dict[str, JsonValue]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def chat_chunk(identity: str, delta: Mapping[str, JsonValue], finish_reason: str | None = None) -> bytes:
    return sse(
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": dict(delta), "finish_reason": finish_reason}],
        }
    )


def chat_usage_chunk(identity: str, **extra: JsonValue) -> bytes:
    return sse(
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [],
            "usage": usage(**extra),
        }
    )


def chat_stream(frames: Sequence[bytes]) -> Reply:
    return Reply(content_type="text/event-stream", chunks=(*frames, b"data: [DONE]\n\n"))


def text_stream(identity: str | None = None) -> Reply:
    """Role chunk, the three `STREAM_PIECES`, a `stop` finish chunk and a usage only chunk."""
    stream_id: Final = identity or f"chatcmpl-dec1-{uuid.uuid4().hex}"
    return chat_stream(
        (
            chat_chunk(stream_id, {"role": "assistant", "content": ""}),
            *(chat_chunk(stream_id, {"content": piece}) for piece in STREAM_PIECES),
            chat_chunk(stream_id, {}, "stop"),
            chat_usage_chunk(stream_id),
        )
    )


def anthropic_message(
    body: Mapping[str, JsonValue],
    *,
    content: Sequence[Mapping[str, JsonValue]] | None = None,
    stop_reason: str = "end_turn",
) -> dict[str, JsonValue]:
    return {
        "id": f"msg_dec1{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": body.get("model"),
        "content": [dict(block) for block in content]
        if content is not None
        else [{"type": "text", "text": REPLY_TEXT}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": PROMPT_TOKENS, "output_tokens": COMPLETION_TOKENS},
    }


def default_respond(call: Call) -> Reply:
    if call.method != "POST":
        return json_reply({"object": "list", "data": []})
    if call.path.endswith("/chat/completions"):
        return text_stream() if call.body.get("stream") is True else json_reply(chat_json(call.body))
    if call.path.endswith("/v1/messages"):
        return json_reply(anthropic_message(call.body))
    return openai_error(404, f"dec1 respapi peer has no route {call.path}")


def _decoded(request: Request) -> Mapping[str, JsonValue]:
    if not request.body:
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
        path: Final = request.target.partition("?")[0]
        call: Final = Call(request.method, path, request.headers, _decoded(request))
        with holder[0].lock:
            holder[0].seen.append(call)
        return respond(call)

    with wire_server(handle) as wire:
        holder.append(Peer(wire.url))
        yield holder[0]


def chat_bridge(scenario: Scenario, provider: Peer, *, model: str = CHAT_BRIDGE, **parameters: JsonValue) -> str:
    """An `openai/` deployment forced through the bridge with `use_chat_completions_api: true`."""
    return scenario.model(
        model=model,
        api_base=f"{provider.url}/v1",
        api_key=PROVIDER_KEY,
        use_chat_completions_api=True,
        **NO_RETRIES,
        **PRICES,
        **parameters,
    )


def deepseek(scenario: Scenario, provider: Peer, **parameters: JsonValue) -> str:
    """A chat only provider: bridged with no opt in, `api_base` ends in `/v1` so the peer sees `/v1/chat/completions`."""
    return scenario.model(
        model=DEEPSEEK, api_base=f"{provider.url}/v1", api_key=PROVIDER_KEY, **NO_RETRIES, **PRICES, **parameters
    )


def anthropic(scenario: Scenario, provider: Peer, *, model: str = ANTHROPIC, **parameters: JsonValue) -> str:
    """An Anthropic deployment; the Anthropic config appends `/v1/messages` to the bare peer origin."""
    return scenario.model(
        model=model, api_base=provider.url, api_key=PROVIDER_KEY, **NO_RETRIES, **PRICES, **parameters
    )


def responses(
    gateway: Gateway,
    model: str,
    *,
    key: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    """`POST /v1/responses` with a fresh marker as `input` unless the test passes its own."""
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "input": extra.pop("input", None) or marker(),
        "cache": NO_CACHE,
        **extra,
    }
    return gateway.request("POST", "/v1/responses", body, key=key, headers=headers)


def stream_events(
    gateway: Gateway, model: str, *, key: str | None = None, **extra: JsonValue
) -> list[dict[str, JsonValue]]:
    """Stream `/v1/responses` and return every `data:` JSON object in order."""
    body: Final[dict[str, JsonValue]] = {
        "model": model,
        "input": extra.pop("input", None) or marker(),
        "cache": NO_CACHE,
        "stream": True,
        **extra,
    }
    with gateway.client.stream(
        "POST", "/v1/responses", json=body, headers={"Authorization": f"Bearer {key or gateway.key}"}
    ) as response:
        assert response.status_code == 200, response.read().decode()
        return [
            JSON_OBJECT.validate_json(line.removeprefix("data:").strip())
            for line in response.iter_lines()
            if line.startswith("data:") and line.removeprefix("data:").strip().startswith("{")
        ]


def types_of(events: Sequence[Mapping[str, JsonValue]]) -> list[str]:
    return [string_value(event["type"]) for event in events]


def one_event(events: Sequence[Mapping[str, JsonValue]], kind: str) -> dict[str, JsonValue]:
    found: Final = [dict(event) for event in events if event.get("type") == kind]
    assert len(found) == 1, types_of(events)
    return found[0]


def served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def error_message(response: httpx.Response) -> str:
    return string_value(object_value(JSON_OBJECT.validate_json(response.content)["error"])["message"])


def output_items(body: Mapping[str, JsonValue], kind: str) -> list[dict[str, JsonValue]]:
    return [object_value(item) for item in list_value(body["output"]) if object_value(item).get("type") == kind]


def output_text(body: Mapping[str, JsonValue]) -> str:
    return "".join(
        string_value(object_value(part)["text"])
        for item in output_items(body, "message")
        for part in list_value(item["content"])
        if object_value(part).get("type") == "output_text"
    )


def chat_messages(call: Call) -> list[dict[str, JsonValue]]:
    return [object_value(message) for message in list_value(call.body["messages"])]


def spend_rows(
    gateway: Gateway, model_group: str, count: int = 1, *, seconds: float = LANDING_SECONDS
) -> list[dict[str, JsonValue]]:
    """The SpendLogs rows of one test's model group (deployment names are unique per test), oldest first."""
    return eventually(
        lambda: gateway.rows(
            f'SELECT {SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE model_group = %s ORDER BY "startTime"',
            (model_group,),
        ),
        lambda found: len(found) == count,
        seconds=seconds,
    )
