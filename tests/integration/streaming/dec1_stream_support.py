"""Shared rig for the `test_dec1_stream_*` contracts.

Every contract here is black box: an owned TCP peer (`wire_server`) plays the provider, a deployment that
points at it is registered through `/model/new`, a virtual key scoped to that deployment is minted, and
the test talks to the shared proxy over HTTP. Nothing in this module patches or imports product code, so
the same tests describe the Python proxy and its Rust replacement.

The shared proxy caches responses in Redis, so every helper that builds a request puts a fresh marker in
the prompt unless the test is about the cache.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import httpx
from integration._support.client import Gateway, eventually, held, object_value
from integration._support.database import read_rows
from integration._support.upstream import aws_event_stream_frame
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue, TypeAdapter

JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
PROVIDER_KEY: Final = "synthetic-provider-key"
UPSTREAM_MODEL: Final = "gpt-4o-mini"
UPSTREAM_CREATED: Final = 1700000000
DONE: Final = b"data: [DONE]\n\n"
USAGE: Final[Mapping[str, JsonValue]] = {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15}
DETAILED_USAGE: Final[Mapping[str, JsonValue]] = {
    "prompt_tokens": 20,
    "completion_tokens": 9,
    "total_tokens": 29,
    "prompt_tokens_details": {"cached_tokens": 8},
    "completion_tokens_details": {"reasoning_tokens": 3},
}
EVENT_STREAM: Final = "application/vnd.amazon.eventstream"
CLAUDE: Final = "claude-sonnet-5-5"
GEMINI: Final = "gemini-3.8-flash"
BEDROCK_CLAUDE: Final = "anthropic.claude-3-5-sonnet-20240620-v1:0"
SPEND_COLUMNS: Final = (
    "request_id, call_type, status, model, model_group, custom_llm_provider, prompt_tokens, completion_tokens, "
    "total_tokens, spend, cache_hit, metadata, litellm_call_id, "
    + ", ".join(
        f'to_char("{column}", \'YYYY-MM-DD"T"HH24:MI:SS.US\') AS "{column}"'
        for column in ("startTime", "endTime", "completionStartTime")
    )
)
"""Timestamps come back as fixed width ISO text so the JSON row adapter accepts them and they sort as text."""


def marker() -> str:
    return uuid.uuid4().hex


def data(payload: JsonValue) -> bytes:
    """One OpenAI style SSE frame."""
    return b"data: " + json.dumps(payload, ensure_ascii=False).encode() + b"\n\n"


def event(name: str, payload: JsonValue) -> bytes:
    """One Anthropic or Responses style SSE frame with an `event:` line."""
    return f"event: {name}\n".encode() + data(payload)


def chunk(identity: str, choices: Sequence[JsonValue], **extra: JsonValue) -> bytes:
    return data(
        {
            "id": identity,
            "object": "chat.completion.chunk",
            "created": UPSTREAM_CREATED,
            "model": UPSTREAM_MODEL,
            "choices": list(choices),
            **extra,
        }
    )


def delta(
    identity: str,
    fields: Mapping[str, JsonValue],
    *,
    finish: str | None = None,
    index: int = 0,
    **choice_extra: JsonValue,
) -> bytes:
    return chunk(identity, [{"index": index, "delta": dict(fields), "finish_reason": finish, **choice_extra}])


def usage_only(identity: str, usage: Mapping[str, JsonValue] = USAGE) -> bytes:
    return chunk(identity, [], usage=dict(usage))


def text_frames(
    identity: str,
    pieces: Sequence[str],
    *,
    finish: str = "stop",
    usage: Mapping[str, JsonValue] | None = USAGE,
) -> tuple[bytes, ...]:
    """Role and first piece, the other pieces, a finish chunk with an empty delta, usage, `[DONE]`."""
    first: Final = delta(identity, {"role": "assistant", "content": pieces[0]})
    rest: Final = tuple(delta(identity, {"content": piece}) for piece in pieces[1:])
    tail: Final = () if usage is None else (usage_only(identity, usage),)
    return (first, *rest, delta(identity, {}, finish=finish), *tail, DONE)


def completion_json(identity: str, text: str, usage: Mapping[str, JsonValue] = USAGE) -> bytes:
    return json.dumps(
        {
            "id": identity,
            "object": "chat.completion",
            "created": UPSTREAM_CREATED,
            "model": UPSTREAM_MODEL,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": dict(usage),
        }
    ).encode()


def sse(frames: Sequence[bytes], **reply: object) -> Reply:
    return Reply(content_type="text/event-stream", chunks=tuple(frames), **reply)  # type: ignore[arg-type]


def serving(frames: Sequence[bytes], *, json_body: bytes | None = None) -> Callable[[Request], Reply]:
    """Answer a POST that asks to stream with `frames`, any other POST with `json_body`."""

    def respond(request: Request) -> Reply:
        if request.method != "POST":
            return Reply(status=404)
        if request_json(request).get("stream") is True or json_body is None:
            return sse(frames)
        return Reply(body=json_body)

    return respond


def request_json(request: Request) -> dict[str, JsonValue]:
    try:
        return JSON_OBJECT.validate_json(request.body)
    except ValueError:
        return {}


def payloads(text: str) -> list[str]:
    """Every `data:` payload of an SSE body in order, the `[DONE]` sentinel included."""
    return [line.removeprefix("data:").strip() for line in text.splitlines() if line.startswith("data:")]


def chunks_of(text: str) -> list[dict[str, JsonValue]]:
    return [JSON_OBJECT.validate_json(payload) for payload in payloads(text) if payload != "[DONE]"]


def events_of(text: str) -> list[tuple[str, dict[str, JsonValue]]]:
    """`(event name, data object)` for every SSE block that has an `event:` line."""
    blocks: Final = [block for block in text.replace("\r\n", "\n").split("\n\n") if "event:" in block]

    def parse(block: str) -> tuple[str, dict[str, JsonValue]]:
        lines: Final = block.splitlines()
        name: Final = next(line.removeprefix("event:").strip() for line in lines if line.startswith("event:"))
        body: Final = next(line.removeprefix("data:").strip() for line in lines if line.startswith("data:"))
        return name, JSON_OBJECT.validate_json(body)

    return [parse(block) for block in blocks]


def choices_of(chunks: Sequence[Mapping[str, JsonValue]]) -> list[dict[str, JsonValue]]:
    return [
        object_value(choice)
        for item in chunks
        for choice in (item.get("choices") or [])  # type: ignore[union-attr]
    ]


def deltas_of(chunks: Sequence[Mapping[str, JsonValue]], index: int = 0) -> list[dict[str, JsonValue]]:
    return [object_value(choice.get("delta") or {}) for choice in choices_of(chunks) if choice.get("index", 0) == index]


def content_of(chunks: Sequence[Mapping[str, JsonValue]], index: int = 0, field: str = "content") -> str:
    return "".join(str(item.get(field) or "") for item in deltas_of(chunks, index))


def finish_reasons(chunks: Sequence[Mapping[str, JsonValue]], index: int = 0) -> list[JsonValue]:
    return [
        choice["finish_reason"]
        for choice in choices_of(chunks)
        if choice.get("index", 0) == index and choice.get("finish_reason") is not None
    ]


def usage_chunks(chunks: Sequence[Mapping[str, JsonValue]]) -> list[Mapping[str, JsonValue]]:
    return [item for item in chunks if item.get("usage") is not None]


def tool_call_fragments(chunks: Sequence[Mapping[str, JsonValue]], tool_index: int) -> list[dict[str, JsonValue]]:
    return [
        object_value(call)
        for item in deltas_of(chunks)
        for call in (item.get("tool_calls") or [])  # type: ignore[union-attr]
        if object_value(call).get("index") == tool_index
    ]


def joined_arguments(fragments: Sequence[Mapping[str, JsonValue]]) -> str:
    return "".join(str(object_value(item.get("function") or {}).get("arguments") or "") for item in fragments)


def bedrock_frame(event_type: str, payload: Mapping[str, JsonValue]) -> bytes:
    return aws_event_stream_frame(
        {":event-type": event_type, ":content-type": "application/json", ":message-type": "event"},
        json.dumps(payload, separators=(",", ":")).encode(),
    )


@dataclass(frozen=True, slots=True)
class Rig:
    """One scripted deployment on the shared proxy, the key allowed to call it, and its provider peer."""

    gateway: Gateway
    wire: Wire
    model: str
    key: str

    def post(
        self,
        path: str,
        body: Mapping[str, JsonValue],
        *,
        headers: Mapping[str, str] | None = None,
        key: str | None = None,
    ) -> httpx.Response:
        return self.gateway.client.post(
            path,
            json=dict(body),
            headers={"Authorization": f"Bearer {self.key if key is None else key}", **(headers or {})},
            timeout=60,
        )

    def chat_body(self, prompt: str | None = None, **extra: JsonValue) -> dict[str, JsonValue]:
        text: Final = f"dec1 stream probe {marker()}" if prompt is None else prompt
        return {"model": self.model, "messages": [{"role": "user", "content": text}], "stream": True, **extra}

    def chat(
        self, prompt: str | None = None, *, headers: Mapping[str, str] | None = None, **extra: JsonValue
    ) -> httpx.Response:
        return self.post("/v1/chat/completions", self.chat_body(prompt, **extra), headers=headers)

    def messages(self, prompt: str | None = None, **extra: JsonValue) -> httpx.Response:
        text: Final = f"dec1 stream probe {marker()}" if prompt is None else prompt
        return self.post(
            "/v1/messages",
            {
                "model": self.model,
                "max_tokens": 64,
                "messages": [{"role": "user", "content": text}],
                "stream": True,
                **extra,
            },
            headers={"anthropic-version": "2023-06-01"},
        )

    def responses(self, prompt: str | None = None, **extra: JsonValue) -> httpx.Response:
        text: Final = f"dec1 stream probe {marker()}" if prompt is None else prompt
        return self.post("/v1/responses", {"model": self.model, "input": text, "stream": True, **extra})

    def upstream_bodies(self) -> list[dict[str, JsonValue]]:
        return [request_json(request) for request in self.wire.drain() if request.method == "POST"]

    def spend_rows(self, expected: int = 1, *, settle: float = 3) -> list[dict[str, JsonValue]]:
        """The SpendLogs rows written for this rig's key, once `expected` of them exist and no more arrive."""
        digest: Final = sha256(self.key.encode()).hexdigest()
        query: Final = (
            f'SELECT {SPEND_COLUMNS} FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "LiteLLM_SpendLogs"."startTime"'
        )
        eventually(lambda: read_rows(query, (digest,)), lambda rows: len(rows) >= expected, seconds=70)  # type: ignore[arg-type]
        return held(lambda: read_rows(query, (digest,)), lambda rows: len(rows) == expected, holding=settle)  # type: ignore[arg-type]

    def key_spend(self) -> float:
        digest: Final = sha256(self.key.encode()).hexdigest()
        rows: Final = read_rows('SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s', (digest,))
        return float(str(rows[0]["spend"])) if rows else 0.0


def _registered(gateway: Gateway, model: str) -> None:
    def names() -> tuple[JsonValue, ...]:
        entries: Final = gateway.get("/model/info")["data"]
        assert isinstance(entries, list), entries
        return tuple(object_value(entry).get("model_name") for entry in entries)

    eventually(names, lambda found: model in found, seconds=60)


@contextmanager
def scripted(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    model: str = f"openai/{UPSTREAM_MODEL}",
    base_suffix: str = "/v1",
    key_fields: Mapping[str, JsonValue] | None = None,
    **litellm_params: JsonValue,
) -> Iterator[Rig]:
    """Register `model` against an owned peer that answers with `respond`, and mint a key scoped to it.

    `base_suffix` is appended to the peer URL for `api_base`; OpenAI compatible providers want `/v1`,
    Anthropic and Gemini want the bare origin. Extra keywords become `litellm_params` of the deployment.
    """
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        parameters: Final[dict[str, JsonValue]] = {
            "model": model,
            "api_key": PROVIDER_KEY,
            "api_base": wire.url + base_suffix,
            **litellm_params,
        }
        name: Final = scenario.model(**parameters)
        _registered(gateway, name)
        key: Final = scenario.key(models=[name], **(key_fields or {}))
        yield Rig(gateway, wire, name, key)


@contextmanager
def bedrock_scripted(gateway: Gateway, respond: Callable[[Request], Reply]) -> Iterator[Rig]:
    """A Bedrock Converse deployment whose runtime endpoint is the owned peer."""
    with wire_server(respond) as wire, gateway.scenario() as scenario:
        name: Final = scenario.model(
            model=f"bedrock/converse/{BEDROCK_CLAUDE}",
            api_key="synthetic-bedrock-token",
            aws_region_name="us-east-1",
            aws_bedrock_runtime_endpoint=wire.url,
        )
        _registered(gateway, name)
        yield Rig(gateway, wire, name, scenario.key(models=[name]))
