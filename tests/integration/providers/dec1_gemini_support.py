"""Shared rig for the December 1 Google AI Studio contracts (`test_dec1_gemini_*`).

Every test registers one `gemini/<model>` deployment on the shared proxy whose `api_base` points at an
owned TCP peer (`wire_server`) and whose `api_key` is a synthetic AI Studio key. The peer answers every
Google AI Studio surface the proxy can call through that base: `models/{model}:generateContent`,
`:streamGenerateContent`, `:batchEmbedContents`, `:countTokens`, `cachedContents`, and the Files API
upload and resource paths, so a test can read back the exact request (method, path, query, headers,
body) the proxy sent and script the answer.

The generic Gemini wire helpers (frames, usage metadata, Google error envelope, SSE parsing) are the
same ones the Vertex suite uses, so they are imported from `dec1_vertex_support`; everything that is
specific to AI Studio (no OAuth, the `x-goog-api-key` header, the `v1beta` path shapes, Files API and
context caching replies) lives here.

Harness facts that shape these helpers: the shared proxy has `litellm_settings.cache: true`, so every
prompt carries a fresh marker; the proxy may probe a new deployment with a `GET`, so tests that count
calls filter on the path they expect; and nothing here builds a URL on a Google host.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, list_value, object_value, string_value
from integration._support.wire import Reply, Request, Wire, wire_server
from integration.providers.dec1_vertex_support import (
    JSON_OBJECT,
    at,
    body_of,
    choice,
    google_error,
    json_of,
    marker,
    message_of,
    path_of,
    query_of,
    sse_payloads,
    streamed_text,
    usage_metadata,
)
from pydantic import JsonValue

__all__ = [
    "JSON_OBJECT",
    "at",
    "body_of",
    "choice",
    "google_error",
    "json_of",
    "marker",
    "message_of",
    "path_of",
    "query_of",
    "sse_payloads",
    "streamed_text",
    "usage_metadata",
]

GEMINI_KEY: Final = "dec1-ai-studio-key"
"""The synthetic AI Studio key every deployment carries; it must only ever travel as `x-goog-api-key`."""
FLASH: Final = "gemini-2.5-flash"
"""A Gemini 2.x model: thinking maps to `thinkingBudget`, media resolution goes in `generationConfig`."""
PRO: Final = "gemini-2.5-pro"
"""The 2.5 model with a 200k token price tier in the cost map (`input_cost_per_token_above_200k_tokens`)."""
GEMINI_3_FLASH: Final = "gemini-3-flash-preview"
"""A Gemini 3 flash model: thinking maps to `thinkingLevel`, `minimal` is a level it accepts."""
GEMINI_3_PRO: Final = "gemini-3.1-pro-preview"
"""A Gemini 3 pro model: thinking maps to `thinkingLevel`, `minimal` falls back to `low`."""
EMBEDDING: Final = "gemini-embedding-001"
FILES_PREFIX: Final = "https://generativelanguage.googleapis.com/v1beta/files/"
"""`GEMINI_FILES_API_URI_PREFIX` in `litellm/llms/vertex_ai/common_utils.py`."""
CACHE_NAME: Final = "cachedContents/dec1-cache"
LONG_TEXT: Final = " ".join(
    f"Contract clause {index}: the gateway keeps this clause in the cache." for index in range(700)
)
"""About ten thousand tokens, well above the 1024 token default `prompt_cache_min_tokens` floor."""


def generate_path(model: str = FLASH, *, prefix: str = "/v1beta", method: str = "generateContent") -> str:
    return f"{prefix}/models/{model}:{method}"


def headers_of(request: Request) -> dict[str, str]:
    return {name.lower(): value for name, value in request.headers.items()}


# ---------------------------------------------------------------- the peer


@dataclass
class Peer:
    """One owned TCP peer standing in for `generativelanguage.googleapis.com`; it keeps every request."""

    wire: Wire
    seen: list[Request] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        return self.wire.url.rstrip("/")

    def requests(self) -> tuple[Request, ...]:
        with self._lock:
            self.seen.extend(self.wire.drain())
            return tuple(self.seen)

    def to(self, path_suffix: str, method: str = "POST") -> tuple[Request, ...]:
        return tuple(
            request
            for request in self.requests()
            if request.method == method and path_of(request).endswith(path_suffix)
        )

    def generate_calls(self) -> tuple[Request, ...]:
        return tuple(
            request
            for request in self.requests()
            if request.method == "POST"
            and (":generateContent" in path_of(request) or ":streamGenerateContent" in path_of(request))
        )

    def one_call(self) -> Request:
        calls: Final = self.generate_calls()
        assert len(calls) == 1, [call.target for call in calls]
        return calls[0]

    def one_body(self) -> dict[str, JsonValue]:
        return body_of(self.one_call())


@contextmanager
def peer(respond: Callable[[Request], Reply]) -> Iterator[Peer]:
    def answer(request: Request) -> Reply:
        if request.method == "GET" and path_of(request) in ("", "/", "/v1beta/models", "/models"):
            return Reply(body=b'{"models": []}')
        return respond(request)

    with wire_server(answer) as wire:
        yield Peer(wire)


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    scenario: Scenario
    peer: Peer
    model: str

    def chat(self, text: str | None = None, **fields: JsonValue) -> httpx.Response:
        body: Final[dict[str, JsonValue]] = {
            "model": self.model,
            "messages": [{"role": "user", "content": text if text is not None else f"ping {marker()}"}],
            **fields,
        }
        return self.gateway.request("POST", "/v1/chat/completions", body)

    def post(
        self, path: str, body: Mapping[str, JsonValue], headers: Mapping[str, str] | None = None
    ) -> httpx.Response:
        return self.gateway.request("POST", path, {"model": self.model, **body}, headers=headers)

    def stream(self, path: str, body: Mapping[str, JsonValue]) -> tuple[httpx.Response, str]:
        with self.gateway.client.stream(
            "POST",
            path,
            json={"model": self.model, **body, "stream": True},
            headers={"Authorization": f"Bearer {self.gateway.key}"},
        ) as response:
            text: Final = "".join(response.iter_text())
            return response, text


@contextmanager
def gemini(
    gateway: Gateway,
    respond: Callable[[Request], Reply],
    *,
    backend: str = FLASH,
    base_suffix: str = "/v1beta",
    api_key: str | None = GEMINI_KEY,
    model_info: Mapping[str, JsonValue] | None = None,
    **params: JsonValue,
) -> Iterator[Rig]:
    """Register `gemini/<backend>` with `api_base` at a new peer (plus `base_suffix`, `/v1beta` by default,
    the shape the AI Studio docs give for a custom base) and the synthetic key."""
    with peer(respond) as owned, gateway.scenario() as scenario:
        name: Final = scenario.model(
            model=f"gemini/{backend}",
            api_base=owned.url + base_suffix,
            api_key=api_key,
            model_info=model_info,
            **params,
        )
        yield Rig(gateway, scenario, owned, name)


# ---------------------------------------------------------------- Gemini replies


def gemini_json(
    parts: Sequence[JsonValue] | None = None,
    *,
    finish: str = "STOP",
    usage: Mapping[str, JsonValue] | None = None,
    text: str = "pong",
    model: str = FLASH,
    **candidate_fields: JsonValue,
) -> dict[str, JsonValue]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": list(parts) if parts is not None else [{"text": text}]},
                "finishReason": finish,
                "index": 0,
                **candidate_fields,
            }
        ],
        "usageMetadata": dict(usage) if usage is not None else usage_metadata(),
        "modelVersion": model,
        "responseId": f"resp-{marker()[:16]}",
    }


def reply_json(value: Mapping[str, JsonValue], status: int = 200, headers: Mapping[str, str] | None = None) -> Reply:
    return Reply(status=status, body=json.dumps(value).encode(), headers=dict(headers or {}))


def gemini_reply(**fields: JsonValue) -> Reply:
    return reply_json(gemini_json(**fields))  # pyright: ignore[reportArgumentType]  # test-local keyword passthrough


def gemini_sse(
    pieces: Sequence[str] = ("hello ", "stream"), *, usage: Mapping[str, JsonValue] | None = None, finish: str = "STOP"
) -> Reply:
    frames: Final[list[dict[str, JsonValue]]] = [
        {"candidates": [{"content": {"role": "model", "parts": [{"text": piece}]}, "index": 0}], "modelVersion": FLASH}
        for piece in pieces
    ]
    frames[-1] = {
        **frames[-1],
        "candidates": [{**object_value(list_value(frames[-1]["candidates"])[0]), "finishReason": finish}],
        "usageMetadata": dict(usage) if usage is not None else usage_metadata(),
    }
    return Reply(
        content_type="text/event-stream",
        chunks=tuple(f"data: {json.dumps(frame)}\r\n\r\n".encode() for frame in frames),
    )


def gemini_echo(request: Request) -> Reply:
    if ":streamGenerateContent" in path_of(request):
        return gemini_sse()
    return gemini_reply()


def search_grounding(queries: Sequence[str] = ("weather paris",)) -> dict[str, JsonValue]:
    """A Google Search `groundingMetadata` with one web source supporting characters 0..5 of the answer."""
    return {
        "webSearchQueries": list(queries),
        "searchEntryPoint": {"renderedContent": "<div>chip</div>"},
        "groundingChunks": [{"web": {"uri": "https://example.com/paris", "title": "example.com"}}],
        "groundingSupports": [
            {"segment": {"startIndex": 0, "endIndex": 5, "text": "Sunny"}, "groundingChunkIndices": [0]}
        ],
    }


def cached_contents_responder(
    *,
    listed: Sequence[Mapping[str, JsonValue]] = (),
    created: Mapping[str, JsonValue] | None = None,
    list_status: int = 200,
    generate: Callable[[Request], Reply] | None = None,
) -> Callable[[Request], Reply]:
    """Answer `GET .../cachedContents` with `listed`, `POST .../cachedContents` with `created` (by default
    `CACHE_NAME` valid for an hour) and `:generateContent` with `generate` (by default a reply that
    reports 5120 cached tokens)."""

    def respond(request: Request) -> Reply:
        path: Final = path_of(request)
        if path.endswith("/cachedContents") and request.method == "GET":
            if list_status != 200:
                return google_error(list_status, "NOT_FOUND", "cachedContents is not served here")
            return reply_json({"cachedContents": [dict(item) for item in listed]})
        if path.endswith("/cachedContents") and request.method == "POST":
            return reply_json(
                dict(created)
                if created is not None
                else {"name": CACHE_NAME, "model": f"models/{FLASH}", "expireTime": "2999-01-01T00:00:00.000000Z"}
            )
        if generate is not None:
            return generate(request)
        return gemini_reply(
            usage=usage_metadata(5129, 2, cachedContentTokenCount=5120),
            text="700",
        )

    return respond


def cached_messages(ttl: str | None = None, *, system: str | None = None) -> list[JsonValue]:
    """A long cached user block plus an uncached question, optionally with a system message and a ttl."""
    control: Final[dict[str, JsonValue]] = {"type": "ephemeral", **({"ttl": ttl} if ttl is not None else {})}
    messages: Final[list[JsonValue]] = [] if system is None else [{"role": "system", "content": system}]
    messages.append(
        {"role": "user", "content": [{"type": "text", "text": f"{LONG_TEXT} {marker()}", "cache_control": control}]}
    )
    messages.append({"role": "user", "content": "How many clauses are there?"})
    return messages


def text_of(content: JsonValue) -> str:
    """The concatenated text parts of a Gemini `contents` entry."""
    return "".join(
        string_value(object_value(part)["text"])
        for part in list_value(object_value(content)["parts"])
        if "text" in object_value(part)
    )


def spend_rows(gateway: Gateway, model_group: str) -> list[dict[str, JsonValue]]:
    return gateway.rows(
        "SELECT custom_llm_provider, model, prompt_tokens, completion_tokens, total_tokens, spend, call_type "
        'FROM "LiteLLM_SpendLogs" WHERE model_group = %s',
        (model_group,),
    )
