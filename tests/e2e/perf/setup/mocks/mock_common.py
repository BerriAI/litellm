"""Shared behaviour of the perf mock providers.

Every reply is generated for the request just received (see `replies.py`): sized from its prompt and output
cap, a tool call when tools were offered, in the provider's own wire format with its usage fields, ids,
event order and chunk sizes, so the gateway parses, prices and streams it as it would the real thing. The
model id stays a real one (`gpt-6-luna`, `anthropic.claude-sonnet-5`) so every name-based branch in the
gateway runs, and one mock pod per provider serves any number of deployments. How long a mock waits before answering is the one setting a case
controls, `POST /mock/latency {"seconds": 1.0, "jitter_seconds": 0.25}`, set by the test's setup directly
on the pod, never through the gateway. The wait, the reply's size and its ids are pure functions of the request marker (`perfreq-000123-n`, put in the prompt by the plan
generator), so the test can join what each provider received against what each client saw.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass
from typing import Final, Literal, cast

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from replies import Route, render_chunks, render_json, reply_for
from reply_common import Provider, Reply, chunks_after, unit

MARKER: Final = re.compile(r"perfreq-\d{6}-[nib]")
DELAY_HEADER: Final = "x-mock-delay-s"
Flag = Literal["n", "i", "b"]


def marker_in(text: str) -> str | None:
    match: Final = MARKER.search(text)
    return None if match is None else match.group(0)


def flag_of(marker: str) -> Flag:
    last: Final = marker[-1]
    assert last in ("n", "i", "b"), marker
    return "n" if last == "n" else "i" if last == "i" else "b"


def collect_text(value: object) -> list[str]:
    """Every prompt string, whatever the wire format nests it in: `text` and `content` parts, the
    Responses API's bare `input`, and `system` prompts."""
    if isinstance(value, dict):
        items: Final = cast(dict[object, object], value).items()
        return [
            part
            for key, item in items
            for part in (
                [item]
                if key in ("text", "content", "input", "system") and isinstance(item, str)
                else collect_text(item)
            )
        ]  # comprehension-ok: flattening one nesting level of the request body
    if isinstance(value, list):
        return [part for item in cast(list[object], value) for part in collect_text(item)]  # comprehension-ok: flatten
    return []


def token_count(text: str) -> int:
    return max(1, len(text) // 4)


def str_field(body: object, key: str, default: str) -> str:
    if isinstance(body, dict):
        value: object = cast(dict[object, object], body).get(key)
        if isinstance(value, str):
            return value
    return default


def bool_field(body: object, key: str) -> bool:
    if isinstance(body, dict):
        value: object = cast(dict[object, object], body).get(key)
        return value is True
    return False


class MockRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    provider: Provider
    marker: str | None
    route: str
    stream: bool
    output_tokens: int
    text_sha1: str
    delay_s: float


class MockRequestsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: list[MockRequest]


class ResetResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    cleared: int


class Latency(BaseModel):
    """Seconds a mock waits before its first byte: `seconds` plus or minus up to `jitter_seconds`, drawn
    from the request marker so the same request always waits the same."""

    model_config = ConfigDict(frozen=True)

    seconds: float = Field(default=0.0, ge=0.0)
    jitter_seconds: float = Field(default=0.0, ge=0.0)

    def for_seed(self, seed: str) -> float:
        return max(0.0, self.seconds + self.jitter_seconds * (2 * unit(f"{seed}:latency") - 1))


@dataclass(frozen=True, slots=True)
class Decision:
    """What this mock does with one request: the reply it generates, on which route, after how long."""

    marker: str | None
    reply: Reply
    route: Route
    delay_s: float
    """Seconds before the first byte, what `gateway_added_seconds` subtracts from the client's time."""

    @property
    def seed(self) -> str:
        return self.marker or "unmarked"

    @property
    def headers(self) -> dict[str, str]:
        return {DELAY_HEADER: f"{self.delay_s:.4f}"}


class RequestLog:
    """What one mock pod received, in order, readable by the test at GET /mock/requests."""

    def __init__(self, provider: Provider, capacity: int = 100_000) -> None:
        self._provider: Final = provider
        self._entries: deque[MockRequest] = deque(maxlen=capacity)
        self._latency = Latency()

    @property
    def latency(self) -> Latency:
        return self._latency

    def set_latency(self, latency: Latency) -> Latency:
        self._latency = latency
        return self._latency

    def decide(self, *, route: Route, stream: bool, body: object, model: str) -> Decision:
        prompt_text: Final = " ".join(collect_text(body))
        marker: Final = marker_in(prompt_text)
        seed: Final = marker or "unmarked"
        reply: Final = reply_for(
            provider=self._provider, body=body, seed=seed, model=model, prompt_tokens=token_count(prompt_text)
        )
        decision: Final = Decision(marker=marker, reply=reply, route=route, delay_s=self._latency.for_seed(seed))
        self._entries.append(
            MockRequest(
                provider=self._provider,
                marker=marker,
                route=route,
                stream=stream,
                output_tokens=reply.output_tokens,
                text_sha1=hashlib.sha1(reply.text.encode()).hexdigest(),
                delay_s=decision.delay_s,
            )
        )
        return decision

    def snapshot(self) -> list[MockRequest]:
        return list(self._entries)

    def reset(self) -> int:
        count: Final = len(self._entries)
        self._entries.clear()
        return count


async def json_reply(decision: Decision) -> Response:
    """The generated body after the case's latency."""
    if decision.delay_s > 0:
        await asyncio.sleep(decision.delay_s)
    return Response(
        content=render_json(decision.reply, decision.route),
        media_type="application/json",
        headers=decision.headers,
    )


def stream_reply(decision: Decision, media_type: str) -> Response:
    """The generated chunks back to back after the case's latency."""
    return StreamingResponse(
        chunks_after(render_chunks(decision.reply, decision.route), decision.delay_s),
        media_type=media_type,
        headers=decision.headers,
    )


def parse_json(raw: bytes) -> object:
    parsed: object = json.loads(raw) if raw else {}
    return parsed


def mount_control(app: FastAPI, log: RequestLog) -> None:
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    async def requests_seen() -> MockRequestsResponse:
        return MockRequestsResponse(requests=log.snapshot())

    async def reset() -> ResetResponse:
        return ResetResponse(cleared=log.reset())

    async def latency(request: Latency) -> Latency:
        return log.set_latency(request)

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/mock/requests", requests_seen, methods=["GET"])
    app.add_api_route("/mock/reset", reset, methods=["POST"])
    app.add_api_route("/mock/latency", latency, methods=["POST"])
