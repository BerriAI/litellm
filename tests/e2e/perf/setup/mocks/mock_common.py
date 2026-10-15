"""Shared behaviour of the perf mock providers.

Every reply is a real provider response captured once (see `replay.py`), chosen by route, stream mode and
the size of the request just received, so the gateway parses the provider's own usage, stop reasons, tool
calls and stream framing, on the provider's own clock. The model id stays a real one (`gpt-6-luna`,
`anthropic.claude-sonnet-5`) so every name-based branch in the gateway runs, and one mock pod per provider
serves any number of deployments. How long a mock waits before answering is the one setting a case
controls, `POST /mock/latency {"seconds": 1.0, "jitter_seconds": 0.25}`, set by the test's setup directly
on the pod, never through the gateway; the captured provider timings are not replayed. The wait and the
replaced ids are pure functions of the request marker (`perfreq-000123-n`, put in the prompt by the plan
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

from replay import Capture, Provider, body_bytes, paced, pick

MARKER: Final = re.compile(r"perfreq-\d{6}-[nib]")
DELAY_HEADER: Final = "x-mock-delay-s"
Flag = Literal["n", "i", "b"]


def unit(seed: str) -> float:
    """A float in [0, 1) that is the same for the same seed on every mock pod."""
    digest: Final = hashlib.sha1(seed.encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


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
    shape: str
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
    """What this mock does with one request: which capture it replays and after how long."""

    marker: str | None
    capture: Capture
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

    def decide(self, *, route: str, stream: bool, body: object) -> Decision:
        prompt_text: Final = " ".join(collect_text(body))
        marker: Final = marker_in(prompt_text)
        prompt_tokens: Final = token_count(prompt_text)
        decision: Final = Decision(
            marker=marker,
            capture=pick(self._provider, route, stream, prompt_tokens),
            delay_s=self._latency.for_seed(marker or "unmarked"),
        )
        self._entries.append(
            MockRequest(
                provider=self._provider,
                marker=marker,
                route=route,
                stream=stream,
                shape=decision.capture.shape,
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
    """The captured body after the case's latency, ids swapped for this request's."""
    if decision.delay_s > 0:
        await asyncio.sleep(decision.delay_s)
    return Response(
        content=body_bytes(decision.capture, decision.seed),
        media_type="application/json",
        headers=decision.headers,
    )


def stream_reply(decision: Decision, media_type: str) -> Response:
    """The captured chunks back to back after the case's latency, ids swapped for this request's."""
    return StreamingResponse(
        paced(decision.capture, decision.seed, 0.0, decision.delay_s),
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
