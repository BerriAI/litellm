"""Shared behaviour of the perf mock providers.

Every knob a deployment needs is encoded in its model id, so one mock pod per provider serves any
number of deployments and a `PerformanceTestCase` controls them without redeploying anything:
`perf-l600-j300-f30-s529-t40` reads as 600 ms base latency, up to 300 ms of jitter, the first
attempt of 30% of requests fails with status 529, and streams emit 40 tokens per second. The reply,
the delay and the flake decision are pure functions of the request marker (`perfreq-000123-n`, put
in the prompt by the plan generator), so the test knows what the gateway should carry back and can
join what each provider received against what each client saw.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from typing import Final, Literal, cast

from fastapi import FastAPI
from pydantic import BaseModel

MARKER: Final = re.compile(r"perfreq-\d{6}-[nib]")
_KNOBS: Final = re.compile(r"perf((?:-[ljfst]\d+)+)")
DELAY_HEADER: Final = "x-mock-delay-s"
Flag = Literal["n", "i", "b"]


@dataclass(frozen=True, slots=True)
class Knobs:
    latency_s: float = 0.0
    jitter_s: float = 0.0
    flake_percent: int = 0
    flake_status: int = 500
    tokens_per_second: float = 50.0

    @property
    def model_suffix(self) -> str:
        """The segment a deployment's model id carries so this mock applies these knobs."""
        parts: Final = [
            f"l{round(self.latency_s * 1000)}",
            f"j{round(self.jitter_s * 1000)}",
            f"f{self.flake_percent}",
            f"s{self.flake_status}",
            f"t{round(self.tokens_per_second)}",
        ]
        return "perf-" + "-".join(parts)


def knobs_from_model(model: str) -> Knobs:
    match: Final = _KNOBS.search(model)
    if match is None:
        return Knobs()
    values: dict[str, int] = {}
    for segment in match.group(1).strip("-").split("-"):
        values[segment[0]] = int(segment[1:])
    return Knobs(
        latency_s=values.get("l", 0) / 1000,
        jitter_s=values.get("j", 0) / 1000,
        flake_percent=values.get("f", 0),
        flake_status=values.get("s", 500),
        tokens_per_second=float(values.get("t", 50)),
    )


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


def reply_for(marker: str | None) -> str:
    return f"ack {marker}" if marker else "ack unmarked"


def expected_delay_s(knobs: Knobs, marker: str) -> float:
    return knobs.latency_s + knobs.jitter_s * unit(f"{marker}:delay")


def collect_text(value: object) -> list[str]:
    """Every prompt string, whatever the wire format nests it in: `text` and `content` parts, the
    Responses API's bare `input`, and `system` prompts."""
    found: list[str] = []
    if isinstance(value, dict):
        for key, item in cast(dict[object, object], value).items():
            if key in ("text", "content", "input", "system") and isinstance(item, str):
                found.append(item)
            else:
                found.extend(collect_text(item))
    elif isinstance(value, list):
        for item in cast(list[object], value):
            found.extend(collect_text(item))
    return found


def token_count(text: str) -> int:
    return max(1, len(text) // 4)


class MockRequest(BaseModel):
    marker: str | None
    model: str
    route: str
    stream: bool
    attempt: int
    failed: bool
    status: int
    delay_s: float
    prompt_text: str
    received_at: float


class MockRequestsResponse(BaseModel):
    requests: list[MockRequest]


class ResetResponse(BaseModel):
    cleared: int


@dataclass(frozen=True, slots=True)
class Decision:
    """What this mock does with one request: how long it sleeps, whether it fails, what it says."""

    marker: str | None
    knobs: Knobs
    delay_s: float
    fail: bool
    reply: str
    prompt_text: str
    prompt_tokens: int
    completion_tokens: int

    @property
    def status(self) -> int:
        return self.knobs.flake_status if self.fail else 200

    @property
    def headers(self) -> dict[str, str]:
        return {DELAY_HEADER: f"{self.delay_s:.4f}"}

    def words(self) -> list[str]:
        """The reply split the way a stream emits it, one word (plus its space) per chunk."""
        parts: Final = self.reply.split(" ")
        return [part if index == len(parts) - 1 else part + " " for index, part in enumerate(parts)]

    @property
    def seconds_per_token(self) -> float:
        return 1.0 / self.knobs.tokens_per_second


class RequestLog:
    """What one mock pod received, in order, readable by the test at GET /mock/requests."""

    def __init__(self, capacity: int = 100_000) -> None:
        self._entries: deque[MockRequest] = deque(maxlen=capacity)
        self._attempts: dict[tuple[str, str], int] = {}

    def decide(self, *, model: str, route: str, stream: bool, body: object) -> Decision:
        prompt_text: Final = " ".join(collect_text(body))
        marker: Final = marker_in(prompt_text)
        knobs: Final = knobs_from_model(model)
        attempt_key: Final = (marker or "unmarked", model)
        attempt: Final = self._attempts.get(attempt_key, 0) + 1
        self._attempts[attempt_key] = attempt
        fail: Final = (
            marker is not None
            and attempt == 1
            and knobs.flake_percent > 0
            and int(unit(f"{marker}:flake") * 100) < knobs.flake_percent
        )
        delay_s: Final = 0.0 if fail else (expected_delay_s(knobs, marker) if marker else knobs.latency_s)
        reply: Final = reply_for(marker)
        decision: Final = Decision(
            marker=marker,
            knobs=knobs,
            delay_s=delay_s,
            fail=fail,
            reply=reply,
            prompt_text=prompt_text[:500],
            prompt_tokens=token_count(prompt_text),
            completion_tokens=token_count(reply),
        )
        self._entries.append(
            MockRequest(
                marker=marker,
                model=model,
                route=route,
                stream=stream,
                attempt=attempt,
                failed=fail,
                status=decision.status,
                delay_s=delay_s,
                prompt_text=decision.prompt_text,
                received_at=time.time(),
            )
        )
        return decision

    def snapshot(self) -> list[MockRequest]:
        return list(self._entries)

    def reset(self) -> int:
        count: Final = len(self._entries)
        self._entries.clear()
        self._attempts.clear()
        return count


async def sleep_for(decision: Decision) -> None:
    if decision.delay_s > 0:
        await asyncio.sleep(decision.delay_s)


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

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/mock/requests", requests_seen, methods=["GET"])
    app.add_api_route("/mock/reset", reset, methods=["POST"])


def sse(event: str | None, data: dict[str, object]) -> bytes:
    """One server-sent event; `event` is omitted for the OpenAI chat format, which only sends data."""
    line: Final = f"data: {json.dumps(data)}\n\n"
    return (f"event: {event}\n{line}" if event else line).encode()
