"""Shared behaviour of the perf mock providers.

Every reply is a real provider response captured once (see `replay.py`), chosen by route, stream mode and
the size of the request just received, so the gateway parses the provider's own usage, stop reasons, tool
calls and stream framing, on the provider's own clock. A deployment's knobs travel in the `x-mock-knobs`
header the gateway forwards from the deployment's `extra_headers`, so the model id stays a real one
(`gpt-6-luna`, `anthropic.claude-sonnet-5`) and every name-based branch in the gateway runs, while one mock
pod per provider serves any number of deployments: `f30-s529-p25` reads as the first attempt of 30% of
requests fails with status 529, and the captured timings play at 25% (four times faster than the provider
did). The flake decision and the replaced ids are pure functions of the request marker
(`perfreq-000123-n`, put in the prompt by the plan generator), so the test can join what each provider
received against what each client saw.
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
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict

from replay import Capture, Provider, body_bytes, paced, pick

MARKER: Final = re.compile(r"perfreq-\d{6}-[nib]")
_KNOBS: Final = re.compile(r"^(?:[fsp]\d+)(?:-[fsp]\d+)*$")
DELAY_HEADER: Final = "x-mock-delay-s"
TOTAL_HEADER: Final = "x-mock-total-s"
SHAPE_HEADER: Final = "x-mock-shape"
KNOBS_HEADER: Final = "x-mock-knobs"
Flag = Literal["n", "i", "b"]


@dataclass(frozen=True, slots=True)
class Knobs:
    flake_percent: int = 0
    flake_status: int = 500
    speed_percent: int = 100

    @property
    def header_value(self) -> str:
        """What a deployment puts in its `x-mock-knobs` extra header so this mock applies these knobs."""
        return f"f{self.flake_percent}-s{self.flake_status}-p{self.speed_percent}"

    @property
    def scale(self) -> float:
        return self.speed_percent / 100


def knobs_from_header(value: str | None) -> Knobs:
    if value is None or _KNOBS.match(value) is None:
        return Knobs()
    values: Final = {segment[0]: int(segment[1:]) for segment in value.split("-")}
    return Knobs(
        flake_percent=values.get("f", 0),
        flake_status=values.get("s", 500),
        speed_percent=values.get("p", 100),
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
    model: str
    route: str
    stream: bool
    shape: str
    attempt: int
    failed: bool
    status: int
    delay_s: float
    total_s: float
    prompt_tokens: int
    prompt_text: str
    received_at: float


class MockRequestsResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests: list[MockRequest]


class ResetResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    cleared: int


@dataclass(frozen=True, slots=True)
class Decision:
    """What this mock does with one request: whether it fails, which capture it replays and how fast."""

    marker: str | None
    knobs: Knobs
    capture: Capture
    fail: bool
    prompt_text: str
    prompt_tokens: int

    @property
    def seed(self) -> str:
        return self.marker or "unmarked"

    @property
    def status(self) -> int:
        return self.knobs.flake_status if self.fail else 200

    @property
    def delay_s(self) -> float:
        """Seconds before the first byte, what `added_latency_s` subtracts from the client's time."""
        return 0.0 if self.fail else self.capture.ttfb_s * self.knobs.scale

    @property
    def total_s(self) -> float:
        return 0.0 if self.fail else self.capture.total_s * self.knobs.scale

    @property
    def headers(self) -> dict[str, str]:
        return {
            DELAY_HEADER: f"{self.delay_s:.4f}",
            TOTAL_HEADER: f"{self.total_s:.4f}",
            SHAPE_HEADER: self.capture.shape,
        }


class RequestLog:
    """What one mock pod received, in order, readable by the test at GET /mock/requests."""

    def __init__(self, provider: Provider, capacity: int = 100_000) -> None:
        self._provider: Final = provider
        self._entries: deque[MockRequest] = deque(maxlen=capacity)
        self._attempts: dict[tuple[str, str], int] = {}

    def decide(self, *, model: str, knobs: Knobs, route: str, stream: bool, body: object) -> Decision:
        prompt_text: Final = " ".join(collect_text(body))
        marker: Final = marker_in(prompt_text)
        prompt_tokens: Final = token_count(prompt_text)
        attempt_key: Final = (marker or "unmarked", model)
        attempt: Final = self._attempts.get(attempt_key, 0) + 1
        self._attempts[attempt_key] = attempt
        fail: Final = (
            marker is not None
            and attempt == 1
            and knobs.flake_percent > 0
            and int(unit(f"{marker}:flake") * 100) < knobs.flake_percent
        )
        decision: Final = Decision(
            marker=marker,
            knobs=knobs,
            capture=pick(self._provider, route, stream, prompt_tokens),
            fail=fail,
            prompt_text=prompt_text[:500],
            prompt_tokens=prompt_tokens,
        )
        self._entries.append(
            MockRequest(
                provider=self._provider,
                marker=marker,
                model=model,
                route=route,
                stream=stream,
                shape=decision.capture.shape,
                attempt=attempt,
                failed=fail,
                status=decision.status,
                delay_s=decision.delay_s,
                total_s=decision.total_s,
                prompt_tokens=prompt_tokens,
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


async def json_reply(decision: Decision) -> Response:
    """The captured body after the captured time to first byte, ids swapped for this request's."""
    if decision.delay_s > 0:
        await asyncio.sleep(decision.delay_s)
    return Response(
        content=body_bytes(decision.capture, decision.seed),
        media_type="application/json",
        headers=decision.headers,
    )


def stream_reply(decision: Decision, media_type: str) -> Response:
    """The captured chunks on the captured clock, ids swapped for this request's."""
    return StreamingResponse(
        paced(decision.capture, decision.seed, decision.knobs.scale),
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

    app.add_api_route("/health", health, methods=["GET"])
    app.add_api_route("/mock/requests", requests_seen, methods=["GET"])
    app.add_api_route("/mock/reset", reset, methods=["POST"])
