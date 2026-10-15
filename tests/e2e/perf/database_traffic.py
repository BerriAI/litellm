from __future__ import annotations

import time
from collections.abc import Iterator
from itertools import count
from threading import Event
from typing import Final, Literal

from e2e_http import Success, without_retries
from models import ChatBody, ChatMessage, ChatResponse
from pydantic import BaseModel, ConfigDict
from transport import Transport

from .database_queries import percentile

TrafficPhase = Literal["before", "stable-during", "target-during", "after"]


class TrafficSample(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: TrafficPhase
    started: float
    seconds: float
    request_id: str | None
    failure: str | None = None


class TrafficSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    phase: str
    requests: int
    errors: int
    error_rate: float
    p50_seconds: float
    p95_seconds: float
    p99_seconds: float
    longest_gap_seconds: float


def request(transport: Transport, key: str, model: str, marker: str, phase: TrafficPhase) -> TrafficSample:
    started: Final = time.time()
    with without_retries():
        result: Final = transport.post(
            "/v1/chat/completions",
            headers=transport.bearer(key),
            json=ChatBody(model=model, messages=(ChatMessage(role="user", content=marker),), max_tokens=8),
            response_type=ChatResponse,
            timeout=10,
        )
    seconds: Final = time.time() - started
    if not isinstance(result, Success):
        return TrafficSample(phase=phase, started=started, seconds=seconds, request_id=None, failure=result.kind)
    valid: Final = (
        result.data.id
        and result.data.choices
        and result.data.choices[0].message
        and result.data.choices[0].message.content
    )
    return TrafficSample(
        phase=phase,
        started=started,
        seconds=seconds,
        request_id=result.data.id,
        failure=None if valid else "Missing completion content or request ID",
    )


def traffic(
    transport: Transport,
    key: str,
    model: str,
    marker: str,
    phase: TrafficPhase,
    stop: Event,
    ready: Event,
) -> Iterator[TrafficSample]:
    for number in count():
        if stop.is_set():
            return
        yield (sample := request(transport, key, model, f"{marker}-{number}", phase))
        if number == 0:
            ready.set()
        stop.wait(max(0, 0.2 - sample.seconds))


def traffic_summary(phase: str, samples: tuple[TrafficSample, ...]) -> TrafficSummary:
    selected: Final = tuple(
        sorted((sample for sample in samples if sample.phase == phase), key=lambda row: row.started)
    )
    assert selected, f"No traffic samples for {phase}"
    seconds: Final = tuple(sample.seconds for sample in selected)
    gaps: Final = tuple(right.started - left.started for left, right in zip(selected, selected[1:]))
    errors: Final = sum(sample.failure is not None for sample in selected)
    return TrafficSummary(
        phase=phase,
        requests=len(selected),
        errors=errors,
        error_rate=errors / len(selected),
        p50_seconds=percentile(seconds, 0.5),
        p95_seconds=percentile(seconds, 0.95),
        p99_seconds=percentile(seconds, 0.99),
        longest_gap_seconds=max((*gaps, *seconds)),
    )
