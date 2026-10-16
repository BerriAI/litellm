"""Turns what a run left behind (the client records, what each mock received, the guardrail's
calls, the workers' memory) into the case's metrics, and the case's limits into a verdict."""

from __future__ import annotations

import hashlib
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import chain
from typing import Final

from e2e_http import NoBody, get_external, post_external, unwrap
from memory_readings import RssCapture, RssReading, WorkerKey
from pydantic import BaseModel

from setup.datastore import DbCounters, RedisCounters
from setup.mocks.reply_common import Provider
from setup.performance_test_case import Limit, Metric, MockLatency, PerformanceTestCase
from setup.wire import WireRecord


class MockRequestSeen(BaseModel):
    provider: Provider
    marker: str | None
    route: str
    stream: bool
    output_tokens: int
    text_sha1: str
    delay_s: float


class MockRequestsSeen(BaseModel):
    requests: list[MockRequestSeen]


class GuardrailCallSeen(BaseModel):
    marker: str | None
    input_type: str
    action: str


class GuardrailCallsSeen(BaseModel):
    calls: list[GuardrailCallSeen]


class Cleared(BaseModel):
    cleared: int


class LatencyBody(BaseModel):
    seconds: float
    jitter_seconds: float


def read_mock_requests(url: str) -> list[MockRequestSeen]:
    return unwrap(get_external(f"{url}/mock/requests", response_type=MockRequestsSeen, timeout=60.0)).requests


def read_guardrail_calls(url: str) -> list[GuardrailCallSeen]:
    return unwrap(get_external(f"{url}/mock/calls", response_type=GuardrailCallsSeen, timeout=60.0)).calls


def reset_mock(url: str) -> int:
    return unwrap(post_external(f"{url}/mock/reset", json=NoBody(), response_type=Cleared)).cleared


def set_mock_latency(url: str, latency: MockLatency) -> LatencyBody:
    body: Final = LatencyBody(seconds=latency.seconds, jitter_seconds=latency.jitter_seconds)
    return unwrap(post_external(f"{url}/mock/latency", json=body, response_type=LatencyBody))


def percentile(values: Iterable[float], p: int) -> float | None:
    ordered: Final = sorted(values)
    if not ordered:
        return None
    index: Final = min(len(ordered) - 1, max(0, round(p / 100 * (len(ordered) - 1))))
    return ordered[index]


def worst_worker_growth_mb(before: RssCapture, after: RssCapture) -> float | None:
    """Worst RSS growth across workers. A worker missing from `before` (a pod replaced during the
    run) is measured against the median of the old workers."""
    old: Final[dict[WorkerKey, RssReading]] = {reading.worker: reading for reading in before.readings}
    if not old or not after.readings:
        return None
    median_before: Final = statistics.median(reading.ram_usage_mb for reading in old.values())
    growth: list[float] = []
    for reading in after.readings:
        baseline = old[reading.worker].ram_usage_mb if reading.worker in old else median_before
        growth.append(reading.ram_usage_mb - baseline)
    return max(growth)


@dataclass(frozen=True, slots=True)
class Observation:
    records: tuple[WireRecord, ...]
    mock_requests: tuple[MockRequestSeen, ...]
    guardrail_calls: tuple[GuardrailCallSeen, ...]
    rss_before: RssCapture
    rss_after: RssCapture
    wall_seconds: float
    spend_logged_share: float | None = None
    db_before: DbCounters | None = None
    db_after: DbCounters | None = None
    redis_before: RedisCounters | None = None
    redis_after: RedisCounters | None = None
    gateway_cpu_seconds: float | None = None

    def per_request(self, before: int | None, after: int | None) -> float | None:
        if before is None or after is None or not self.records:
            return None
        return (after - before) / len(self.records)

    @property
    def retried_markers(self) -> frozenset[str]:
        counts: Final = Counter(seen.marker for seen in self.mock_requests if seen.marker is not None)
        return frozenset(marker for marker, count in counts.items() if count > 1)

    @property
    def served(self) -> tuple[WireRecord, ...]:
        return tuple(record for record in self.records if record.ok and not record.blocked)

    @property
    def _delay_by_marker(self) -> dict[str, float]:
        return {seen.marker: seen.delay_s for seen in self.mock_requests if seen.marker is not None}

    def _clean(self, *, stream: bool) -> list[tuple[WireRecord, float]]:
        """Served requests of one kind that were answered on the first attempt, with the mock's delay."""
        delays: Final = self._delay_by_marker
        retried: Final = self.retried_markers
        pairs: list[tuple[WireRecord, float]] = []
        for record in self.served:
            delay = record.mock_delay_s if record.mock_delay_s is not None else delays.get(record.marker)
            if record.stream == stream and delay is not None and record.marker not in retried:
                pairs.append((record, delay))
        return pairs

    def unexpected_status_count(self) -> int:
        """Requests the plan expected to succeed that came back with any other status (0 is a client timeout)."""
        return sum(1 for record in self.records if not record.blocked and record.status != 200)

    def gateway_added_seconds(self) -> list[float]:
        """Seconds the gateway added to each first-attempt request: total time minus the mock's own delay for a
        non-streamed request, time to the first token minus the mock's delay to its first byte for a stream."""
        non_stream: Final = [record.total_s - delay for record, delay in self._clean(stream=False)]
        stream: Final = [
            record.first_token_s - delay
            for record, delay in self._clean(stream=True)
            if record.first_token_s is not None
        ]
        return non_stream + stream

    def time_to_first_token_seconds(self) -> list[float]:
        """Seconds from the client sending each first-attempt streamed request to its first token, mock delay
        included, comparable across versions because the mocks generate the same replies after the same wait."""
        return [record.first_token_s for record, _ in self._clean(stream=True) if record.first_token_s is not None]

    def bodies_off_reply(self) -> int:
        """Served requests whose text is not the reply the mock generated for them (plus the post-call rewrite
        for requests in the intervene slice), a translation or streaming fault somewhere in the gateway."""
        seen_by_marker: Final = {seen.marker: seen for seen in self.mock_requests if seen.marker is not None}
        return sum(
            1
            for record in self.served
            if (seen := seen_by_marker.get(record.marker)) is not None
            and _text_sha1(record.text, intervened=record.marker.endswith("-i")) != seen.text_sha1
        )

    def guardrail_missed(self) -> int:
        """Served requests that did not get exactly one pre-call and one post-call guardrail call."""
        pre: Final = Counter(call.marker for call in self.guardrail_calls if call.input_type == "request")
        post: Final = Counter(call.marker for call in self.guardrail_calls if call.input_type == "response")
        return sum(1 for record in self.served if pre[record.marker] != 1 or post[record.marker] != 1)


def _text_sha1(text: str, *, intervened: bool) -> str:
    original: Final = text.removesuffix(" [post]") if intervened else text
    return hashlib.sha1(original.encode()).hexdigest()


def observe(metric: Metric, percentile_rank: int | None, observation: Observation) -> float | None:
    """The observed value of one metric, at a percentile when the metric is per request."""
    per_request: dict[str, list[float]] = {
        "gateway_added_seconds": observation.gateway_added_seconds(),
        "time_to_first_token_seconds": observation.time_to_first_token_seconds(),
    }
    if metric in per_request:
        return percentile(per_request[metric], 50 if percentile_rank is None else percentile_rank)
    if metric == "unexpected_status_count":
        return float(observation.unexpected_status_count())
    if metric == "postgres_calls_per_request":
        db_before, db_after = observation.db_before, observation.db_after
        return observation.per_request(db_before and db_before.commits, db_after and db_after.commits)
    if metric == "redis_calls_per_request":
        redis_before, redis_after = observation.redis_before, observation.redis_after
        return observation.per_request(redis_before and redis_before.commands, redis_after and redis_after.commands)
    if metric == "cpu_seconds_per_request":
        cpu: Final = observation.gateway_cpu_seconds
        return None if cpu is None or not observation.records else cpu / len(observation.records)
    return worst_worker_growth_mb(observation.rss_before, observation.rss_after)


@dataclass(frozen=True, slots=True)
class Row:
    metric: Metric
    percentile: int | None
    observed: float | None
    limit: float | None

    @property
    def label(self) -> str:
        return f"{self.metric} percentile {self.percentile}" if self.percentile is not None else self.metric

    @property
    def property_name(self) -> str:
        return f"perf_{self.metric}" if self.percentile is None else f"perf_{self.metric}_p{self.percentile}"

    @property
    def over(self) -> bool:
        return self.limit is not None and self.observed is not None and self.observed > self.limit

    @property
    def missing(self) -> bool:
        return self.limit is not None and self.observed is None


REPORTED: Final[tuple[tuple[Metric, int | None], ...]] = (
    ("unexpected_status_count", None),
    ("gateway_added_seconds", 50),
    ("gateway_added_seconds", 99),
    ("time_to_first_token_seconds", 50),
    ("time_to_first_token_seconds", 99),
    ("memory_growth_megabytes", None),
    ("postgres_calls_per_request", None),
    ("redis_calls_per_request", None),
    ("cpu_seconds_per_request", None),
)


def evaluate(case: PerformanceTestCase, observation: Observation) -> tuple[list[Row], list[str]]:
    """Every reported metric with its observed value and the case's limit if it has one, plus one line per
    limit that was exceeded or that the run could not observe."""
    limits: Final[dict[tuple[Metric, int | None], Limit]] = {
        (limit.metric, limit.percentile): limit for limit in case.limits
    }
    keys: list[tuple[Metric, int | None]] = list(REPORTED)
    keys.extend(key for key in limits if key not in keys)
    rows: list[Row] = []
    for metric, rank in keys:
        limit = limits.get((metric, rank))
        rows.append(
            Row(
                metric=metric,
                percentile=rank,
                observed=observe(metric, rank, observation),
                limit=limit.max if limit else None,
            )
        )
    exceeded: Final = [
        f"{row.label}: observed {row.observed:.4f} > limit {row.limit:.4f}"
        if row.over and row.observed is not None and row.limit is not None
        else f"{row.label}: limit set but nothing observed"
        for row in rows
        if row.over or row.missing
    ]
    return rows, exceeded


def _row_properties(row: Row) -> tuple[tuple[str, float], ...]:
    observed: Final = ((row.property_name, row.observed),) if row.observed is not None else ()
    limit: Final = ((f"{row.property_name}_limit", row.limit),) if row.limit is not None else ()
    return observed + limit


def junit_properties(rows: Iterable[Row]) -> tuple[tuple[str, float], ...]:
    return tuple(chain.from_iterable(map(_row_properties, rows)))


def render(case: PerformanceTestCase, observation: Observation, rows: list[Row]) -> str:
    by_endpoint: Final = Counter((record.endpoint, record.stream) for record in observation.records)
    failures: Final = [record for record in observation.records if not record.ok][:10]
    lines: list[str] = [
        f"perf case {case.name}: {len(observation.records)} requests, {len(observation.served)} served, "
        f"{sum(1 for r in observation.records if r.blocked and r.ok)} blocked as planned, "
        f"{sum(1 for r in observation.records if not r.ok)} wrong, wall {observation.wall_seconds:.1f}s",
        "  "
        + ", ".join(
            f"{endpoint}{' stream' if stream else ''}={count}"
            for (endpoint, stream), count in sorted(by_endpoint.items())
        ),
        f"  {'metric':32} {'observed':>12} {'limit':>12}",
    ]
    for row in rows:
        observed = "n/a" if row.observed is None else f"{row.observed:.4f}"
        limit = "" if row.limit is None else f"{row.limit:.4f}"
        flag = "  OVER" if row.over else "  MISSING" if row.missing else ""
        lines.append(f"  {row.label:32} {observed:>12} {limit:>12}{flag}")
    lines.extend(_diagnostics(observation))
    for record in failures:
        lines.append(
            f"  wrong: seq {record.seq} {record.caller} {record.endpoint}{' stream' if record.stream else ''} status {record.status}: {record.error}"
        )
    return "\n".join(lines)


def _diagnostics(observation: Observation) -> list[str]:
    """Functional side-channels printed for whoever reads a red run, never limits of a case."""
    count: Final = max(1, len(observation.records))
    share: Final = observation.spend_logged_share
    return [
        "  diagnostics (not limits):",
        f"    bodies off reply {observation.bodies_off_reply()}",
        f"    guardrail calls per request {len(observation.guardrail_calls) / count:.3f}",
        f"    guardrail missed {observation.guardrail_missed() if observation.guardrail_calls else 'n/a'}",
        f"    retried share {len(observation.retried_markers) / count:.4f}",
        f"    spend logged share {'n/a' if share is None else f'{share:.3f}'}",
        f"    redis connected clients {_clients(observation.redis_before)} -> {_clients(observation.redis_after)}",
    ]


def _clients(counters: RedisCounters | None) -> str:
    return "n/a" if counters is None else str(counters.connected_clients)
