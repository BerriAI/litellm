"""Turns what a run left behind (the client records, what each mock received, the guardrail's
calls, the workers' memory) into the case's metrics, and the case's limits into a verdict."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from itertools import chain, groupby
from types import MappingProxyType
from typing import Final, Literal

from e2e_http import NoBody, get_external, post_external, unwrap
from memory_readings import RssCapture, RssReading, WorkerKey
from pydantic import BaseModel
from typing_extensions import ReadOnly, TypedDict

from setup.datastore import (
    DbCounters,
    RedisCommand,
    RedisCounters,
    StatementCalls,
    active_sessions,
    calls_in,
    slowest_statement,
    statement_deltas,
)
from setup.mocks.reply_common import Provider
from setup.performance_test_case import Limit, Metric, MockLatency, PerformanceTestCase, Ramp
from setup.ramp import StepReading, render_steps, sustained_rps
from setup.series import Series
from setup.telemetry import GatewayTelemetry
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
    redis_commands: tuple[RedisCommand, ...] | None = None
    gateway_cpu_seconds: float | None = None
    telemetry_before: GatewayTelemetry | None = None
    telemetry_after: GatewayTelemetry | None = None
    peak_connections: int | None = None
    ramp: Ramp | None = None
    cpu_series: Series[float] | None = None
    connection_series: Series[int] | None = None
    telemetry_series: Series[GatewayTelemetry] | None = None

    @property
    def telemetry(self) -> GatewayTelemetry | None:
        """What the gateway's `/metrics` counted during the run."""
        if self.telemetry_before is None or self.telemetry_after is None:
            return None
        return self.telemetry_after.minus(self.telemetry_before)

    def per_request(self, before: int | None, after: int | None) -> float | None:
        if before is None or after is None or not self.records:
            return None
        return (after - before) / len(self.records)

    def count_per_request(self, count: int | None) -> float | None:
        return None if count is None or not self.records else count / len(self.records)

    @property
    def statements(self) -> tuple[StatementCalls, ...] | None:
        return statement_deltas(self.db_before, self.db_after)

    @property
    def busiest_awaited_statement(self) -> StatementCalls | None:
        """The awaited statement with the most calls: a lookup added to every request shows up here at
        1.0 per request while cache refreshes stay fractional, which the awaited total cannot tell apart."""
        awaited: Final = tuple(item for item in self.statements or () if item.bucket == "awaited")
        return max(awaited, key=_item_count, default=None)

    def busiest_background_statement(self, exempt_tables: frozenset[str]) -> StatementCalls | None:
        """The background statement with the most calls, skipping the case's per-flush exemptions: a rollup
        written once per request reads 1.0 here while batched writers stay below it."""
        background: Final = tuple(
            item for item in self.statements or () if item.bucket == "background" and item.table not in exempt_tables
        )
        return max(background, key=_item_count, default=None)

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

    def latency_pairs(self, metric: LatencyMetric) -> list[tuple[WireRecord, float]]:
        """Each first-attempt request with its value of one latency metric. `gateway_added_seconds` is the
        seconds the gateway added (total time minus the mock's own delay for a non-streamed request, time to the
        first token minus the mock's delay for a stream), `time_to_first_token_seconds` is the client's wait for
        the first token with the mock delay included, and the two `litellm_*` metrics are the gateway's own
        `x-litellm-overhead-duration-ms` and `x-litellm-response-duration-ms` headers in seconds."""
        if metric == "gateway_added_seconds":
            non_stream: Final = [(record, record.total_s - delay) for record, delay in self._clean(stream=False)]
            stream: Final = [
                (record, record.first_token_s - delay)
                for record, delay in self._clean(stream=True)
                if record.first_token_s is not None
            ]
            return non_stream + stream
        if metric == "time_to_first_token_seconds":
            return [(r, r.first_token_s) for r, _ in self._clean(stream=True) if r.first_token_s is not None]
        if metric == "litellm_overhead_seconds":
            return [
                (r, r.litellm_overhead_ms / 1000) for r in self._first_attempts() if r.litellm_overhead_ms is not None
            ]
        return [(r, r.litellm_response_ms / 1000) for r in self._first_attempts() if r.litellm_response_ms is not None]

    def _first_attempts(self) -> list[WireRecord]:
        return [record for record, _ in self._clean(stream=False)] + [record for record, _ in self._clean(stream=True)]

    def gateway_added_seconds(self) -> list[float]:
        return [value for _, value in self.latency_pairs("gateway_added_seconds")]

    def litellm_overhead_seconds(self) -> list[float]:
        return [value for _, value in self.latency_pairs("litellm_overhead_seconds")]

    def litellm_response_seconds(self) -> list[float]:
        return [value for _, value in self.latency_pairs("litellm_response_seconds")]

    def time_to_first_token_seconds(self) -> list[float]:
        return [value for _, value in self.latency_pairs("time_to_first_token_seconds")]

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

    def steps(self) -> tuple[StepReading, ...]:
        """One reading per step of a ramped run: the gateway's added latency over the step's requests, the rate
        at which requests completed inside the step's window, the step's unexpected statuses, and what the pod
        was doing in that window (CPU per completed request, Redis latency, Postgres connections)."""
        ramp: Final = self.ramp
        if ramp is None or not self.records:
            return ()
        started: Final = min(record.started_at for record in self.records)
        added: Final = _by_step(ramp, self.latency_pairs("gateway_added_seconds"))
        first_token: Final = _by_step(ramp, self.latency_pairs("time_to_first_token_seconds"))
        completed: Final = Counter(
            int((record.started_at + record.total_s - started) // ramp.step_seconds) for record in self.records
        )
        unexpected: Final = Counter(
            ramp.step_of(record.seq) for record in self.records if not record.blocked and record.status != 200
        )
        return tuple(
            StepReading(
                rps=rate,
                planned=count,
                completed_rps=completed[index] / ramp.step_seconds,
                gateway_added_p99=percentile(added.get(index, ()), 99),
                unexpected=unexpected[index],
                held_p99_seconds=ramp.held_p99_seconds,
                gateway_added_p50=percentile(added.get(index, ()), 50),
                time_to_first_token_p99=percentile(first_token.get(index, ()), 99),
                **self._window_readings(started + index * ramp.step_seconds, ramp.step_seconds, completed[index]),
            )
            for index, (rate, count) in enumerate(zip(ramp.rates, ramp.counts, strict=True))
        )

    def _window_readings(self, start: float, seconds: float, completed: int) -> WindowReadings:
        end: Final = start + seconds
        cpu: Final = None if self.cpu_series is None else sum(self.cpu_series.between(start, end))
        conns: Final = None if self.connection_series is None else self.connection_series.between(start, end)
        before: Final = None if self.telemetry_series is None else self.telemetry_series.last_before(start)
        after: Final = None if self.telemetry_series is None else self.telemetry_series.last_before(end)
        return WindowReadings(
            cpu_seconds_per_request=None if cpu is None or completed == 0 else cpu / completed,
            redis_latency_mean_seconds=None
            if before is None or after is None
            else after.minus(before).redis_latency.mean,
            postgres_peak_connections=max(conns, default=None) if conns else None,
        )


class WindowReadings(TypedDict):
    cpu_seconds_per_request: ReadOnly[float | None]
    redis_latency_mean_seconds: ReadOnly[float | None]
    postgres_peak_connections: ReadOnly[int | None]


def _by_step(ramp: Ramp, pairs: Iterable[tuple[WireRecord, float]]) -> Mapping[int, tuple[float, ...]]:
    ordered: Final = sorted(pairs, key=lambda pair: pair[0].seq)
    return MappingProxyType(
        {
            step: tuple(value for _, value in group)
            for step, group in groupby(ordered, key=lambda pair: ramp.step_of(pair[0].seq))
        }
    )


def _text_sha1(text: str, *, intervened: bool) -> str:
    original: Final = text.removesuffix(" [post]") if intervened else text
    return hashlib.sha1(original.encode()).hexdigest()


LatencyMetric = Literal[
    "gateway_added_seconds", "time_to_first_token_seconds", "litellm_overhead_seconds", "litellm_response_seconds"
]
TOP_CONTRIBUTORS: Final = 5


@dataclass(frozen=True, slots=True)
class Contributor:
    """One slice of a metric: a statement, a Redis command, an endpoint, a worker or a status, with its own
    value in the metric's unit and, where the metric is a total, its share of that total."""

    label: str
    value: float
    share: float | None = None


def _top(items: Iterable[Contributor]) -> tuple[Contributor, ...]:
    return tuple(sorted(items, key=lambda item: item.value, reverse=True)[:TOP_CONTRIBUTORS])


def _endpoint_label(record: WireRecord) -> str:
    return f"{record.endpoint}{' stream' if record.stream else ''}"


def _latency_breakdown(metric: LatencyMetric, rank: int, observation: Observation) -> tuple[Contributor, ...]:
    pairs: Final = observation.latency_pairs(metric)
    by_endpoint: dict[str, list[float]] = defaultdict(list)
    for record, value in pairs:
        by_endpoint[_endpoint_label(record)].append(value)
    return _top(
        Contributor(label, percentile(values, rank) or 0.0, len(values) / len(pairs))
        for label, values in by_endpoint.items()
    )


def _statements_for(
    metric: Metric, statements: tuple[StatementCalls, ...], exempt_tables: frozenset[str]
) -> tuple[StatementCalls, ...]:
    if metric in ("postgres_calls_per_request", "postgres_max_statement_per_request"):
        return tuple(item for item in statements if item.bucket == "awaited")
    if metric == "postgres_background_calls_per_request":
        return tuple(item for item in statements if item.bucket == "background")
    if metric == "postgres_max_background_statement_per_request":
        return tuple(item for item in statements if item.bucket == "background" and item.table not in exempt_tables)
    return tuple(item for item in statements if item.calls > 0)


def _statement_breakdown(
    metric: Metric, observation: Observation, exempt_tables: frozenset[str]
) -> tuple[Contributor, ...]:
    chosen: Final = _statements_for(metric, observation.statements or (), exempt_tables)
    requests: Final = max(1, len(observation.records))
    total_calls: Final = sum(item.calls for item in chosen)
    total_ms: Final = sum(item.total_exec_ms for item in chosen)
    if metric == "postgres_slowest_statement_seconds":
        return _top(
            Contributor(
                item.summary, item.mean_exec_seconds or 0.0, item.total_exec_ms / total_ms if total_ms else None
            )
            for item in chosen
        )
    if metric == "postgres_active_sessions":
        return _top(
            Contributor(
                item.summary, item.total_exec_ms / 1000 / observation.wall_seconds, item.total_exec_ms / total_ms
            )
            for item in chosen
            if total_ms
        )
    return _top(
        Contributor(item.summary, item.calls / requests, item.calls / total_calls) for item in chosen if total_calls
    )


def _redis_breakdown(metric: Metric, observation: Observation) -> tuple[Contributor, ...]:
    bucket: Final = "awaited" if metric == "redis_calls_per_request" else "background"
    chosen: Final = tuple(item for item in observation.redis_commands or () if item.bucket == bucket)
    requests: Final = max(1, len(observation.records))
    total: Final = sum(item.count for item in chosen)
    return _top(
        Contributor(f"{item.command} {item.key}".strip(), item.count / requests, item.count / total)
        for item in chosen
        if total
    )


def _memory_breakdown(observation: Observation) -> tuple[Contributor, ...]:
    before: Final = {reading.worker: reading.ram_usage_mb for reading in observation.rss_before.readings}
    return _top(
        Contributor(f"{reading.replica} pid {reading.worker_pid}", reading.ram_usage_mb - before[reading.worker])
        for reading in observation.rss_after.readings
        if reading.worker in before
    )


def _status_breakdown(observation: Observation) -> tuple[Contributor, ...]:
    unexpected: Final = [r for r in observation.records if not r.blocked and r.status != 200]
    counts: Final = Counter(f"{_endpoint_label(record)} status {record.status}" for record in unexpected)
    return _top(Contributor(label, float(count), count / len(unexpected)) for label, count in counts.items())


def breakdown(
    metric: Metric, percentile_rank: int | None, observation: Observation, exempt_tables: frozenset[str] = frozenset()
) -> tuple[Contributor, ...]:
    """The top contributors to one metric, so a spike on the chart names its statement, command, endpoint or
    worker. Gauges with no slices (`cpu_seconds_per_request`, `prometheus_redis_latency_mean_seconds`,
    `postgres_peak_connections`) have none."""
    if metric in (
        "gateway_added_seconds",
        "time_to_first_token_seconds",
        "litellm_overhead_seconds",
        "litellm_response_seconds",
    ):
        return _latency_breakdown(metric, 50 if percentile_rank is None else percentile_rank, observation)
    if metric.startswith("postgres_") and metric != "postgres_peak_connections":
        return _statement_breakdown(metric, observation, exempt_tables)
    if metric in {"redis_calls_per_request", "redis_background_calls_per_request"}:
        return _redis_breakdown(metric, observation)
    if metric == "memory_growth_megabytes":
        return _memory_breakdown(observation)
    if metric == "unexpected_status_count":
        return _status_breakdown(observation)
    return ()


def observe(
    metric: Metric, percentile_rank: int | None, observation: Observation, exempt_tables: frozenset[str] = frozenset()
) -> float | None:
    """The observed value of one metric, at a percentile when the metric is per request."""
    per_request: dict[str, list[float]] = {
        "gateway_added_seconds": observation.gateway_added_seconds(),
        "time_to_first_token_seconds": observation.time_to_first_token_seconds(),
        "litellm_overhead_seconds": observation.litellm_overhead_seconds(),
        "litellm_response_seconds": observation.litellm_response_seconds(),
    }
    if metric in per_request:
        return percentile(per_request[metric], 50 if percentile_rank is None else percentile_rank)
    if metric == "unexpected_status_count":
        return float(observation.unexpected_status_count())
    if metric == "prometheus_redis_latency_mean_seconds":
        counted: Final = observation.telemetry
        return None if counted is None else counted.redis_latency.mean
    if metric == "postgres_active_sessions":
        return active_sessions(observation.statements, observation.wall_seconds)
    if metric == "postgres_slowest_statement_seconds":
        slowest: Final = slowest_statement(observation.statements)
        return None if slowest is None else slowest.mean_exec_seconds
    if metric == "postgres_peak_connections":
        return None if observation.peak_connections is None else float(observation.peak_connections)
    if metric == "postgres_calls_per_request":
        return observation.count_per_request(calls_in("awaited", observation.statements))
    if metric == "postgres_background_calls_per_request":
        return observation.count_per_request(calls_in("background", observation.statements))
    if metric == "postgres_max_statement_per_request":
        if observation.statements is None:
            return None
        busiest: Final = observation.busiest_awaited_statement
        return observation.count_per_request(0 if busiest is None else busiest.calls)
    if metric == "postgres_max_background_statement_per_request":
        if observation.statements is None:
            return None
        busiest_background: Final = observation.busiest_background_statement(exempt_tables)
        return observation.count_per_request(0 if busiest_background is None else busiest_background.calls)
    if metric == "redis_calls_per_request":
        return observation.count_per_request(calls_in("awaited", observation.redis_commands))
    if metric == "redis_background_calls_per_request":
        return observation.count_per_request(calls_in("background", observation.redis_commands))
    if metric == "cpu_seconds_per_request":
        cpu: Final = observation.gateway_cpu_seconds
        return None if cpu is None or not observation.records else cpu / len(observation.records)
    if metric == "sustained_rps":
        return None if observation.ramp is None else sustained_rps(observation.steps())
    return worst_worker_growth_mb(observation.rss_before, observation.rss_after)


@dataclass(frozen=True, slots=True)
class Row:
    metric: Metric
    percentile: int | None
    observed: float | None
    limit: float | None
    floor: float | None = None
    detail: str = ""
    breakdown: tuple[Contributor, ...] = ()

    @property
    def label(self) -> str:
        return f"{self.metric} percentile {self.percentile}" if self.percentile is not None else self.metric

    @property
    def property_name(self) -> str:
        return f"perf_{self.metric}" if self.percentile is None else f"perf_{self.metric}_p{self.percentile}"

    @property
    def over(self) -> bool:
        """Past the limit: above a max, or below a floor."""
        if self.observed is None:
            return False
        return (self.limit is not None and self.observed > self.limit) or (
            self.floor is not None and self.observed < self.floor
        )

    @property
    def missing(self) -> bool:
        return (self.limit is not None or self.floor is not None) and self.observed is None

    @property
    def bound(self) -> str:
        if self.limit is not None:
            return f"{self.limit:.4f}"
        return "" if self.floor is None else f">= {self.floor:.4f}"


REPORTED: Final[tuple[tuple[Metric, int | None], ...]] = (
    ("unexpected_status_count", None),
    ("gateway_added_seconds", 50),
    ("gateway_added_seconds", 99),
    ("litellm_overhead_seconds", 50),
    ("litellm_overhead_seconds", 99),
    ("litellm_response_seconds", 50),
    ("litellm_response_seconds", 99),
    ("prometheus_redis_latency_mean_seconds", None),
    ("postgres_active_sessions", None),
    ("postgres_slowest_statement_seconds", None),
    ("postgres_peak_connections", None),
    ("time_to_first_token_seconds", 50),
    ("time_to_first_token_seconds", 99),
    ("memory_growth_megabytes", None),
    ("postgres_calls_per_request", None),
    ("postgres_background_calls_per_request", None),
    ("postgres_max_statement_per_request", None),
    ("postgres_max_background_statement_per_request", None),
    ("redis_calls_per_request", None),
    ("redis_background_calls_per_request", None),
    ("cpu_seconds_per_request", None),
)


def evaluate(case: PerformanceTestCase, observation: Observation) -> tuple[list[Row], list[str]]:
    """Every reported metric with its observed value and the case's limit if it has one, plus one line per
    limit that was exceeded or that the run could not observe."""
    limits: Final[dict[tuple[Metric, int | None], Limit]] = {
        (limit.metric, limit.percentile): limit for limit in case.limits
    }
    keys: list[tuple[Metric, int | None]] = list(REPORTED)
    if observation.ramp is not None:
        keys.append(("sustained_rps", None))
    keys.extend(key for key in limits if key not in keys)
    rows: list[Row] = []
    for metric, rank in keys:
        limit = limits.get((metric, rank))
        rows.append(
            Row(
                metric=metric,
                percentile=rank,
                observed=observe(metric, rank, observation, case.background_cap_exempt_tables),
                limit=limit.max if limit else None,
                floor=limit.min if limit else None,
                detail=_detail(metric, observation, case.background_cap_exempt_tables),
                breakdown=breakdown(metric, rank, observation, case.background_cap_exempt_tables),
            )
        )
    exceeded: Final = [_breach(row) for row in rows if row.over or row.missing]
    return rows, exceeded


def _breach(row: Row) -> str:
    if row.observed is None:
        return f"{row.label}: limit set but nothing observed"
    if row.floor is not None:
        return f"{row.label}: observed {row.observed:.4f} < floor {row.floor:.4f}"
    return f"{row.label}: observed {row.observed:.4f} > limit {row.bound}{f' ({row.detail})' if row.detail else ''}"


def _detail(metric: Metric, observation: Observation, exempt_tables: frozenset[str]) -> str:
    if metric == "postgres_max_statement_per_request":
        busiest: Final = observation.busiest_awaited_statement
        return "" if busiest is None else busiest.summary
    if metric == "postgres_max_background_statement_per_request":
        busiest_background: Final = observation.busiest_background_statement(exempt_tables)
        return "" if busiest_background is None else busiest_background.summary
    if metric == "postgres_slowest_statement_seconds":
        slowest: Final = slowest_statement(observation.statements)
        return "" if slowest is None else slowest.summary
    return ""


def _row_properties(row: Row) -> tuple[tuple[str, float | str], ...]:
    observed: Final = ((row.property_name, row.observed),) if row.observed is not None else ()
    limit: Final = ((f"{row.property_name}_limit", row.limit),) if row.limit is not None else ()
    floor: Final = ((f"{row.property_name}_floor", row.floor),) if row.floor is not None else ()
    contributors: Final = (
        ((f"{row.property_name}_breakdown", json.dumps([asdict(item) for item in row.breakdown])),)
        if row.breakdown
        else ()
    )
    return observed + limit + floor + contributors


def junit_properties(rows: Iterable[Row]) -> tuple[tuple[str, float | str], ...]:
    """One numeric property per observed metric and per limit, plus a `_breakdown` JSON list of the top
    contributors for every metric that has slices, all under the `perf_` prefix the dashboard ingests."""
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
        flag = (
            "  UNDER"
            if row.over and row.floor is not None
            else "  OVER"
            if row.over
            else "  MISSING"
            if row.missing
            else ""
        )
        lines.append(f"  {row.label:32} {observed:>12} {row.bound:>12}{flag}")
    lines.extend(render_steps(observation.steps()))
    lines.extend(_diagnostics(observation))
    lines.extend(_datastore_lines(observation))
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
        f"    mock routes {_mock_routes(observation)}",
        f"    guardrail calls per request {len(observation.guardrail_calls) / count:.3f}",
        f"    guardrail missed {observation.guardrail_missed() if observation.guardrail_calls else 'n/a'}",
        f"    retried share {len(observation.retried_markers) / count:.4f}",
        f"    spend logged share {'n/a' if share is None else f'{share:.3f}'}",
        f"    postgres transactions per request {_fmt(observation.per_request(_commits(observation.db_before), _commits(observation.db_after)))}",
        f"    redis commands per request {_fmt(observation.per_request(_commands(observation.redis_before), _commands(observation.redis_after)))}",
        f"    redis connected clients {_clients(observation.redis_before)} -> {_clients(observation.redis_after)}",
    ]


def _mock_routes(observation: Observation) -> str:
    """Which provider route each request landed on, so a direct-vs-bridge case shows its split."""
    counts: Final = Counter(f"{seen.provider}/{seen.route}" for seen in observation.mock_requests)
    return ", ".join(f"{route} {count}" for route, count in sorted(counts.items())) or "n/a"


def _datastore_lines(observation: Observation) -> list[str]:
    """What the per-request Postgres and Redis counts are made of, so a red run names the statement or
    command that grew. `plumbing` (BEGIN, COMMIT, session settings) and `unclassified` count in neither limit."""
    count: Final = max(1, len(observation.records))
    lines: list[str] = []
    for title, items in (
        ("postgres statements", observation.statements),
        ("redis commands", observation.redis_commands),
    ):
        if items is None:
            lines.append(f"  {title}: n/a")
            continue
        for bucket in ("awaited", "background", "plumbing", "unclassified"):
            in_bucket = sorted((item for item in items if item.bucket == bucket), key=_item_count, reverse=True)
            total = sum(_item_count(item) for item in in_bucket)
            if not in_bucket:
                continue
            lines.append(f"  {title} {bucket}: {total / count:.3f} per request, {len(in_bucket)} distinct")
            lines.extend(f"    {_item_count(item) / count:8.3f}  {item.summary}" for item in in_bucket[:8])
    return lines


def _item_count(item: StatementCalls | RedisCommand) -> int:
    return item.calls if isinstance(item, StatementCalls) else item.count


def _commits(counters: DbCounters | None) -> int | None:
    return None if counters is None else counters.commits


def _commands(counters: RedisCounters | None) -> int | None:
    return None if counters is None else counters.commands


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _clients(counters: RedisCounters | None) -> str:
    return "n/a" if counters is None else str(counters.connected_clients)
