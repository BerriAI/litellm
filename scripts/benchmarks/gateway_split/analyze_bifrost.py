import statistics
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Final, Literal

from bifrost_bench import FrozenModel, Stats


class Record(FrozenModel):
    generator: Literal["bifrost"]
    variant: str
    prompt_bytes: int
    body_bytes: int
    stream: bool
    concurrency: int
    repetition: int
    python_body: str
    python_client: str
    rust_client: str
    event_loop: str
    profile: str
    http_parser: str
    gateway_cpu_s: float
    gateway_cpu_cores: float
    gateway_peak_rss_bytes: int
    loadgen_cpu_cores: float
    stats: Stats
    backend_completed: int
    backend_requests: int
    backend_cancelled: int


class Range(FrozenModel):
    median: float
    minimum: float
    maximum: float


class Summary(FrozenModel):
    variant: str
    prompt_bytes: int
    body_bytes: int
    stream: bool
    concurrency: int
    python_body: str
    python_client: str
    rust_client: str
    event_loop: str
    profile: str
    http_parser: str
    repetitions: int
    requests: int
    rps: Range
    p50_ms: Range
    p95_ms: Range
    p99_ms: Range
    first_event_p50_ms: Range
    first_event_p99_ms: Range
    gateway_cpu_cores: Range
    gateway_cpu_us_per_request: Range
    gateway_peak_rss_mib: Range
    loadgen_cpu_cores: Range


def distribution(values: Iterable[float]) -> Range:
    samples: Final = tuple(values)
    return Range(median=statistics.median(samples), minimum=min(samples), maximum=max(samples))


def records(paths: tuple[Path, ...]) -> Iterator[Record]:
    for path in paths:
        for line in path.read_text().splitlines():
            yield validate_record(line)


def validate_record(line: str) -> Record:
    row: Final = Record.model_validate_json(line)
    assert row.stats.status_code_counts == {"200": row.stats.requests}
    assert row.backend_requests == row.stats.requests
    assert row.backend_cancelled == 0
    assert not row.stats.drop_reasons
    if row.stream:
        assert row.backend_completed == row.stats.requests
    return row


def key(row: Record) -> tuple[str, int, bool, int, str]:
    return (
        row.variant,
        row.prompt_bytes,
        row.stream,
        row.concurrency,
        row.profile,
    )


def summarize(rows: tuple[Record, ...], repetitions: int) -> Summary:
    assert tuple(sorted(row.repetition for row in rows)) == tuple(range(repetitions))
    first: Final = rows[0]
    return Summary(
        variant=first.variant,
        prompt_bytes=first.prompt_bytes,
        body_bytes=first.body_bytes,
        stream=first.stream,
        concurrency=first.concurrency,
        python_body=first.python_body,
        python_client=first.python_client,
        rust_client=first.rust_client,
        http_parser=first.http_parser,
        event_loop=first.event_loop,
        profile=first.profile,
        repetitions=len(rows),
        requests=sum(row.stats.requests for row in rows),
        rps=distribution(row.stats.throughput_rps for row in rows),
        p50_ms=distribution(row.stats.p50_latency_ms for row in rows),
        p95_ms=distribution(row.stats.p95_latency_ms for row in rows),
        p99_ms=distribution(row.stats.p99_latency_ms for row in rows),
        first_event_p50_ms=distribution(row.stats.first_event_p50_ms for row in rows),
        first_event_p99_ms=distribution(row.stats.first_event_p99_ms for row in rows),
        gateway_cpu_cores=distribution(row.gateway_cpu_cores for row in rows),
        gateway_cpu_us_per_request=distribution(row.gateway_cpu_s * 1_000_000 / row.stats.requests for row in rows),
        gateway_peak_rss_mib=distribution(row.gateway_peak_rss_bytes / 1024**2 for row in rows),
        loadgen_cpu_cores=distribution(row.loadgen_cpu_cores for row in rows),
    )
