from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Final

from pydantic import BaseModel, TypeAdapter

_LOCUSTFILE = Path(__file__).with_name("locustfile.py")
_CSV_PREFIX = "locust"
_GENERATOR_SATURATION_MARKER = "CPU usage above"
_MAX_REPORTED_ERRORS = 5


class LocustStatEntry(BaseModel):
    name: str
    num_requests: int
    num_failures: int
    start_time: float
    last_request_timestamp: float
    response_times: dict[int, int]


_STATS_ADAPTER: TypeAdapter[list[LocustStatEntry]] = TypeAdapter(list[LocustStatEntry])


@dataclass(frozen=True, slots=True)
class LoadError:
    name: str
    error: str
    occurrences: int


@dataclass(frozen=True, slots=True)
class EndpointLoad:
    """One route's share of a phase, so a run that silently drove only one of them is visible."""

    name: str
    requests: int
    failures: int
    p50_seconds: float


@dataclass(frozen=True, slots=True)
class LoadResult:
    requests: int
    failures: int
    requests_per_second: float
    p50_seconds: float
    p90_seconds: float
    p99_seconds: float
    endpoints: tuple[EndpointLoad, ...]
    errors: tuple[LoadError, ...]
    generator_warnings: tuple[str, ...]

    @property
    def failure_ratio(self) -> float:
        return self.failures / self.requests if self.requests else 1.0

    def diagnosis(self) -> str:
        """What the failed requests actually got, so a red run reads without log archaeology."""
        ranked = sorted(self.errors, key=lambda error: error.occurrences, reverse=True)
        lines = [f"{error.occurrences}x {error.name}: {error.error}" for error in ranked[:_MAX_REPORTED_ERRORS]]
        remainder = len(ranked) - len(lines)
        if remainder > 0:
            lines.append(f"and {remainder} more distinct errors")
        if not lines:
            lines.append("locust recorded no error breakdown")
        return "; ".join((*lines, *self.generator_warnings))

    def latency_summary(self) -> str:
        return f"p50 {self.p50_seconds:.3f}s, p90 {self.p90_seconds:.3f}s, p99 {self.p99_seconds:.3f}s"

    def endpoint_summary(self) -> str:
        return ", ".join(
            f"{endpoint.name} {endpoint.requests} requests, {endpoint.failures} failures, "
            f"p50 {endpoint.p50_seconds:.3f}s"
            for endpoint in self.endpoints
        )


def percentile_seconds(entries: Sequence[LocustStatEntry], fraction: float) -> float:
    """The response time at `fraction` of the merged histograms, in seconds.

    Locust buckets response times by millisecond, so this reads the first bucket whose
    running count reaches the rank, the same lower-sample convention locust's own
    percentiles use.
    """
    samples = sorted((milliseconds, count) for entry in entries for milliseconds, count in entry.response_times.items())
    total = sum(count for _, count in samples)
    if total == 0:
        return 0.0
    running = accumulate(count for _, count in samples)
    rank: Final = total * fraction
    return next(milliseconds for (milliseconds, _), seen in zip(samples, running) if seen >= rank) / 1000.0


def per_endpoint(entries: Sequence[LocustStatEntry]) -> tuple[EndpointLoad, ...]:
    """Each locust request name's own totals, in the order the names first appear."""
    names: Final = tuple(dict.fromkeys(entry.name for entry in entries))
    grouped: Final = ((name, tuple(entry for entry in entries if entry.name == name)) for name in names)
    return tuple(
        EndpointLoad(
            name=name,
            requests=sum(entry.num_requests for entry in group),
            failures=sum(entry.num_failures for entry in group),
            p50_seconds=percentile_seconds(group, 0.5),
        )
        for name, group in grouped
    )


def aggregate_stats(
    entries: Sequence[LocustStatEntry],
    errors: tuple[LoadError, ...],
    generator_warnings: tuple[str, ...],
) -> LoadResult:
    requests = sum(entry.num_requests for entry in entries)
    failures = sum(entry.num_failures for entry in entries)
    endpoints = per_endpoint(entries)
    if not entries or requests == 0:
        return LoadResult(
            requests=requests,
            failures=failures,
            requests_per_second=0.0,
            p50_seconds=0.0,
            p90_seconds=0.0,
            p99_seconds=0.0,
            endpoints=endpoints,
            errors=errors,
            generator_warnings=generator_warnings,
        )
    elapsed = max(entry.last_request_timestamp for entry in entries) - min(entry.start_time for entry in entries)
    return LoadResult(
        requests=requests,
        failures=failures,
        requests_per_second=requests / elapsed if elapsed > 0 else 0.0,
        p50_seconds=percentile_seconds(entries, 0.5),
        p90_seconds=percentile_seconds(entries, 0.9),
        p99_seconds=percentile_seconds(entries, 0.99),
        endpoints=endpoints,
        errors=errors,
        generator_warnings=generator_warnings,
    )


def read_errors(failures_csv: Path) -> tuple[LoadError, ...]:
    """Locust's per-error breakdown, which its --json summary omits entirely.

    Written on a one-second tick, so the final second of a run may be missing. That is fine
    for a diagnostic: the counts that decide the assertions come from the JSON summary.
    A run with no failures writes no rows, and locust omits the file altogether.
    """
    if not failures_csv.exists():
        return ()
    with failures_csv.open(newline="") as handle:
        return tuple(
            LoadError(name=row["Name"], error=row["Error"], occurrences=int(row["Occurrences"]))
            for row in csv.DictReader(handle)
        )


def read_generator_warnings(stderr: str) -> tuple[str, ...]:
    """Locust reports its own CPU saturation on stderr; a saturated generator caps the measured rate.

    Kept from the marker onward so the per-line timestamp does not defeat the de-duplication.
    """
    saturated = (
        line[line.index(_GENERATOR_SATURATION_MARKER) :].strip()
        for line in stderr.splitlines()
        if _GENERATOR_SATURATION_MARKER in line
    )
    return tuple(dict.fromkeys(saturated))


def run_gateway_load(
    *,
    base_url: str,
    api_keys: tuple[str, ...],
    model: str,
    endpoints: tuple[str, ...],
    users: int,
    spawn_rate: float,
    duration_seconds: float,
) -> LoadResult:
    """Drive `endpoints` from headless locust and aggregate what it reported.

    Each simulated user picks one of `api_keys`, so auth and budget lookups spread over a
    pool of virtual keys instead of keeping one key's cache entry permanently warm, and one
    of `endpoints` round robin, so the run covers every route the caller asked for.
    """
    with tempfile.TemporaryDirectory(prefix="e2e-load-") as report_dir:
        csv_prefix = Path(report_dir) / _CSV_PREFIX
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "locust",
                "--headless",
                "--json",
                "--csv",
                str(csv_prefix),
                "--locustfile",
                str(_LOCUSTFILE),
                "--host",
                base_url,
                "--users",
                str(users),
                "--spawn-rate",
                str(spawn_rate),
                "--run-time",
                f"{int(duration_seconds)}s",
                "--exit-code-on-error",
                "0",
            ],
            env={
                **os.environ,
                "LOAD_API_KEYS": ",".join(api_keys),
                "LOAD_MODEL": model,
                "LOAD_ENDPOINTS": ",".join(endpoints),
            },
            capture_output=True,
            text=True,
            timeout=duration_seconds + 120,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"locust exited {completed.returncode} before it could report throughput "
                f"(a startup failure, not request failures, which are folded into the JSON summary via "
                f"--exit-code-on-error 0):\n{completed.stderr}"
            )
        try:
            entries = _STATS_ADAPTER.validate_json(completed.stdout)
        except ValueError as exc:
            raise RuntimeError(
                f"locust exited 0 but did not print a parseable --json throughput summary on stdout; "
                f"got stdout={completed.stdout!r}, stderr={completed.stderr!r}"
            ) from exc
        return aggregate_stats(
            entries,
            read_errors(csv_prefix.with_name(f"{_CSV_PREFIX}_failures.csv")),
            read_generator_warnings(completed.stderr),
        )
