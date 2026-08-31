from __future__ import annotations

import csv
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path

from pydantic import BaseModel, TypeAdapter

_GENERATOR_SATURATION_MARKER = "CPU usage above"
_MAX_REPORTED_ERRORS = 5


class LocustStatEntry(BaseModel):
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
class LoadResult:
    requests: int
    failures: int
    requests_per_second: float
    median_response_seconds: float
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


def median_seconds(entries: list[LocustStatEntry]) -> float:
    samples = sorted(
        (milliseconds, count) for entry in entries for milliseconds, count in entry.response_times.items()
    )
    total = sum(count for _, count in samples)
    if total == 0:
        return 0.0
    running = accumulate(count for _, count in samples)
    return next(
        milliseconds for (milliseconds, _), seen in zip(samples, running) if seen >= total / 2
    ) / 1000.0


def aggregate_stats(
    entries: list[LocustStatEntry],
    errors: tuple[LoadError, ...],
    generator_warnings: tuple[str, ...],
) -> LoadResult:
    requests = sum(entry.num_requests for entry in entries)
    failures = sum(entry.num_failures for entry in entries)
    if not entries or requests == 0:
        return LoadResult(
            requests=requests,
            failures=failures,
            requests_per_second=0.0,
            median_response_seconds=0.0,
            errors=errors,
            generator_warnings=generator_warnings,
        )
    elapsed = max(entry.last_request_timestamp for entry in entries) - min(entry.start_time for entry in entries)
    return LoadResult(
        requests=requests,
        failures=failures,
        requests_per_second=requests / elapsed if elapsed > 0 else 0.0,
        median_response_seconds=median_seconds(entries),
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
