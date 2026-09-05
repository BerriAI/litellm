from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from typing import Final

from pydantic import TypeAdapter

from ...shared.reporting.models import CaseResult
from ...shared.reporting.rendering import ReportSection, render_case_outcome
from .models import Measurement

MEASUREMENTS: Final = TypeAdapter(tuple[Measurement, ...])
ARTIFACT_KIND: Final = "e2e_benchmark"


def percentile(samples: Sequence[float], quantile: float) -> float:
    if not samples or not 0 < quantile <= 1:
        raise ValueError("percentile requires samples and a quantile in (0, 1]")
    return sorted(samples)[math.ceil(len(samples) * quantile) - 1]


def measurements(results: Sequence[CaseResult]) -> tuple[Measurement, ...]:
    return tuple(
        measurement
        for result in results
        for artifacts in result.artifacts.values()
        for artifact in artifacts
        if artifact.kind == ARTIFACT_KIND
        for measurement in MEASUREMENTS.validate_json(artifact.body)
    )


def render_measurements(values: tuple[Measurement, ...]) -> str:
    keys: Final = tuple(dict.fromkeys((value.route, value.profile) for value in values))
    header: Final = (
        "route/profile | backend | p50/p95/p99 ms | CPU ms/call | calls/s | RSS baseline/peak/after MiB | speedup"
    )

    def row(group: tuple[Measurement, ...], baseline: float) -> str:
        samples: Final = tuple(sample for value in group for sample in value.timing.latency_ms)
        median: Final = statistics.median(samples)
        cpu: Final = sum(value.timing.cpu_ms for value in group) / len(samples)
        rps: Final = len(samples) * 1000 / sum(value.timing.elapsed_ms for value in group)
        rss: Final = (
            statistics.median(value.memory.baseline_rss_bytes for value in group),
            max(value.memory.sampled_peak_rss_bytes for value in group),
            statistics.median(value.memory.retained_rss_bytes for value in group),
        )
        return (
            f"{group[0].route}/{group[0].profile} | {group[0].backend} | "
            f"{median:.3f}/{percentile(samples, 0.95):.3f}/{percentile(samples, 0.99):.3f} | {cpu:.3f} | {rps:.1f} | "
            f"{'/'.join(f'{value / 2**20:.1f}' for value in rss)} | {baseline / median:.2f}x"
        )

    def rows(route: str, profile: str) -> tuple[str, ...]:
        python: Final = tuple(
            value for value in values if (value.route, value.profile, value.backend) == (route, profile, "python")
        )
        rust: Final = tuple(
            value for value in values if (value.route, value.profile, value.backend) == (route, profile, "rust")
        )
        baseline: Final = statistics.median(sample for value in python for sample in value.timing.latency_ms)
        return row(python, baseline), row(rust, baseline)

    return "\n".join((header, *(line for route, profile in keys for line in rows(route, profile))))


def render_benchmark_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    values: Final = measurements(results)
    blocks: Final = tuple(render_case_outcome(result) for result in results)
    return (
        ReportSection(
            "End-to-end benchmark measurements",
            (*blocks, *((render_measurements(values),) if values else ())),
        ),
    )
