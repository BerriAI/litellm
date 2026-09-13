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


def measurement_warnings(values: tuple[Measurement, ...]) -> tuple[str, ...]:
    keys: Final = tuple(dict.fromkeys((value.route, value.profile, value.backend) for value in values))

    def warnings(route: str, profile: str, backend: str) -> tuple[str, ...]:
        group: Final = tuple(
            value for value in values if (value.route, value.profile, value.backend) == (route, profile, backend)
        )
        medians: Final = tuple(statistics.median(value.timing.latency_ms) for value in group)
        prefix: Final = f"{route}/{profile}/{backend}"
        return (
            *((f"{prefix}: one process repeat cannot establish repeatability",) if len(group) == 1 else ()),
            *((f"{prefix}: backend order is unbalanced",) if len(group) % 2 else ()),
            *(
                (f"{prefix}: batch shorter than 1 second; increase --min-time",)
                if any(value.timing.elapsed_ms < 1000 for value in group)
                else ()
            ),
            *(
                (f"{prefix}: repeat p50 range exceeds 10% of its median; investigate variability",)
                if max(medians) - min(medians) > statistics.median(medians) * 0.1
                else ()
            ),
        )

    return tuple(warning for route, profile, backend in keys for warning in warnings(route, profile, backend))


def render_measurements(values: tuple[Measurement, ...]) -> str:
    keys: Final = tuple(dict.fromkeys((value.route, value.profile) for value in values))
    header: Final = (
        "route/profile | backend | p50/p95/p99 ms | mean/stdev ms | CPU ms/call | calls/s | "
        "RSS baseline/peak/after MiB | pooled p50 ratio"
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
            f"{median:.3f}/{percentile(samples, 0.95):.3f}/{percentile(samples, 0.99):.3f} | "
            f"{statistics.mean(samples):.3f}/{statistics.pstdev(samples):.3f} | {cpu:.3f} | {rps:.1f} | "
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

    repeats: Final = tuple(
        f"{value.route}/{value.profile} | {value.backend} | {value.repeat + 1} | "
        f"{len(value.timing.latency_ms)} | {value.timing.elapsed_ms:.1f} | "
        f"{statistics.median(value.timing.latency_ms):.3f} | {statistics.mean(value.timing.latency_ms):.3f} | "
        f"{len(value.timing.latency_ms) * 1000 / value.timing.elapsed_ms:.1f}"
        for value in values
    )
    return "\n".join(
        (
            header,
            *(line for route, profile in keys for line in rows(route, profile)),
            "Per-repeat measurements (independent processes):",
            "route/profile | backend | repeat | calls | batch ms | p50 ms | mean ms | calls/s",
            *repeats,
            "Completed measurements do not establish statistical significance or isolate PyO3 overhead",
            *(f"WARNING: {warning}" for warning in measurement_warnings(values)),
        )
    )


def render_benchmark_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    values: Final = measurements(results)
    blocks: Final = tuple(render_case_outcome(result) for result in results)
    return (
        ReportSection(
            "End-to-end benchmark measurements",
            (*blocks, *((render_measurements(values),) if values else ())),
        ),
    )
