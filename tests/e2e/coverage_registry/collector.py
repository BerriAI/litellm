"""Collect e2e coverage metadata from pytest markers.

Coverage is declared directly on tests:

    @pytest.mark.e2e_coverage(
        module="core_llms",
        endpoint="/chat/completions",
        provider="openai",
        params=["tools"],
    )

The collector runs pytest in collect-only mode, validates those markers, expands
them into endpoint x provider x parameter units, and renders Grafana/Loki-friendly
summaries. It does not execute live e2e tests.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pytest
from pydantic import ValidationError

from .schema import (
    MODULE_DISPLAY_NAMES,
    MODULE_ORDER,
    CoverageModule,
    CoveragePoint,
    CoverageUnit,
    units_for_point,
)

E2E_DIR = Path(__file__).resolve().parent.parent


class _CoverageSink:
    """Pytest plugin that records e2e coverage markers after collection."""

    def __init__(self) -> None:
        self.points_by_nodeid: dict[str, tuple[CoveragePoint, ...]] = {}
        self.collected_nodeids: tuple[str, ...] = ()
        self.unmarked_nodeids: tuple[str, ...] = ()
        self.invalid_markers: tuple[str, ...] = ()
        self.collection_errors: tuple[str, ...] = ()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        collected_nodeids: list[str] = []
        unmarked_nodeids: list[str] = []
        invalid_markers: list[str] = []
        points_by_nodeid: dict[str, tuple[CoveragePoint, ...]] = {}

        for item in session.items:
            collected_nodeids.append(item.nodeid)
            markers = tuple(item.iter_markers(name="e2e_coverage"))
            if not markers:
                unmarked_nodeids.append(item.nodeid)
                continue

            points: list[CoveragePoint] = []
            for marker in markers:
                if marker.args:
                    invalid_markers.append(
                        f"{item.nodeid}: e2e_coverage uses keyword fields only"
                    )
                    continue
                try:
                    points.append(CoveragePoint.model_validate(marker.kwargs))
                except ValidationError as exc:
                    invalid_markers.append(f"{item.nodeid}: {exc}")
            if points:
                points_by_nodeid[item.nodeid] = tuple(points)

        self.points_by_nodeid = points_by_nodeid
        self.collected_nodeids = tuple(sorted(collected_nodeids))
        self.unmarked_nodeids = tuple(sorted(unmarked_nodeids))
        self.invalid_markers = tuple(sorted(invalid_markers))

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_errors = (*self.collection_errors, report.nodeid)


@dataclass(frozen=True, slots=True)
class CoverageMarkers:
    points_by_nodeid: Mapping[str, tuple[CoveragePoint, ...]]
    collected_nodeids: tuple[str, ...]
    unmarked_nodeids: tuple[str, ...]
    invalid_markers: tuple[str, ...]
    collection_errors: tuple[str, ...]


def collect_coverage_markers(e2e_dir: Path = E2E_DIR) -> CoverageMarkers:
    """Return validated coverage markers plus collection diagnostics."""
    sink = _CoverageSink()
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main(
            [
                "--collect-only",
                "-qq",
                "--continue-on-collection-errors",
                "-p",
                "no:cacheprovider",
                str(e2e_dir),
            ],
            plugins=[sink],
        )
    return CoverageMarkers(
        points_by_nodeid=sink.points_by_nodeid,
        collected_nodeids=sink.collected_nodeids,
        unmarked_nodeids=sink.unmarked_nodeids,
        invalid_markers=sink.invalid_markers,
        collection_errors=sink.collection_errors,
    )


@dataclass(frozen=True, slots=True)
class ModuleCoverage:
    module: CoverageModule
    unit_count: int
    test_count: int

    @property
    def display_name(self) -> str:
        return MODULE_DISPLAY_NAMES[self.module]

    @property
    def coverage_percent(self) -> float:
        return 100.0 if self.unit_count else 0.0


@dataclass(frozen=True, slots=True)
class EndpointCoverage:
    module: CoverageModule
    endpoint: str
    provider: str
    param_count: int
    test_count: int

    @property
    def display_name(self) -> str:
        return MODULE_DISPLAY_NAMES[self.module]

    @property
    def coverage_percent(self) -> float:
        return 100.0 if self.param_count else 0.0


@dataclass(frozen=True, slots=True)
class CoverageReport:
    units: tuple[CoverageUnit, ...]
    modules: tuple[ModuleCoverage, ...]
    endpoints: tuple[EndpointCoverage, ...]
    test_count: int
    collected_test_count: int
    unmarked_test_count: int
    invalid_marker_count: int
    unmarked_nodeids: tuple[str, ...]
    invalid_markers: tuple[str, ...]
    collection_errors: tuple[str, ...]

    @property
    def total(self) -> int:
        return len(self.units)

    @property
    def covered(self) -> int:
        return len(self.units)

    @property
    def coverage_percent(self) -> float:
        return 100.0 if self.units else 0.0


def _unique_units(
    points_by_nodeid: Mapping[str, tuple[CoveragePoint, ...]],
) -> tuple[CoverageUnit, ...]:
    units = {
        unit.key: unit
        for points in points_by_nodeid.values()
        for point in points
        for unit in units_for_point(point)
    }
    return tuple(units[key] for key in sorted(units))


def _module_coverage(
    module: CoverageModule,
    units: tuple[CoverageUnit, ...],
    points_by_nodeid: Mapping[str, tuple[CoveragePoint, ...]],
) -> ModuleCoverage:
    test_nodeids = frozenset(
        nodeid
        for nodeid, points in points_by_nodeid.items()
        if any(point.module == module for point in points)
    )
    return ModuleCoverage(
        module=module,
        unit_count=sum(1 for unit in units if unit.module == module),
        test_count=len(test_nodeids),
    )


def _endpoint_coverage(
    units: tuple[CoverageUnit, ...],
    points_by_nodeid: Mapping[str, tuple[CoveragePoint, ...]],
) -> tuple[EndpointCoverage, ...]:
    keys = sorted({(unit.module, unit.endpoint, unit.provider) for unit in units})
    rows: list[EndpointCoverage] = []
    for module, endpoint, provider in keys:
        params = frozenset(
            unit.param
            for unit in units
            if unit.module == module
            and unit.endpoint == endpoint
            and unit.provider == provider
        )
        test_nodeids = frozenset(
            nodeid
            for nodeid, points in points_by_nodeid.items()
            if any(
                point.module == module
                and point.endpoint == endpoint
                and point.provider == provider
                for point in points
            )
        )
        rows.append(
            EndpointCoverage(
                module=module,
                endpoint=endpoint,
                provider=provider,
                param_count=len(params),
                test_count=len(test_nodeids),
            )
        )
    return tuple(rows)


def compute_coverage(
    markers: CoverageMarkers,
) -> CoverageReport:
    units = _unique_units(markers.points_by_nodeid)
    mapped_nodeids = frozenset(markers.points_by_nodeid)
    return CoverageReport(
        units=units,
        modules=tuple(
            _module_coverage(module, units, markers.points_by_nodeid)
            for module in MODULE_ORDER
        ),
        endpoints=_endpoint_coverage(units, markers.points_by_nodeid),
        test_count=len(mapped_nodeids),
        collected_test_count=len(markers.collected_nodeids),
        unmarked_test_count=len(markers.unmarked_nodeids),
        invalid_marker_count=len(markers.invalid_markers),
        unmarked_nodeids=markers.unmarked_nodeids,
        invalid_markers=markers.invalid_markers,
        collection_errors=markers.collection_errors,
    )


def _row(label: str, unit_count: int, test_count: int) -> str:
    return f"{label:30}{unit_count:>12}{test_count:>9}"


def render(report: CoverageReport) -> str:
    rows = tuple(
        _row(m.display_name, m.unit_count, m.test_count) for m in report.modules
    )
    lines = (
        f"{'MODULE':30}{'UNITS':>12}{'TESTS':>9}",
        *rows,
        "-" * 51,
        _row("ALL", report.total, report.test_count),
        "",
        f"Coverage metadata: {report.test_count}/{report.collected_test_count} tests mapped",
        f"Unique endpoint x provider x parameter units: {report.total}",
        f"Unmarked collected tests: {report.unmarked_test_count}",
        f"Invalid coverage markers: {report.invalid_marker_count}",
    )
    invalid = (
        (
            f"\n{len(report.invalid_markers)} invalid marker(s):\n  "
            + "\n  ".join(report.invalid_markers),
        )
        if report.invalid_markers
        else ()
    )
    unmarked = (
        (
            f"\n{len(report.unmarked_nodeids)} collected test(s) missing "
            "e2e_coverage:\n  " + "\n  ".join(report.unmarked_nodeids),
        )
        if report.unmarked_nodeids
        else ()
    )
    warning = (
        (
            f"\nWARNING: {len(report.collection_errors)} node(s) failed to import "
            "during collection:\n  " + "\n  ".join(report.collection_errors),
        )
        if report.collection_errors
        else ()
    )
    return "\n".join((*lines, *invalid, *unmarked, *warning))


def _report_dict(report: CoverageReport) -> dict[str, object]:
    return {
        "covered": report.covered,
        "total": report.total,
        "coverage_percent": report.coverage_percent,
        "test_count": report.test_count,
        "collected_test_count": report.collected_test_count,
        "unmarked_test_count": report.unmarked_test_count,
        "invalid_marker_count": report.invalid_marker_count,
        "modules": [
            {
                "module": m.module,
                "display_name": m.display_name,
                "covered": m.unit_count,
                "total": m.unit_count,
                "coverage_percent": m.coverage_percent,
                "test_count": m.test_count,
            }
            for m in report.modules
        ],
        "endpoints": [
            {
                "module": e.module,
                "display_name": e.display_name,
                "endpoint": e.endpoint,
                "provider": e.provider,
                "covered": e.param_count,
                "total": e.param_count,
                "coverage_percent": e.coverage_percent,
                "test_count": e.test_count,
            }
            for e in report.endpoints
        ],
        "units": [
            {
                "module": unit.module,
                "endpoint": unit.endpoint,
                "provider": unit.provider,
                "param": unit.param,
            }
            for unit in report.units
        ],
        "unmarked_nodeids": list(report.unmarked_nodeids),
        "invalid_markers": list(report.invalid_markers),
        "collection_errors": list(report.collection_errors),
    }


def render_json(report: CoverageReport) -> str:
    return json.dumps(_report_dict(report), indent=2, sort_keys=True)


def _label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_prometheus(report: CoverageReport) -> str:
    lines = [
        "# HELP litellm_e2e_coverage_units Unique endpoint x provider x parameter units covered by e2e tests.",
        "# TYPE litellm_e2e_coverage_units gauge",
    ]
    for module in report.modules:
        label = _label_value(module.module)
        lines.append(
            f'litellm_e2e_coverage_units{{module="{label}"}} {module.unit_count}'
        )
    lines.extend(
        [
            f'litellm_e2e_coverage_units{{module="ALL"}} {report.total}',
            "# HELP litellm_e2e_coverage_tests E2E pytest test count with coverage metadata.",
            "# TYPE litellm_e2e_coverage_tests gauge",
        ]
    )
    for module in report.modules:
        label = _label_value(module.module)
        lines.append(
            f'litellm_e2e_coverage_tests{{module="{label}"}} {module.test_count}'
        )
    lines.extend(
        [
            f'litellm_e2e_coverage_tests{{module="ALL"}} {report.test_count}',
            "# HELP litellm_e2e_coverage_unmarked_tests E2E collected tests without e2e_coverage metadata.",
            "# TYPE litellm_e2e_coverage_unmarked_tests gauge",
            f"litellm_e2e_coverage_unmarked_tests {report.unmarked_test_count}",
            "# HELP litellm_e2e_coverage_invalid_markers E2E tests with invalid e2e_coverage metadata.",
            "# TYPE litellm_e2e_coverage_invalid_markers gauge",
            f"litellm_e2e_coverage_invalid_markers {report.invalid_marker_count}",
            "# HELP litellm_e2e_coverage_collection_errors Pytest nodes that failed during collection.",
            "# TYPE litellm_e2e_coverage_collection_errors gauge",
            f"litellm_e2e_coverage_collection_errors {len(report.collection_errors)}",
        ]
    )
    return "\n".join(lines)


def render_loki(report: CoverageReport) -> str:
    lines = [
        (
            f"COVERAGE_TOTAL percent={report.coverage_percent:.1f} "
            f"covered={report.covered} total={report.total} tests={report.test_count} "
            f"unmarked_tests={report.unmarked_test_count} "
            f"invalid_markers={report.invalid_marker_count}"
        )
    ]
    lines.extend(
        (
            f"COVERAGE_MODULE module={module.module} "
            f"percent={module.coverage_percent:.1f} "
            f"covered={module.unit_count} total={module.unit_count} "
            f"tests={module.test_count}"
        )
        for module in report.modules
    )
    return "\n".join(lines)


def main() -> int:
    parser = ArgumentParser()
    parser.add_argument(
        "--format",
        choices=("text", "json", "prometheus", "loki"),
        default="text",
        help="Output format. Use loki for structured stdout lines in the e2e job.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if tests are missing or have invalid coverage metadata.",
    )
    parser.add_argument(
        "--fail-on-collection-errors",
        action="store_true",
        help="Exit non-zero if pytest collection errors are found.",
    )
    args = parser.parse_args()
    report = compute_coverage(collect_coverage_markers())
    output = {
        "text": render,
        "json": render_json,
        "prometheus": render_prometheus,
        "loki": render_loki,
    }[args.format](report)
    print(output)  # noqa: T201  # CLI entrypoint output
    if args.strict and (report.unmarked_nodeids or report.invalid_markers):
        return 1
    if args.fail_on_collection_errors and report.collection_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
