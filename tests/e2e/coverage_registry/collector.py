"""Diff the registry (denominator) against the @pytest.mark.covers markers on the
live tests (numerator) and report coverage per module.

Coverage here is static: it reads the markers via a collect-only pass, so it runs
no test and needs no live proxy. Whether a covered cell currently passes or fails
(covered_pass vs covered_fail) is a separate, live concern layered on top later.

A skipped test asserts nothing, so its markers do not count: a cell is covered
only when at least one test that pytest would actually run declares it. Skip
state is read with pytest's own evaluator, so `skip` and `skipif` are resolved
exactly as the e2e run resolves them in this environment. The one skip the
collector cannot see is `pytest.skip()` called from inside a test body, which
does not exist until the test runs.

    cd tests/e2e && PYTHONPATH=. python -m coverage_registry.collector
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
from _pytest.skipping import evaluate_skip_marks
from pydantic import BaseModel

from .registry import load_registry
from .schema import MODULE_ORDER, Cell, Tier, dashboard_module, loki_module_label

E2E_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True, slots=True)
class CollectedMarkers:
    """What a collect-only pass saw: cell ids declared by tests that would run,
    cell ids only ever declared by skipped tests, and nodes that failed to import."""

    covered: frozenset[str]
    skipped_only: frozenset[str]
    collection_errors: tuple[str, ...]


def _is_skipped(item: pytest.Item) -> bool:
    """True when pytest would skip this test instead of running it.

    A marker pytest cannot evaluate (for example a bare boolean `skipif` with no
    reason) turns into a setup failure at run time, so the test asserts nothing
    either way and is treated the same as a skip.
    """
    try:
        return evaluate_skip_marks(item) is not None
    except (pytest.fail.Exception, TypeError):
        return True


class _CoversSink:
    """Pytest plugin: after collection, capture every cell id declared via
    @pytest.mark.covers(...) split by whether its test would run, plus any nodes
    that failed to import."""

    def __init__(self) -> None:
        self.covered_ids: frozenset[str] = frozenset()
        self.skipped_only_ids: frozenset[str] = frozenset()
        self.collection_errors: tuple[str, ...] = ()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        marker_args: tuple[tuple[bool, tuple[object, ...]], ...] = tuple(
            (skipped, marker.args)
            for item, skipped in ((i, _is_skipped(i)) for i in session.items)
            for marker in item.iter_markers(name="covers")
        )
        declared = tuple(
            (skipped, arg)
            for skipped, args in marker_args
            for arg in args
            if isinstance(arg, str)
        )
        self.covered_ids = frozenset(
            cell_id for skipped, cell_id in declared if not skipped
        )
        self.skipped_only_ids = (
            frozenset(cell_id for skipped, cell_id in declared if skipped)
            - self.covered_ids
        )

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_errors = (*self.collection_errors, report.nodeid)


def collect_markers(e2e_dir: Path = E2E_DIR) -> CollectedMarkers:
    """Read every @pytest.mark.covers marker in `e2e_dir` via a collect-only pass."""
    sink = _CoversSink()
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
    return CollectedMarkers(
        covered=sink.covered_ids,
        skipped_only=sink.skipped_only_ids,
        collection_errors=sink.collection_errors,
    )


@dataclass(frozen=True, slots=True)
class ModuleCoverage:
    module: str
    total: int
    covered: int
    p0_total: int
    p0_covered: int

    @property
    def coverage_percent(self) -> float:
        return _percent(self.covered, self.total)


@dataclass(frozen=True, slots=True)
class CoverageReport:
    modules: tuple[ModuleCoverage, ...]
    total: int
    covered: int
    p0_total: int
    p0_covered: int
    p0_gaps: tuple[str, ...]
    orphan_markers: tuple[str, ...]
    skipped_markers: tuple[str, ...]
    collection_errors: tuple[str, ...]

    @property
    def coverage_percent(self) -> float:
        return _percent(self.covered, self.total)


def _percent(covered: int, total: int) -> float:
    return (100.0 * covered / total) if total else 0.0


def _module_coverage(
    module: str, cells: tuple[Cell, ...], covered: frozenset[str]
) -> ModuleCoverage:
    in_module = tuple(c for c in cells if dashboard_module(c) == module)
    p0 = tuple(c for c in in_module if c.tier is Tier.P0)
    return ModuleCoverage(
        module=module,
        total=len(in_module),
        covered=sum(1 for c in in_module if c.id in covered),
        p0_total=len(p0),
        p0_covered=sum(1 for c in p0 if c.id in covered),
    )


def compute_coverage(
    cells: tuple[Cell, ...],
    covered: frozenset[str],
    collection_errors: tuple[str, ...] = (),
    skipped_only: frozenset[str] = frozenset(),
) -> CoverageReport:
    p0_cells = tuple(c for c in cells if c.tier is Tier.P0)
    registry_ids = frozenset(c.id for c in cells)
    return CoverageReport(
        modules=tuple(_module_coverage(m, cells, covered) for m in MODULE_ORDER),
        total=len(cells),
        covered=sum(1 for c in cells if c.id in covered),
        p0_total=len(p0_cells),
        p0_covered=sum(1 for c in p0_cells if c.id in covered),
        p0_gaps=tuple(sorted(c.id for c in p0_cells if c.id not in covered)),
        orphan_markers=tuple(sorted((covered | skipped_only) - registry_ids)),
        skipped_markers=tuple(sorted(skipped_only & registry_ids)),
        collection_errors=collection_errors,
    )


def _row(label: str, covered: int, total: int) -> str:
    frac = f"{covered}/{total}"
    return f"{label:30}{frac:>12}{_percent(covered, total):>11.1f}%"


def render(report: CoverageReport) -> str:
    rows = tuple(_row(m.module, m.covered, m.total) for m in report.modules)
    lines = (
        f"{'MODULE':30}{'COVERED':>12}{'COVERAGE':>12}",
        *rows,
        "-" * 54,
        _row("ALL", report.covered, report.total),
        "",
        f"Headline coverage: {report.covered}/{report.total}  ({report.coverage_percent:.1f}%)",
    )
    orphans = (
        (
            f"\n{len(report.orphan_markers)} marker(s) point at ids not in the registry "
            f"(reconcile: fix the marker or add the cell):\n  "
            + "\n  ".join(report.orphan_markers),
        )
        if report.orphan_markers
        else ()
    )
    skipped = (
        (
            f"\n{len(report.skipped_markers)} cell(s) are claimed only by skipped tests, "
            f"so they count as uncovered (unskip the test or drop the marker):\n  "
            + "\n  ".join(report.skipped_markers),
        )
        if report.skipped_markers
        else ()
    )
    warning = (
        (
            f"\nWARNING: {len(report.collection_errors)} node(s) failed to import during "
            f"collection, so coverage may undercount:\n  "
            + "\n  ".join(report.collection_errors),
        )
        if report.collection_errors
        else ()
    )
    return "\n".join((*lines, *orphans, *skipped, *warning))


def _report_dict(report: CoverageReport) -> dict[str, object]:
    return {
        "covered": report.covered,
        "total": report.total,
        "coverage_percent": report.coverage_percent,
        "modules": [
            {
                "module": m.module,
                "covered": m.covered,
                "total": m.total,
                "coverage_percent": m.coverage_percent,
                "p0_covered": m.p0_covered,
                "p0_total": m.p0_total,
            }
            for m in report.modules
        ],
        "orphan_markers": list(report.orphan_markers),
        "skipped_markers": list(report.skipped_markers),
        "collection_errors": list(report.collection_errors),
    }


def render_json(report: CoverageReport) -> str:
    return json.dumps(_report_dict(report), indent=2, sort_keys=True)


def _label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def render_prometheus(report: CoverageReport) -> str:
    lines = [
        "# HELP litellm_e2e_coverage_cells E2E coverage registry cells by module and state.",
        "# TYPE litellm_e2e_coverage_cells gauge",
    ]
    for module in report.modules:
        label = _label_value(module.module)
        lines.append(
            f'litellm_e2e_coverage_cells{{module="{label}",state="covered"}} {module.covered}'
        )
        lines.append(
            f'litellm_e2e_coverage_cells{{module="{label}",state="total"}} {module.total}'
        )
    lines.extend(
        [
            f'litellm_e2e_coverage_cells{{module="ALL",state="covered"}} {report.covered}',
            f'litellm_e2e_coverage_cells{{module="ALL",state="total"}} {report.total}',
            "# HELP litellm_e2e_coverage_percent E2E coverage percent by module.",
            "# TYPE litellm_e2e_coverage_percent gauge",
        ]
    )
    for module in report.modules:
        label = _label_value(module.module)
        lines.append(
            f'litellm_e2e_coverage_percent{{module="{label}"}} {module.coverage_percent:.6f}'
        )
    lines.extend(
        [
            f'litellm_e2e_coverage_percent{{module="ALL"}} {report.coverage_percent:.6f}',
            "# HELP litellm_e2e_coverage_orphan_markers Coverage markers not found in the registry.",
            "# TYPE litellm_e2e_coverage_orphan_markers gauge",
            f"litellm_e2e_coverage_orphan_markers {len(report.orphan_markers)}",
            "# HELP litellm_e2e_coverage_skipped_markers Registry cells claimed only by skipped tests.",
            "# TYPE litellm_e2e_coverage_skipped_markers gauge",
            f"litellm_e2e_coverage_skipped_markers {len(report.skipped_markers)}",
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
            f"covered={report.covered} total={report.total}"
        )
    ]
    lines.extend(
        (
            f"COVERAGE_MODULE module={loki_module_label(module.module)} "
            f"percent={module.coverage_percent:.1f} "
            f"covered={module.covered} total={module.total}"
        )
        for module in report.modules
    )
    return "\n".join(lines)


class _CliArgs(BaseModel):
    format: Literal["text", "json", "prometheus", "loki"]
    strict: bool
    fail_on_collection_errors: bool


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
        help="Exit non-zero if markers outside the registry are found.",
    )
    parser.add_argument(
        "--fail-on-collection-errors",
        action="store_true",
        help="Exit non-zero if pytest collection errors are found.",
    )
    args = _CliArgs.model_validate(vars(parser.parse_args()))
    cells = load_registry()
    markers = collect_markers()
    report = compute_coverage(
        cells,
        markers.covered,
        markers.collection_errors,
        markers.skipped_only,
    )
    output = {
        "text": render,
        "json": render_json,
        "prometheus": render_prometheus,
        "loki": render_loki,
    }[args.format](report)
    print(output)  # noqa: T201  # CLI entrypoint output
    if args.strict and report.orphan_markers:
        return 1
    if args.fail_on_collection_errors and report.collection_errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
