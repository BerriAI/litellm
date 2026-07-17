"""Diff the registry (denominator) against the @pytest.mark.covers markers on the
live tests (numerator) and report coverage per module.

Coverage here is static: it reads the markers via a collect-only pass, so it runs
no test and needs no live proxy. Whether a covered cell currently passes or fails
(covered_pass vs covered_fail) is a separate, live concern layered on top later.

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
from pydantic import BaseModel

from .product_surface import ROUTE_CHECKABLE_ENDPOINTS, route_table_endpoints
from .registry import load_registry
from .schema import MODULE_ORDER, Cell, Tier, dashboard_module, loki_module_label, parse_llm_id

E2E_DIR = Path(__file__).resolve().parent.parent


class _CoversSink:
    """Pytest plugin: after collection, capture every cell id declared via
    @pytest.mark.covers(...), plus any nodes that failed to import."""

    def __init__(self) -> None:
        self.covered_ids: frozenset[str] = frozenset()
        self.collection_errors: tuple[str, ...] = ()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        marker_args: tuple[tuple[object, ...], ...] = tuple(
            marker.args for item in session.items for marker in item.iter_markers(name="covers")
        )
        self.covered_ids = frozenset(arg for args in marker_args for arg in args if isinstance(arg, str))

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_errors = (*self.collection_errors, report.nodeid)


def collect_covered_ids(
    e2e_dir: Path = E2E_DIR,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return (covered cell ids, nodeids that failed to import)."""
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
    return sink.covered_ids, sink.collection_errors


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
    collection_errors: tuple[str, ...]
    route_table_drift: tuple[str, ...] = ()

    @property
    def coverage_percent(self) -> float:
        return _percent(self.covered, self.total)


def _percent(covered: int, total: int) -> float:
    return (100.0 * covered / total) if total else 0.0


def _module_coverage(module: str, cells: tuple[Cell, ...], covered: frozenset[str]) -> ModuleCoverage:
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
    route_table_drift: tuple[str, ...] = (),
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
        orphan_markers=tuple(sorted(covered - registry_ids)),
        collection_errors=collection_errors,
        route_table_drift=route_table_drift,
    )


def compute_route_table_drift(cells: tuple[Cell, ...]) -> tuple[str, ...]:
    """LLM endpoints the denominator enumerates that the live proxy route table does
    not serve. Empty when the route table cannot be read (litellm not importable) or
    when everything the denominator enumerates is served."""
    served = route_table_endpoints()
    if served is None:
        return ()
    enumerated = frozenset(parsed.endpoint for c in cells if (parsed := parse_llm_id(c.id)) is not None)
    checkable = enumerated & ROUTE_CHECKABLE_ENDPOINTS
    return tuple(sorted(endpoint for endpoint in checkable if endpoint not in served))


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
            f"(reconcile: fix the marker or add the cell):\n  " + "\n  ".join(report.orphan_markers),
        )
        if report.orphan_markers
        else ()
    )
    warning = (
        (
            f"\nWARNING: {len(report.collection_errors)} node(s) failed to import during "
            f"collection, so coverage may undercount:\n  " + "\n  ".join(report.collection_errors),
        )
        if report.collection_errors
        else ()
    )
    drift = (
        (
            f"\nWARNING: {len(report.route_table_drift)} enumerated LLM endpoint(s) are not in "
            f"the live proxy route table (denominator drift, reconcile the schema vocab):\n  "
            + "\n  ".join(report.route_table_drift),
        )
        if report.route_table_drift
        else ()
    )
    return "\n".join((*lines, *orphans, *warning, *drift))


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
        lines.append(f'litellm_e2e_coverage_cells{{module="{label}",state="covered"}} {module.covered}')
        lines.append(f'litellm_e2e_coverage_cells{{module="{label}",state="total"}} {module.total}')
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
        lines.append(f'litellm_e2e_coverage_percent{{module="{label}"}} {module.coverage_percent:.6f}')
    lines.extend(
        [
            f'litellm_e2e_coverage_percent{{module="ALL"}} {report.coverage_percent:.6f}',
            "# HELP litellm_e2e_coverage_orphan_markers Coverage markers not found in the registry.",
            "# TYPE litellm_e2e_coverage_orphan_markers gauge",
            f"litellm_e2e_coverage_orphan_markers {len(report.orphan_markers)}",
            "# HELP litellm_e2e_coverage_collection_errors Pytest nodes that failed during collection.",
            "# TYPE litellm_e2e_coverage_collection_errors gauge",
            f"litellm_e2e_coverage_collection_errors {len(report.collection_errors)}",
        ]
    )
    return "\n".join(lines)


def render_loki(report: CoverageReport) -> str:
    lines = [(f"COVERAGE_TOTAL percent={report.coverage_percent:.1f} covered={report.covered} total={report.total}")]
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
    covered, errors = collect_covered_ids()
    report = compute_coverage(cells, covered, errors, compute_route_table_drift(cells))
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
