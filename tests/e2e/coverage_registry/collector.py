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
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

from .registry import load_registry
from .schema import MODULE_ORDER, ROLLUP, Cell, Tier

E2E_DIR = Path(__file__).resolve().parent.parent


class _CoversSink:
    """Pytest plugin: after collection, capture every cell id declared via
    @pytest.mark.covers(...), plus any nodes that failed to import."""

    def __init__(self) -> None:
        self.covered_ids: frozenset[str] = frozenset()
        self.collection_errors: tuple[str, ...] = ()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        self.covered_ids = frozenset(
            arg
            for item in session.items
            for marker in item.iter_markers(name="covers")
            for arg in marker.args
            if isinstance(arg, str)
        )

    def pytest_collectreport(self, report: pytest.CollectReport) -> None:
        if report.failed:
            self.collection_errors = (*self.collection_errors, report.nodeid)


def collect_covered_ids(e2e_dir: Path = E2E_DIR) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return (covered cell ids, nodeids that failed to import)."""
    sink = _CoversSink()
    with contextlib.redirect_stdout(io.StringIO()):
        pytest.main(
            ["--collect-only", "-qq", "--continue-on-collection-errors", "-p", "no:cacheprovider", str(e2e_dir)],
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


def _module_coverage(module: str, cells: tuple[Cell, ...], covered: frozenset[str]) -> ModuleCoverage:
    in_module = tuple(c for c in cells if ROLLUP[c.module] == module)
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
    )


def _row(label: str, covered: int, total: int, p0_covered: int, p0_total: int) -> str:
    frac = f"{covered}/{total}"
    p0 = f"{p0_covered}/{p0_total}"
    return f"{label:30}{frac:>12}{p0:>14}"


def render(report: CoverageReport) -> str:
    rows = tuple(_row(m.module, m.covered, m.total, m.p0_covered, m.p0_total) for m in report.modules)
    pct = (100.0 * report.p0_covered / report.p0_total) if report.p0_total else 0.0
    lines = (
        f"{'MODULE':30}{'COVERED':>12}{'P0 COVERED':>14}",
        *rows,
        "-" * 56,
        _row("ALL", report.covered, report.total, report.p0_covered, report.p0_total),
        "",
        f"Headline (P0 coverage): {report.p0_covered}/{report.p0_total}  ({pct:.1f}%)",
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
    return "\n".join((*lines, *orphans, *warning))


def main() -> int:
    cells = load_registry()
    covered, errors = collect_covered_ids()
    print(render(compute_coverage(cells, covered, errors)))  # noqa: T201  # CLI entrypoint output
    return 0


if __name__ == "__main__":
    sys.exit(main())
