"""Diff the registry (denominator) against the @pytest.mark.covers markers on the
live tests (numerator) and report coverage per module.

Coverage here is static: a cell is covered when a test declaring it exists in the
source tree. Whether that test is deselected on this run, skipped for a missing
optional dependency, or currently passing is a runtime concern and must not move
the number, so the markers are read straight off the source with `ast` rather
than off whatever pytest happened to keep after collection. A collect-only pytest
pass still runs alongside it, for two things the source text cannot give: markers
built at import time (`pytest.mark.covers(*fn(...))` inside `pytest.param`) and
the nodeids that failed to import, whose cells are genuinely unknowable.

The TypeScript Playwright suite under `tests/e2e/ui/` emits no pytest markers, so
it declares the cells it covers in `tests/e2e/ui/coverage.yaml` instead. Each row
names the spec and the test title that prove it, and both are resolved against
the tree, so deleting or renaming that Playwright test drops the cell out of the
numerator and fails `--strict` the same way an unknown cell id does.

    cd tests/e2e && PYTHONPATH=. python -m coverage_registry.collector
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import re
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pytest
import yaml
from pydantic import BaseModel, ConfigDict

from .registry import load_registry
from .schema import MODULE_ORDER, Cell, Tier, dashboard_module, loki_module_label

E2E_DIR = Path(__file__).resolve().parent.parent
UI_DECLARATION_FILE = E2E_DIR / "ui" / "coverage.yaml"

_UNSCANNED_DIRS: frozenset[str] = frozenset(
    {"__pycache__", ".git", ".venv", "node_modules", "site-packages", "venv"}
)


def _is_covers_call(func: ast.expr) -> bool:
    """True for `<anything>.mark.covers` and the `from pytest import mark` spelling."""
    if not isinstance(func, ast.Attribute) or func.attr != "covers":
        return False
    owner = func.value
    if isinstance(owner, ast.Attribute):
        return owner.attr == "mark"
    return isinstance(owner, ast.Name) and owner.id == "mark"


def _covers_ids_in_source(source: str) -> frozenset[str]:
    tree = ast.parse(source)
    return frozenset(
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and _is_covers_call(node.func)
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    )


def _scannable(path: Path) -> bool:
    return not _UNSCANNED_DIRS.intersection(path.parts)


def _read_covers_ids(path: Path) -> frozenset[str] | None:
    """The file's declared cell ids, or None when it cannot be parsed."""
    try:
        return _covers_ids_in_source(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None


def scan_covers_markers(
    e2e_dir: Path = E2E_DIR,
) -> tuple[frozenset[str], tuple[str, ...]]:
    """Return (cell ids declared by a `covers` marker anywhere in the source tree,
    paths that could not be parsed). Independent of deselection, of env-gated
    opt-ins, and of optional dependencies, because nothing is imported."""
    parsed = tuple(
        (path, _read_covers_ids(path))
        for path in sorted(e2e_dir.rglob("*.py"))
        if _scannable(path)
    )
    return (
        frozenset(
            cell_id for _, ids in parsed if ids is not None for cell_id in ids
        ),
        tuple(str(path) for path, ids in parsed if ids is None),
    )


class _UiCoveredCell(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    spec: str
    test: str


class _UiDeclaration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    covers: tuple[_UiCoveredCell, ...] = ()


@dataclass(frozen=True, slots=True)
class UiDeclarations:
    """Cells the Playwright suite claims, split by whether the claim still holds."""

    ids: frozenset[str]
    unresolved: tuple[str, ...]


_TITLED_CALL = re.compile(
    r"\btest(?:\.(?:describe|only|skip|fixme|serial|parallel|step))*\s*\(\s*"
)
_TITLE_LITERAL = re.compile(r"(['\"`])((?:\\.|(?!\1).)*?)\1\s*[,)]", re.DOTALL)
_INTERPOLATION = re.compile(r"\$\{[^{}]*\}")

_STRING_OR_COMMENT = re.compile(
    r"""
      (?P<string> '(?:\\.|[^'\\\n])*'
                | "(?:\\.|[^"\\\n])*"
                | `(?:\\.|[^`\\])*` )
    | (?P<comment> //[^\n]* | /\*.*?\*/ )
    """,
    re.VERBOSE | re.DOTALL,
)


def _without_comments(source: str) -> str:
    """The source with its comments blanked out, so a commented-out test cannot
    back a declaration. String literals are matched by the same pass and kept
    intact, so a `//` inside a title (a URL, say) is never mistaken for a comment
    opener; a false negative there would be worse than the staleness this catches."""
    return _STRING_OR_COMMENT.sub(
        lambda match: match.group("string") or " ", source
    )


@dataclass(frozen=True, slots=True)
class _SpecTitles:
    """What a spec's test titles can be matched against: the ones written out in
    full, and a pattern per interpolated title covering the titles it can produce."""

    literal: frozenset[str]
    patterns: tuple[re.Pattern[str], ...]

    def covers(self, title: str) -> bool:
        return title in self.literal or any(
            pattern.fullmatch(title) for pattern in self.patterns
        )


def _title_pattern(template: str) -> re.Pattern[str] | None:
    """A pattern for the titles an interpolated title can produce, or None when it
    is all interpolation and would therefore match anything."""
    segments = tuple(_INTERPOLATION.split(template))
    if not any(segment.strip() for segment in segments):
        return None
    return re.compile(".*".join(re.escape(segment) for segment in segments), re.DOTALL)


def _spec_titles(spec_source: str) -> _SpecTitles:
    """Every test title the spec can produce, read from the whole `test` family.

    A first argument that is not a string literal contributes nothing: that covers
    `test.skip(condition, reason)`, which shares its name with the titled form, and
    a title assembled from variables, which cannot be matched by text at all. A
    declaration naming one of those fails loudly rather than being waved through."""
    source = _without_comments(spec_source)
    titles = tuple(
        (literal.group(1), literal.group(2))
        for call in _TITLED_CALL.finditer(source)
        if (literal := _TITLE_LITERAL.match(source, call.end())) is not None
    )
    return _SpecTitles(
        literal=frozenset(
            text for quote, text in titles if not (quote == "`" and "${" in text)
        ),
        patterns=tuple(
            pattern
            for quote, text in titles
            if quote == "`" and "${" in text
            if (pattern := _title_pattern(text)) is not None
        ),
    )


def _unresolved_reason(cell: _UiCoveredCell, ui_dir: Path) -> str | None:
    """Why this row no longer resolves against the suite, or None when it holds.

    The check is per declaration, never per file: an interpolated title elsewhere
    in the spec is irrelevant unless it could itself have produced this title, so
    renaming a literal test still fails even when a dynamic sibling sits beside
    it."""
    spec = ui_dir / cell.spec
    if not spec.is_file():
        return f"{cell.id}: spec {cell.spec} does not exist"
    if _spec_titles(spec.read_text(encoding="utf-8")).covers(cell.test):
        return None
    return f"{cell.id}: {cell.spec} has no test titled {cell.test!r}"


def load_ui_declarations(path: Path = UI_DECLARATION_FILE) -> UiDeclarations:
    """Cells the TypeScript Playwright suite declares it covers, each checked
    against the spec and test title it names. A row whose spec or title no longer
    exists is dropped from the numerator and reported, so renaming or deleting a
    UI test cannot leave a cell counted forever. An id that resolves but is not in
    the registry lands in the covered set and surfaces as an orphan marker,
    exactly like a typo in a pytest marker."""
    if not path.is_file():
        return UiDeclarations(frozenset(), ())
    declaration = _UiDeclaration.model_validate(yaml.safe_load(path.read_text()) or {})
    checked = tuple(
        (cell, _unresolved_reason(cell, path.parent)) for cell in declaration.covers
    )
    return UiDeclarations(
        ids=frozenset(cell.id for cell, reason in checked if reason is None),
        unresolved=tuple(
            reason for _, reason in checked if reason is not None
        ),
    )


class _CoversSink:
    """Pytest plugin: after collection, capture every cell id declared via
    @pytest.mark.covers(...), plus any nodes that failed to import."""

    def __init__(self) -> None:
        self.covered_ids: frozenset[str] = frozenset()
        self.collection_errors: tuple[str, ...] = ()

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        marker_args: tuple[tuple[object, ...], ...] = tuple(
            marker.args
            for item in session.items
            for marker in item.iter_markers(name="covers")
        )
        self.covered_ids = frozenset(
            arg for args in marker_args for arg in args if isinstance(arg, str)
        )

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
class CoveredIds:
    ids: frozenset[str]
    collection_errors: tuple[str, ...]
    stale_ui_declarations: tuple[str, ...]


def covered_ids(
    e2e_dir: Path = E2E_DIR,
    ui_declaration: Path = UI_DECLARATION_FILE,
) -> CoveredIds:
    """The numerator and everything that undermines it: every cell id declared in
    the source, plus the ones only a live import can resolve, plus the TypeScript
    suite's still-resolving declarations; every node that failed to parse or to
    import; and every UI declaration that no longer points at a real test."""
    scanned, unparseable = scan_covers_markers(e2e_dir)
    collected, import_errors = collect_covered_ids(e2e_dir)
    ui = load_ui_declarations(ui_declaration)
    return CoveredIds(
        ids=scanned | collected | ui.ids,
        collection_errors=(*unparseable, *import_errors),
        stale_ui_declarations=ui.unresolved,
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
    collection_errors: tuple[str, ...]
    stale_ui_declarations: tuple[str, ...] = ()

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
    stale_ui_declarations: tuple[str, ...] = (),
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
        stale_ui_declarations=stale_ui_declarations,
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
    stale = (
        (
            f"\n{len(report.stale_ui_declarations)} UI declaration(s) no longer point at a "
            f"real test and are not counted (reconcile tests/e2e/ui/coverage.yaml):\n  "
            + "\n  ".join(report.stale_ui_declarations),
        )
        if report.stale_ui_declarations
        else ()
    )
    warning = (
        (
            f"\nWARNING: {len(report.collection_errors)} node(s) failed to parse or import, "
            f"so coverage may undercount:\n  "
            + "\n  ".join(report.collection_errors),
        )
        if report.collection_errors
        else ()
    )
    return "\n".join((*lines, *orphans, *stale, *warning))


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
        "stale_ui_declarations": list(report.stale_ui_declarations),
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
            "# HELP litellm_e2e_coverage_collection_errors Pytest nodes that failed during collection.",
            "# TYPE litellm_e2e_coverage_collection_errors gauge",
            f"litellm_e2e_coverage_collection_errors {len(report.collection_errors)}",
            "# HELP litellm_e2e_coverage_stale_ui_declarations UI coverage declarations whose spec or test title no longer exists.",
            "# TYPE litellm_e2e_coverage_stale_ui_declarations gauge",
            f"litellm_e2e_coverage_stale_ui_declarations {len(report.stale_ui_declarations)}",
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


class CliArgs(BaseModel):
    format: Literal["text", "json", "prometheus", "loki"]
    strict: bool
    fail_on_collection_errors: bool


def exit_code(report: CoverageReport, args: CliArgs) -> int:
    """Non-zero when the run found something the operator asked to fail on: a claim
    the tree does not support (an unknown cell id or a dead UI declaration) under
    --strict, or a node whose cells could not be read under
    --fail-on-collection-errors."""
    if args.strict and (report.orphan_markers or report.stale_ui_declarations):
        return 1
    if args.fail_on_collection_errors and report.collection_errors:
        return 1
    return 0


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
        help=(
            "Exit non-zero if markers outside the registry are found, or if a UI "
            "coverage declaration no longer points at a real Playwright test."
        ),
    )
    parser.add_argument(
        "--fail-on-collection-errors",
        action="store_true",
        help="Exit non-zero if pytest collection errors are found.",
    )
    args = CliArgs.model_validate(vars(parser.parse_args()))
    cells = load_registry()
    covered = covered_ids()
    report = compute_coverage(
        cells, covered.ids, covered.collection_errors, covered.stale_ui_declarations
    )
    output = {
        "text": render,
        "json": render_json,
        "prometheus": render_prometheus,
        "loki": render_loki,
    }[args.format](report)
    print(output)  # noqa: T201  # CLI entrypoint output
    return exit_code(report, args)


if __name__ == "__main__":
    sys.exit(main())
