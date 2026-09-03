from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ...shared.reporting.models import SURFACES, CaseResult, RunStatus, SdkFunction, Surface
from ...shared.reporting.rendering import ReportSection
from ...shared.reporting.strategy import NotImplementedCaseSpec, SkippedCaseSpec
from ...shared.tracing.profiler import FunctionTraceEvent
from ...shared.tracing.steps import trace_diff

TRACE_COMPARISON_ARTIFACT: Final = "trace_comparison"
TRACE_PARITY_HINT: Final = (
    "rebuild the native bridge with the trace-parity feature, e.g. `uvx maturin develop --features trace-parity`"
)

_COLORS: Final[dict[str, str]] = {"green": "32", "yellow": "33", "red": "31", "cyan": "36"}
_RESET: Final = "\033[0m"


def _paint(text: str, color: str) -> str:
    if not sys.stdout.isatty() or os.environ.get("NO_COLOR"):
        return text
    return f"\033[{_COLORS[color]}m{text}{_RESET}"


class TraceEventArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    function: str
    depth: int

    def event(self) -> FunctionTraceEvent:
        return FunctionTraceEvent(self.function, self.depth)


class TraceComparisonArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    surface: Surface
    sdk_function: SdkFunction
    mode: Literal["sync", "async"]
    python: tuple[TraceEventArtifact, ...]
    rust: tuple[TraceEventArtifact, ...]
    python_issues: tuple[str, ...]
    rust_issues: tuple[str, ...]
    requires_matching_steps: bool
    requires_exact_trace: bool
    python_error: str | None = None
    rust_error: str | None = None

    @classmethod
    def from_traces(
        cls,
        *,
        surface: Surface,
        sdk_function: SdkFunction,
        mode: Literal["sync", "async"],
        python: Sequence[FunctionTraceEvent],
        rust: Sequence[FunctionTraceEvent],
        python_issues: tuple[str, ...],
        rust_issues: tuple[str, ...],
        requires_matching_steps: bool,
        requires_exact_trace: bool,
        python_error: str | None = None,
        rust_error: str | None = None,
    ) -> TraceComparisonArtifact:
        return cls(
            surface=surface,
            sdk_function=sdk_function,
            mode=mode,
            python=tuple(TraceEventArtifact(function=event.function, depth=event.depth) for event in python),
            rust=tuple(TraceEventArtifact(function=event.function, depth=event.depth) for event in rust),
            python_issues=python_issues,
            rust_issues=rust_issues,
            requires_matching_steps=requires_matching_steps,
            requires_exact_trace=requires_exact_trace,
            python_error=python_error,
            rust_error=rust_error,
        )

    def python_events(self) -> tuple[FunctionTraceEvent, ...]:
        return tuple(event.event() for event in self.python)

    def rust_events(self) -> tuple[FunctionTraceEvent, ...]:
        return tuple(event.event() for event in self.rust)

    def has_errors(self) -> bool:
        return self.python_error is not None or self.rust_error is not None

    def contract_matches(self) -> bool:
        python: Final = self.python_events()
        rust: Final = self.rust_events()
        diff: Final = trace_diff(python, rust)
        return (
            not self.has_errors()
            and not self.python_issues
            and not self.rust_issues
            and (not self.requires_matching_steps or diff.matches)
            and (not self.requires_exact_trace or python == rust)
        )


def _trace_lines(
    label: str,
    events: tuple[FunctionTraceEvent, ...],
    exclusive: frozenset[str],
    color: str,
) -> str:
    lines: Final = tuple(
        _paint(
            f"{'  ' * event.depth}{event.function}{'  [' + label + ' only]' if event.function in exclusive else ''}",
            color,
        )
        for event in events
    )
    return f"{_paint(label.upper(), color)} ({len(events)} steps)\n" + ("\n".join(lines) if lines else "(empty)")


def _shared_nesting_matches(
    python: tuple[FunctionTraceEvent, ...],
    rust: tuple[FunctionTraceEvent, ...],
) -> bool:
    rust_depths: Final = {event.function: event.depth for event in rust}
    shared: Final = tuple(event for event in python if event.function in rust_depths)
    return bool(shared) and all(event.depth == rust_depths[event.function] for event in shared)


def _state_text(state: str, *, good: bool) -> str:
    return _paint(state, "green" if good else "red")


def _contract_line(artifact: TraceComparisonArtifact) -> str:
    matches: Final = artifact.contract_matches()
    status: Final = _state_text("PASS" if matches else "FAIL", good=matches)
    if artifact.python_error or artifact.rust_error:
        return f"Contract: {status}"
    python: Final = artifact.python_events()
    rust: Final = artifact.rust_events()
    diff: Final = trace_diff(python, rust)
    if python == rust:
        return f"Contract: {status}"
    if not artifact.requires_matching_steps:
        return f"Contract: {status} (path drift is allowed for this case)"
    if not artifact.requires_exact_trace and diff.matches:
        return f"Contract: {status} (nesting drift is allowed for this case)"
    return f"Contract: {status}"


def _error_lines(artifact: TraceComparisonArtifact) -> tuple[str, ...]:
    lines: list[str] = []
    for engine, error in (("Python", artifact.python_error), ("Rust", artifact.rust_error)):
        if error is None:
            continue
        lines.append(_paint(f"{engine} error: {error}", "red"))
        if "trace-parity feature" in error:
            lines.append(f"hint: {TRACE_PARITY_HINT}")
    return tuple(lines)


def _comparison_status_lines(
    artifact: TraceComparisonArtifact,
    python: tuple[FunctionTraceEvent, ...],
    rust: tuple[FunctionTraceEvent, ...],
) -> tuple[str, ...]:
    diff: Final = trace_diff(python, rust)
    exact_match: Final = python == rust
    shared: Final = frozenset(event.function for event in python) & frozenset(event.function for event in rust)
    if artifact.has_errors():
        return (*_error_lines(artifact), _contract_line(artifact))
    drift_lines: Final[tuple[str, ...]] = (
        (_state_text("Same steps, order, and nesting", good=True),)
        if exact_match
        else (
            f"Shared order: {_state_text('MATCH' if diff.shared_order_matches else 'DRIFT' if shared else 'NO SHARED STEPS', good=diff.shared_order_matches)}",
            f"Shared nesting: {_state_text('MATCH' if _shared_nesting_matches(python, rust) else 'DRIFT' if shared else 'NO SHARED STEPS', good=_shared_nesting_matches(python, rust))}",
            _paint(f"Python only: {', '.join(diff.python_only) or 'none'}", "cyan"),
            _paint(f"Rust only: {', '.join(diff.rust_only) or 'none'}", "yellow"),
        )
    )
    issue_lines: Final[tuple[str, ...]] = tuple(
        _paint(f"{engine} issue: {issue}", "red")
        for engine, issues in (("Python", artifact.python_issues), ("Rust", artifact.rust_issues))
        for issue in issues
    )
    return (
        f"Trace: {_state_text('MATCH' if exact_match else 'DRIFT', good=exact_match)}",
        *drift_lines,
        *issue_lines,
        _contract_line(artifact),
    )


def _render_comparison(artifact: TraceComparisonArtifact) -> str:
    python: Final = artifact.python_events()
    rust: Final = artifact.rust_events()
    diff: Final = trace_diff(python, rust)
    status_lines: Final = _comparison_status_lines(artifact, python, rust)
    return "\n\n".join(
        (
            _trace_lines("python", python, frozenset(diff.python_only), "cyan"),
            _trace_lines("rust", rust, frozenset(diff.rust_only), "yellow"),
            "\n".join(status_lines),
        )
    )


def _mode(nodeid: str) -> str:
    if "[" in nodeid:
        return nodeid.rsplit("[", 1)[-1].removesuffix("]")
    head, _, tail = nodeid.rpartition(":")
    return tail if head else "unknown mode"


def _unavailable(status: RunStatus) -> str:
    return f"Trace: NOT AVAILABLE\nTest outcome: {status.value}"


def _render_artifact(body: str) -> str:
    try:
        artifact: Final = TraceComparisonArtifact.model_validate_json(body)
    except ValidationError as error:
        return f"Trace comparison artifact is invalid: {error}"
    return _render_comparison(artifact)


def _mode_section(result: CaseResult, nodeid: str, status: RunStatus) -> str:
    artifacts: Final = tuple(
        artifact for artifact in result.artifacts.get(nodeid, ()) if artifact.kind == TRACE_COMPARISON_ARTIFACT
    )
    body: Final = (
        "\n\n".join(_render_artifact(artifact.body) for artifact in artifacts) if artifacts else _unavailable(status)
    )
    label: Final = f"Mode: {_mode(nodeid)}"
    return f"{label}\n{'-' * len(label)}\n\n{body}"


def _case_block(result: CaseResult) -> str:
    header: Final = f"Case: {result.case.sdk_function}"
    outcomes: Final = tuple(result.outcomes.items()) or (
        (nodeid, RunStatus.NOT_RUN) for nodeid in sorted(result.collected)
    )
    sections: Final = tuple(_mode_section(result, nodeid, status) for nodeid, status in outcomes)
    return "\n\n".join((f"{header}\n{'=' * len(header)}", *sections))


def _unavailable_block(title: str, lines: tuple[str, ...]) -> str | None:
    if not lines:
        return None
    return f"{title}\n{'-' * len(title)}\n" + "\n".join(lines)


def _surface_section(surface: Surface, results: Sequence[CaseResult]) -> ReportSection | None:
    selected: Final = tuple(result for result in results if result.case.surface == surface)
    if not selected:
        return None
    outcome_blocks: Final = tuple(_case_block(result) for result in selected if result.outcomes)
    not_implemented: Final = _unavailable_block(
        "Not implemented",
        tuple(
            f"- {result.case.sdk_function}: {spec.reason}"
            for result in selected
            if isinstance((spec := result.case.spec), NotImplementedCaseSpec)
        ),
    )
    skipped: Final = _unavailable_block(
        "Skipped",
        tuple(
            f"- {result.case.sdk_function}: {spec.reason}"
            for result in selected
            if isinstance((spec := result.case.spec), SkippedCaseSpec)
        ),
    )
    blocks: Final = (
        *outcome_blocks,
        *((not_implemented,) if not_implemented else ()),
        *((skipped,) if skipped else ()),
    )
    return ReportSection(f"{surface.upper()} trace comparisons", blocks or ("No runnable trace comparisons",))


def render_trace_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    sections: Final = tuple(
        section for surface in SURFACES if (section := _surface_section(surface, results)) is not None
    )
    return sections or (ReportSection("Trace comparisons", ("No trace comparisons selected",)),)
