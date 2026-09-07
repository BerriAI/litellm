from __future__ import annotations

import os
import re
import sys
from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ...shared.reporting.models import SURFACES, CaseResult, RunStatus, SdkFunction, Surface
from ...shared.reporting.rendering import ReportSection
from ...shared.reporting.strategy import NotImplementedCaseSpec, SkippedCaseSpec
from ...shared.tracing.steps import (
    PipelineStep,
    TraceContract,
    TraceDiff,
    TraceMapping,
    trace_depths,
    trace_diff,
)

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

    id: int
    parent_id: int | None
    span: str
    raw: str

    def step(self) -> PipelineStep:
        return PipelineStep(self.id, self.parent_id, self.span, self.raw)


class TraceMappingArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    span: str
    python: str | None
    rust: str | None


class TraceComparisonArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    surface: Surface
    sdk_function: SdkFunction
    scenario: str
    mode: Literal["sync", "async"]
    mappings: tuple[TraceMappingArtifact, ...]
    python: tuple[TraceEventArtifact, ...]
    rust: tuple[TraceEventArtifact, ...]
    python_unmatched: int
    unordered_children_of: frozenset[str]
    python_error: str | None = None
    rust_error: str | None = None

    @classmethod
    def from_traces(
        cls,
        *,
        surface: Surface,
        sdk_function: SdkFunction,
        scenario: str,
        mode: Literal["sync", "async"],
        mappings: Sequence[TraceMapping],
        contract: TraceContract,
        python: Sequence[PipelineStep],
        rust: Sequence[PipelineStep],
        python_unmatched: int,
        python_error: str | None = None,
        rust_error: str | None = None,
    ) -> TraceComparisonArtifact:
        return cls(
            surface=surface,
            sdk_function=sdk_function,
            scenario=scenario,
            mode=mode,
            mappings=tuple(
                TraceMappingArtifact(
                    span=item.span,
                    python=item.python.pattern if item.python else None,
                    rust=item.rust,
                )
                for item in mappings
            ),
            python=tuple(
                TraceEventArtifact(id=step.id, parent_id=step.parent_id, span=step.span, raw=step.raw)
                for step in python
            ),
            rust=tuple(
                TraceEventArtifact(id=step.id, parent_id=step.parent_id, span=step.span, raw=step.raw)
                for step in rust
            ),
            python_unmatched=python_unmatched,
            unordered_children_of=contract.unordered_children_of,
            python_error=python_error,
            rust_error=rust_error,
        )

    def python_steps(self) -> tuple[PipelineStep, ...]:
        return tuple(event.step() for event in self.python)

    def rust_steps(self) -> tuple[PipelineStep, ...]:
        return tuple(event.step() for event in self.rust)

    def diff(self) -> TraceDiff:
        return trace_diff(
            self.python_steps(),
            self.rust_steps(),
            tuple(
                TraceMapping(
                    item.span,
                    re.compile(item.python) if item.python is not None else None,
                    item.rust,
                )
                for item in self.mappings
            ),
            TraceContract(self.unordered_children_of),
        )

    def exact_match(self) -> bool:
        return self.diff().matches

    def has_errors(self) -> bool:
        return self.python_error is not None or self.rust_error is not None

    def contract_matches(self) -> bool:
        if self.has_errors():
            return False
        return self.diff().matches


def _split_raw(raw: str) -> tuple[str, str]:
    location, separator, name = raw.partition(" ")
    if separator:
        return name, location
    return raw, ""


def _python_line(index: int, step: PipelineStep, depth: int, exclusive: frozenset[str]) -> str:
    name: Final = _split_raw(step.raw)[0]
    location: Final = _split_raw(step.raw)[1]
    suffix: Final = f"  ({location})" if location else ""
    marker: Final = "  [python only]" if step.span in exclusive else ""
    return _paint(f"{index} {'  ' * depth}{name}{suffix}{marker}", "cyan")


def _python_lines(steps: tuple[PipelineStep, ...], exclusive: frozenset[str]) -> str:
    depths: Final = trace_depths(steps)
    lines: Final = tuple(
        _python_line(index, step, depths[step.id], exclusive) for index, step in enumerate(steps, start=1)
    )
    return f"{_paint('PYTHON', 'cyan')} ({len(steps)} steps)\n" + ("\n".join(lines) if lines else "(empty)")


def _python_references(steps: tuple[PipelineStep, ...]) -> dict[tuple[str, int], str]:
    references: dict[tuple[str, int], str] = {}
    occurrences: dict[str, int] = {}
    for index, step in enumerate(steps, start=1):
        name = _split_raw(step.raw)[0]
        occurrence = occurrences.get(step.span, 0) + 1
        occurrences[step.span] = occurrence
        references[(step.span, occurrence)] = f"{index} {name}"
    return references


def _rust_line(
    step: PipelineStep,
    depth: int,
    occurrence: int,
    references: dict[tuple[str, int], str],
) -> str:
    span: Final = _paint(step.span, "yellow")
    key: Final = (step.span, occurrence)
    reference: Final = (
        _paint(references[key], "cyan") if key in references else _paint("[rust only]", "yellow")
    )
    suffix: Final = f"#{occurrence}" if occurrence > 1 else ""
    return f"{'  ' * depth}{span}{suffix}  ->  {reference}"


def _rust_lines(steps: tuple[PipelineStep, ...], references: dict[tuple[str, int], str]) -> str:
    depths: Final = trace_depths(steps)
    occurrences: dict[str, int] = {}
    lines: list[str] = []
    for step in steps:
        occurrence = occurrences.get(step.span, 0) + 1
        occurrences[step.span] = occurrence
        lines.append(_rust_line(step, depths[step.id], occurrence, references))
    return f"{_paint('RUST', 'yellow')} ({len(steps)} steps)\n" + ("\n".join(lines) if lines else "(empty)")


def _state_text(state: str, *, good: bool) -> str:
    return _paint(state, "green" if good else "red")


def _contract_line(artifact: TraceComparisonArtifact) -> str:
    matches: Final = artifact.contract_matches()
    status: Final = _state_text("PASS" if matches else "FAIL", good=matches)
    if artifact.python_error or artifact.rust_error:
        return f"Contract: {status}"
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


def _unseen_mappings(
    artifact: TraceComparisonArtifact,
    python: tuple[PipelineStep, ...],
    rust: tuple[PipelineStep, ...],
) -> tuple[str, ...]:
    return artifact.diff().missing_mappings


def _comparison_status_lines(
    artifact: TraceComparisonArtifact,
    python: tuple[PipelineStep, ...],
    rust: tuple[PipelineStep, ...],
) -> tuple[str, ...]:
    diff: Final = artifact.diff()
    exact_match: Final = artifact.exact_match()
    if artifact.has_errors():
        return (*_error_lines(artifact), _contract_line(artifact))
    unseen: Final = _unseen_mappings(artifact, python, rust)
    unseen_line: Final[tuple[str, ...]] = (f"Unseen mappings: {', '.join(unseen)}",) if unseen else ()
    drift_lines: Final[tuple[str, ...]] = (
        (_state_text("Same steps, order, and nesting", good=True),)
        if exact_match
        else (
            _paint(f"Python only: {', '.join(diff.python_only) or 'none'}", "cyan"),
            _paint(f"Rust only: {', '.join(diff.rust_only) or 'none'}", "yellow"),
            f"First difference: {diff.first_difference or 'none'}",
            f"Python frames outside mapping: {artifact.python_unmatched}",
        )
    )
    return (
        f"Trace: {_state_text('MATCH' if exact_match else 'DRIFT', good=exact_match)}",
        *drift_lines,
        *unseen_line,
        _contract_line(artifact),
    )


def _render_comparison(artifact: TraceComparisonArtifact) -> str:
    python: Final = artifact.python_steps()
    rust: Final = artifact.rust_steps()
    diff: Final = artifact.diff()
    python_exclusive: Final = frozenset(item.span for item in artifact.mappings if item.rust is None)
    status_lines: Final = _comparison_status_lines(artifact, python, rust)
    return "\n\n".join(
        (
            _python_lines(python, python_exclusive | frozenset(diff.python_only)),
            _rust_lines(rust, _python_references(python)),
            "\n".join(status_lines),
        )
    )


def _mode(nodeid: str) -> str:
    if "[" in nodeid:
        return nodeid.rsplit("[", 1)[-1].removesuffix("]")
    head, _, tail = nodeid.rpartition(":")
    return tail if head else "unknown mode"


def _scenario(nodeid: str) -> str:
    parts: Final = nodeid.split(":")
    return parts[-2] if len(parts) >= 5 else "default"


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
    label: Final = f"Scenario: {_scenario(nodeid)} / Mode: {_mode(nodeid)}"
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
