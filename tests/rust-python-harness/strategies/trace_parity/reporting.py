from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from ...shared.reporting.models import CaseResult, RunStatus, SdkFunction, Surface
from ...shared.reporting.rendering import ReportSection
from ...shared.reporting.strategy import NotImplementedCaseSpec, SkippedCaseSpec
from ...shared.tracing.profiler import FunctionTraceEvent
from ...shared.tracing.steps import TraceDiff, trace_diff

TRACE_COMPARISON_ARTIFACT: Final = "trace_comparison"


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
        )

    def python_events(self) -> tuple[FunctionTraceEvent, ...]:
        return tuple(event.event() for event in self.python)

    def rust_events(self) -> tuple[FunctionTraceEvent, ...]:
        return tuple(event.event() for event in self.rust)

    def contract_matches(self) -> bool:
        python: Final = self.python_events()
        rust: Final = self.rust_events()
        diff: Final = trace_diff(python, rust)
        return (
            not self.python_issues
            and not self.rust_issues
            and (not self.requires_matching_steps or diff.matches)
            and (not self.requires_exact_trace or python == rust)
        )


def _trace_lines(
    label: str,
    events: tuple[FunctionTraceEvent, ...],
    exclusive: frozenset[str],
) -> str:
    lines: Final = tuple(
        f"{'  ' * event.depth}{event.function}{f'  [{label} only]' if event.function in exclusive else ''}"
        for event in events
    )
    return f"{label} ({len(events)} steps)\n" + ("\n".join(lines) if lines else "(empty)")


def _shared_nesting_matches(
    python: tuple[FunctionTraceEvent, ...],
    rust: tuple[FunctionTraceEvent, ...],
) -> bool:
    rust_depths: Final = {event.function: event.depth for event in rust}
    shared: Final = tuple(event for event in python if event.function in rust_depths)
    return bool(shared) and all(event.depth == rust_depths[event.function] for event in shared)


def _contract_line(
    artifact: TraceComparisonArtifact,
    python: tuple[FunctionTraceEvent, ...],
    rust: tuple[FunctionTraceEvent, ...],
    diff: TraceDiff,
) -> str:
    status: Final = "PASS" if artifact.contract_matches() else "FAIL"
    if python == rust or status == "FAIL":
        return f"Contract: {status}"
    if not artifact.requires_matching_steps:
        return f"Contract: {status} (path drift is allowed for this case)"
    if not artifact.requires_exact_trace and diff.matches:
        return f"Contract: {status} (nesting drift is allowed for this case)"
    return f"Contract: {status}"


def _render_comparison(artifact: TraceComparisonArtifact) -> str:
    python: Final = artifact.python_events()
    rust: Final = artifact.rust_events()
    diff: Final = trace_diff(python, rust)
    exact_match: Final = python == rust
    shared: Final = frozenset(event.function for event in python) & frozenset(event.function for event in rust)
    drift_lines: Final = (
        ("Same steps, order, and nesting",)
        if exact_match
        else (
            f"Shared order: {'MATCH' if diff.shared_order_matches else 'DRIFT' if shared else 'NO SHARED STEPS'}",
            f"Shared nesting: {'MATCH' if _shared_nesting_matches(python, rust) else 'DRIFT' if shared else 'NO SHARED STEPS'}",
            f"Python only: {', '.join(diff.python_only) or 'none'}",
            f"Rust only: {', '.join(diff.rust_only) or 'none'}",
        )
    )
    issue_lines: Final = tuple(
        f"{engine} issue: {issue}"
        for engine, issues in (("Python", artifact.python_issues), ("Rust", artifact.rust_issues))
        for issue in issues
    )
    return "\n\n".join(
        (
            f"Trace comparison: {artifact.surface}/{artifact.sdk_function} ({artifact.mode})",
            _trace_lines("python", python, frozenset(diff.python_only)),
            _trace_lines("rust", rust, frozenset(diff.rust_only)),
            "\n".join(
                (
                    f"Trace: {'MATCH' if exact_match else 'DRIFT'}",
                    *drift_lines,
                    *issue_lines,
                    _contract_line(artifact, python, rust, diff),
                )
            ),
        )
    )


def _mode(nodeid: str) -> str:
    return nodeid.rsplit("[", 1)[-1].removesuffix("]") if "[" in nodeid else "unknown mode"


def _unavailable(result: CaseResult, nodeid: str, status: RunStatus) -> str:
    return (
        f"Trace comparison: {result.case.surface}/{result.case.sdk_function} ({_mode(nodeid)})\n"
        "Trace: NOT AVAILABLE\n"
        f"Test outcome: {status.value}"
    )


def _render_artifact(body: str) -> str:
    try:
        artifact: Final = TraceComparisonArtifact.model_validate_json(body)
    except ValidationError as error:
        return f"Trace comparison artifact is invalid: {error}"
    return _render_comparison(artifact)


def _node_blocks(result: CaseResult, nodeid: str, status: RunStatus) -> tuple[str, ...]:
    artifacts: Final = tuple(
        artifact for artifact in result.artifacts.get(nodeid, ()) if artifact.kind == TRACE_COMPARISON_ARTIFACT
    )
    if artifacts:
        return tuple(_render_artifact(artifact.body) for artifact in artifacts)
    return (_unavailable(result, nodeid, status),)


def _inert_block(result: CaseResult) -> str | None:
    spec: Final = result.case.spec
    if isinstance(spec, NotImplementedCaseSpec):
        return (
            f"Trace comparison: {result.case.surface}/{result.case.sdk_function}\n"
            "Trace: NOT IMPLEMENTED\n"
            f"Reason: {spec.reason}"
        )
    if isinstance(spec, SkippedCaseSpec):
        return (
            f"Trace comparison: {result.case.surface}/{result.case.sdk_function}\n"
            "Trace: SKIPPED\n"
            f"Reason: {spec.reason}"
        )
    return None


def render_trace_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    outcome_blocks: Final = tuple(
        block
        for result in results
        for nodeid, status in result.outcomes.items()
        for block in _node_blocks(result, nodeid, status)
    )
    inert_blocks: Final = tuple(block for result in results if (block := _inert_block(result)) is not None)
    blocks: Final = (*outcome_blocks, *inert_blocks)
    visible: Final = blocks or ("No runnable trace comparisons",)
    return (ReportSection("Trace comparisons", visible),)
