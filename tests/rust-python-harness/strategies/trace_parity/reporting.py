from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from ...shared.reporting.models import SURFACES, CaseResult, RunStatus, SdkFunction, Surface
from ...shared.reporting.rendering import ReportSection
from ...shared.reporting.strategy import NotImplementedCaseSpec, SkippedCaseSpec
from ...shared.tracing.steps import PipelineStep, trace_depths
from .models import TraceEngine

TRACE_ARTIFACT: Final = "trace"
TRACE_PARITY_HINT: Final = (
    "rebuild the native bridge with the trace-parity feature, e.g. `uvx maturin develop --features trace-parity`"
)

_COLORS: Final[dict[str, str]] = {"yellow": "33", "red": "31", "cyan": "36"}
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


class TraceArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: TraceEngine = "both"
    surface: Surface
    sdk_function: SdkFunction
    scenario: str
    python: tuple[TraceEventArtifact, ...]
    rust: tuple[TraceEventArtifact, ...]
    python_error: str | None = None
    rust_error: str | None = None

    @classmethod
    def from_traces(
        cls,
        *,
        engine: TraceEngine = "both",
        surface: Surface,
        sdk_function: SdkFunction,
        scenario: str,
        python: Sequence[PipelineStep],
        rust: Sequence[PipelineStep],
        python_error: str | None = None,
        rust_error: str | None = None,
    ) -> TraceArtifact:
        return cls(
            engine=engine,
            surface=surface,
            sdk_function=sdk_function,
            scenario=scenario,
            python=tuple(
                TraceEventArtifact(id=step.id, parent_id=step.parent_id, span=step.span, raw=step.raw)
                for step in python
            ),
            rust=tuple(
                TraceEventArtifact(id=step.id, parent_id=step.parent_id, span=step.span, raw=step.raw) for step in rust
            ),
            python_error=python_error,
            rust_error=rust_error,
        )

    def python_steps(self) -> tuple[PipelineStep, ...]:
        return tuple(event.step() for event in self.python)

    def rust_steps(self) -> tuple[PipelineStep, ...]:
        return tuple(event.step() for event in self.rust)

    def has_errors(self) -> bool:
        return self.python_error is not None or self.rust_error is not None


def _split_raw(raw: str) -> tuple[str, str]:
    location, separator, name = raw.partition(" ")
    if separator:
        return name, location
    return raw, ""


def _python_line(index: int, step: PipelineStep, depth: int) -> str:
    name: Final = _split_raw(step.raw)[0]
    location: Final = _split_raw(step.raw)[1]
    suffix: Final = f"  ({location})" if location else ""
    return _paint(f"{index} {'  ' * depth}{name}{suffix}", "cyan")


def _python_lines(steps: tuple[PipelineStep, ...]) -> str:
    depths: Final = trace_depths(steps)
    lines: Final = tuple(_python_line(index, step, depths[step.id]) for index, step in enumerate(steps, start=1))
    return f"{_paint('PYTHON', 'cyan')} ({len(steps)} steps)\n" + ("\n".join(lines) if lines else "(empty)")


def _rust_lines(steps: tuple[PipelineStep, ...]) -> str:
    depths: Final = trace_depths(steps)
    lines: Final = tuple(
        _paint(f"{index} {'  ' * depths[step.id]}{step.span}", "yellow") for index, step in enumerate(steps, 1)
    )
    return f"{_paint('RUST', 'yellow')} ({len(steps)} steps)\n" + ("\n".join(lines) if lines else "(empty)")


def _error_lines(artifact: TraceArtifact) -> tuple[str, ...]:
    lines: list[str] = []
    for engine, error in (("Python", artifact.python_error), ("Rust", artifact.rust_error)):
        if error is None:
            continue
        lines.append(_paint(f"{engine} error: {error}", "red"))
        if "trace-parity feature" in error:
            lines.append(f"hint: {TRACE_PARITY_HINT}")
    return tuple(lines)


def _render_trace(artifact: TraceArtifact) -> str:
    traces: tuple[str, ...]
    if artifact.engine == "python":
        traces = (_python_lines(artifact.python_steps()),)
    elif artifact.engine == "rust":
        traces = (_rust_lines(artifact.rust_steps()),)
    else:
        traces = (_python_lines(artifact.python_steps()), _rust_lines(artifact.rust_steps()))
    return "\n\n".join((*traces, *_error_lines(artifact)))


def _scenario(nodeid: str) -> str:
    parts: Final = nodeid.split(":")
    return parts[-1] if len(parts) >= 4 else "default"


def _unavailable(status: RunStatus) -> str:
    return f"Trace: NOT AVAILABLE\nTest outcome: {status.value}"


def _render_artifact(body: str) -> str:
    try:
        artifact: Final = TraceArtifact.model_validate_json(body)
    except ValidationError as error:
        return f"Trace artifact is invalid: {error}"
    return _render_trace(artifact)


def _scenario_section(result: CaseResult, nodeid: str, status: RunStatus) -> str:
    artifacts: Final = tuple(
        artifact for artifact in result.artifacts.get(nodeid, ()) if artifact.kind == TRACE_ARTIFACT
    )
    body: Final = (
        "\n\n".join(_render_artifact(artifact.body) for artifact in artifacts) if artifacts else _unavailable(status)
    )
    label: Final = f"Scenario: {_scenario(nodeid)}"
    return f"{label}\n{'-' * len(label)}\n\n{body}"


def _case_block(result: CaseResult) -> str:
    header: Final = f"Case: {result.case.sdk_function}"
    outcomes: Final = tuple(result.outcomes.items()) or (
        (nodeid, RunStatus.NOT_RUN) for nodeid in sorted(result.collected)
    )
    sections: Final = tuple(_scenario_section(result, nodeid, status) for nodeid, status in outcomes)
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
    return ReportSection(f"{surface.upper()} traces", blocks or ("No runnable traces",))


def render_trace_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    sections: Final = tuple(
        section for surface in SURFACES if (section := _surface_section(surface, results)) is not None
    )
    return sections or (ReportSection("Traces", ("No traces selected",)),)
