from __future__ import annotations

from typing import Final

from ...shared.reporting.models import CaseResult, Coverage, HarnessCase, ResultArtifact, RunStatus
from ...shared.reporting.strategy import ModuleCaseSpec
from ...shared.tracing.profiler import FunctionTraceEvent
from .reporting import TRACE_COMPARISON_ARTIFACT, TraceComparisonArtifact, render_trace_results


def _result(comparison: TraceComparisonArtifact) -> CaseResult:
    case: Final = HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function=comparison.sdk_function,
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
    )
    result: Final = CaseResult(case=case)
    nodeid: Final = f"trace:sdk:{comparison.sdk_function}:{comparison.mode}"
    result.collected.add(nodeid)
    result.record(
        nodeid,
        RunStatus.PASSED,
        artifacts=(ResultArtifact(TRACE_COMPARISON_ARTIFACT, comparison.model_dump_json()),),
    )
    return result


def _comparison(
    python: tuple[FunctionTraceEvent, ...],
    rust: tuple[FunctionTraceEvent, ...],
    *,
    matching_steps: bool = True,
) -> TraceComparisonArtifact:
    return TraceComparisonArtifact.from_traces(
        surface="sdk",
        sdk_function="ocr",
        mode="sync",
        python=python,
        rust=rust,
        python_issues=(),
        rust_issues=(),
        requires_matching_steps=matching_steps,
        requires_exact_trace=False,
    )


def test_renderer_shows_matching_python_and_rust_paths() -> None:
    events: Final = (FunctionTraceEvent("ocr", 0), FunctionTraceEvent("http_request", 1))

    section: Final = render_trace_results((_result(_comparison(events, events)),))[0]
    report: Final = "\n\n".join(section.blocks)

    assert section.title == "Trace comparisons"
    assert "Trace comparison: sdk/ocr (sync)" in report
    assert "python (2 steps)\nocr\n  http_request" in report
    assert "rust (2 steps)\nocr\n  http_request" in report
    assert "Trace: MATCH" in report
    assert "Same steps, order, and nesting" in report


def test_renderer_marks_exclusive_steps_and_allowed_drift() -> None:
    python: Final = (FunctionTraceEvent("ocr", 0), FunctionTraceEvent("python_prepare", 1))
    rust: Final = (FunctionTraceEvent("ocr", 0), FunctionTraceEvent("rust_prepare", 1))

    section: Final = render_trace_results((_result(_comparison(python, rust, matching_steps=False)),))[0]
    report: Final = "\n\n".join(section.blocks)

    assert "python_prepare  [python only]" in report
    assert "rust_prepare  [rust only]" in report
    assert "Trace: DRIFT" in report
    assert "Python only: python_prepare" in report
    assert "Rust only: rust_prepare" in report
    assert "Contract: PASS (path drift is allowed for this case)" in report
