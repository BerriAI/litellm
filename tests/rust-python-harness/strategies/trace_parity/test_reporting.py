from __future__ import annotations

from typing import Final

import pytest

from ...shared.reporting.models import CaseResult, Coverage, HarnessCase, ResultArtifact, RunStatus
from ...shared.reporting.strategy import ModuleCaseSpec, NotImplementedCaseSpec
from ...shared.tracing.profiler import FunctionTraceEvent
from . import reporting
from .reporting import TRACE_COMPARISON_ARTIFACT, TraceComparisonArtifact, render_trace_results


def _result(comparison: TraceComparisonArtifact) -> CaseResult:
    case: Final = HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function=comparison.sdk_function,
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface=comparison.surface,
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
    rust_error: str | None = None,
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
        rust_error=rust_error,
    )


def test_renderer_shows_matching_python_and_rust_paths() -> None:
    events: Final = (FunctionTraceEvent("ocr", 0), FunctionTraceEvent("http_request", 1))

    section: Final = render_trace_results((_result(_comparison(events, events)),))[0]
    report: Final = "\n\n".join(section.blocks)

    assert section.title == "SDK trace comparisons"
    assert "Case: ocr" in report
    assert "Mode: sync\n----------\n\nPYTHON (2 steps)\nocr\n  http_request" in report
    assert "RUST (2 steps)\nocr\n  http_request" in report
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


def test_unavailable_check_reports_mode_from_nodeid() -> None:
    case: Final = HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function="ocr",
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface="sdk",
    )
    result: Final = CaseResult(case=case)
    result.collected.add("trace:sdk:ocr:sync")
    result.record("trace:sdk:ocr:sync", RunStatus.ERROR)

    section: Final = render_trace_results((result,))[0]
    report: Final = "\n\n".join(section.blocks)

    assert "Case: ocr" in report
    assert "Mode: sync\n----------\n\nTrace: NOT AVAILABLE\nTest outcome: error" in report
    assert "unknown mode" not in report


def test_renderer_keeps_collected_trace_when_one_engine_errors() -> None:
    events: Final = (FunctionTraceEvent("ocr", 0), FunctionTraceEvent("http_request", 1))

    section: Final = render_trace_results(
        (_result(_comparison(events, (), rust_error="rust: native Rust bridge must include the trace-parity feature")),)
    )[0]
    report: Final = "\n\n".join(section.blocks)

    assert "PYTHON (2 steps)\nocr  [python only]\n  http_request  [python only]" in report
    assert "Rust error: rust: native Rust bridge must include the trace-parity feature" in report
    assert "hint: rebuild the native bridge with the trace-parity feature" in report
    assert "Contract: FAIL" in report


def test_renderer_groups_all_modes_under_one_case_header() -> None:
    case: Final = HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function="ocr",
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface="sdk",
    )
    result: Final = CaseResult(case=case)
    events: Final = (FunctionTraceEvent("ocr", 0),)
    for mode in ("sync", "async"):
        nodeid: Final = f"trace:sdk:ocr:{mode}"
        result.collected.add(nodeid)
        comparison: Final = TraceComparisonArtifact.from_traces(
            surface="sdk",
            sdk_function="ocr",
            mode=mode,  # type: ignore[arg-type]
            python=events,
            rust=events,
            python_issues=(),
            rust_issues=(),
            requires_matching_steps=True,
            requires_exact_trace=False,
        )
        result.record(
            nodeid,
            RunStatus.PASSED,
            artifacts=(ResultArtifact(TRACE_COMPARISON_ARTIFACT, comparison.model_dump_json()),),
        )

    section: Final = render_trace_results((result,))[0]

    assert len(section.blocks) == 1
    report: Final = section.blocks[0]
    assert report.count("Case: ocr") == 1
    assert "Mode: sync\n----------" in report
    assert "Mode: async\n-----------" in report


def test_renderer_colors_every_trace_line_in_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    events: Final = (FunctionTraceEvent("ocr", 0), FunctionTraceEvent("http_request", 1))
    monkeypatch.setattr(reporting.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)

    section: Final = render_trace_results((_result(_comparison(events, events)),))[0]
    report: Final = "\n\n".join(section.blocks)

    assert "\033[36mPYTHON\033[0m (2 steps)" in report
    assert "\033[36mocr\033[0m\n\033[36m  http_request\033[0m" in report
    assert "\033[33mRUST\033[0m (2 steps)" in report
    assert "\033[33mocr\033[0m\n\033[33m  http_request\033[0m" in report


def test_renderer_groups_cases_and_unavailable_entries_by_surface() -> None:
    events: Final = (FunctionTraceEvent("ocr", 0),)
    gateway_results: Final = tuple(
        CaseResult(
            case=HarnessCase(
                strategy_id="trace_parity",
                strategy_label="Trace parity",
                sdk_function=sdk_function,
                spec=NotImplementedCaseSpec(reason=f"No {sdk_function} case is registered."),
                surface="gateway",
            ),
            status=RunStatus.NOT_IMPLEMENTED,
        )
        for sdk_function in ("ocr", "messages")
    )

    sections: Final = render_trace_results((_result(_comparison(events, events)), *gateway_results))

    assert tuple(section.title for section in sections) == ("SDK trace comparisons", "GATEWAY trace comparisons")
    gateway_report: Final = "\n\n".join(sections[1].blocks)
    assert gateway_report.count("Not implemented") == 1
    assert "- ocr: No ocr case is registered." in gateway_report
    assert "- messages: No messages case is registered." in gateway_report
