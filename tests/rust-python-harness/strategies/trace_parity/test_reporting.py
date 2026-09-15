from __future__ import annotations

from typing import Final, Literal

import pytest

from ...shared.reporting.models import CaseResult, Coverage, HarnessCase, ResultArtifact, RunStatus
from ...shared.reporting.strategy import ModuleCaseSpec, NotImplementedCaseSpec
from ...shared.tracing.steps import PipelineStep
from . import reporting
from .reporting import TRACE_ARTIFACT, TraceArtifact, render_trace_results


def _result(trace: TraceArtifact) -> CaseResult:
    case: Final = HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function=trace.sdk_function,
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface=trace.surface,
    )
    result: Final = CaseResult(case=case)
    nodeid: Final = f"trace:{trace.surface}:{trace.sdk_function}:{trace.scenario}"
    result.collected.add(nodeid)
    result.record(nodeid, RunStatus.PASSED, artifacts=(ResultArtifact(TRACE_ARTIFACT, trace.model_dump_json()),))
    return result


def _trace(
    python: tuple[PipelineStep, ...],
    rust: tuple[PipelineStep, ...],
    *,
    rust_error: str | None = None,
    engine: Literal["python", "rust", "both"] = "both",
    scenario: str = "sync-default",
) -> TraceArtifact:
    return TraceArtifact.from_traces(
        engine=engine,
        surface="sdk",
        sdk_function="ocr",
        scenario=scenario,
        python=python,
        rust=rust,
        rust_error=rust_error,
    )


def _events(*items: tuple[str, int, str | None]) -> tuple[PipelineStep, ...]:
    parents: dict[int, int] = {}
    steps: list[PipelineStep] = []
    for event_id, (span, depth, raw) in enumerate(items):
        parent_id = parents.get(depth - 1) if depth else None
        steps.append(PipelineStep(event_id, parent_id, span, raw if raw is not None else span))
        parents[depth] = event_id
    return tuple(steps)


def test_renderer_prints_python_and_rust_traces_independently() -> None:
    python: Final = _events(
        ("ocr", 0, "ocr/main.py:88 aocr"),
        ("python_prepare", 1, "prep.py:1 python_prepare"),
    )
    rust: Final = _events(("ocr", 0, None), ("rust_prepare", 1, None))

    section: Final = render_trace_results((_result(_trace(python, rust)),))[0]
    report: Final = "\n\n".join(section.blocks)

    assert section.title == "SDK traces"
    assert "PYTHON (2 steps)\n1 aocr  (ocr/main.py:88)\n2   python_prepare  (prep.py:1)" in report
    assert "RUST (2 steps)\n1 ocr\n2   rust_prepare" in report
    assert "python only" not in report
    assert "rust only" not in report
    assert " -> " not in report
    assert "Trace: MATCH" not in report
    assert "Trace: DRIFT" not in report
    assert "Contract:" not in report


@pytest.mark.parametrize(
    ("engine", "present", "absent"),
    (("python", "PYTHON (1 steps)", "RUST"), ("rust", "RUST (1 steps)", "PYTHON")),
)
def test_renderer_prints_only_selected_engine(engine: Literal["python", "rust"], present: str, absent: str) -> None:
    events: Final = _events(("ocr", 0, None))

    report: Final = "\n\n".join(render_trace_results((_result(_trace(events, events, engine=engine)),))[0].blocks)

    assert present in report
    assert absent not in report


def test_renderer_keeps_collected_trace_when_one_engine_errors() -> None:
    python: Final = _events(("ocr", 0, "ocr/main.py:88 aocr"))

    report: Final = "\n\n".join(
        render_trace_results(
            (_result(_trace(python, (), rust_error="rust: native Rust bridge must include the trace-parity feature")),)
        )[0].blocks
    )

    assert "PYTHON (1 steps)\n1 aocr  (ocr/main.py:88)" in report
    assert "Rust error: rust: native Rust bridge must include the trace-parity feature" in report
    assert "hint: rebuild the native bridge with the trace-parity feature" in report


def test_unavailable_trace_reports_scenario_from_nodeid() -> None:
    case: Final = HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function="ocr",
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface="sdk",
    )
    result: Final = CaseResult(case=case)
    result.collected.add("trace:sdk:ocr:async-error")
    result.record("trace:sdk:ocr:async-error", RunStatus.ERROR)

    report: Final = "\n\n".join(render_trace_results((result,))[0].blocks)

    assert "Scenario: async-error" in report
    assert "Trace: NOT AVAILABLE\nTest outcome: error" in report


def test_renderer_groups_scenarios_under_one_case_header() -> None:
    result: Final = _result(_trace(_events(("ocr", 0, None)), (), scenario="sync-default"))
    async_trace: Final = _trace((), _events(("ocr", 0, None)), scenario="async-default")
    nodeid: Final = "trace:sdk:ocr:async-default"
    result.collected.add(nodeid)
    result.record(nodeid, RunStatus.PASSED, artifacts=(ResultArtifact(TRACE_ARTIFACT, async_trace.model_dump_json()),))

    report: Final = render_trace_results((result,))[0].blocks[0]

    assert report.count("Case: ocr") == 1
    assert "Scenario: sync-default" in report
    assert "Scenario: async-default" in report


def test_renderer_colors_every_trace_line_in_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    events: Final = _events(("ocr", 0, "ocr/main.py:88 aocr"))
    monkeypatch.setattr(reporting.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)

    report: Final = "\n\n".join(render_trace_results((_result(_trace(events, events)),))[0].blocks)

    assert "\033[36mPYTHON\033[0m (1 steps)" in report
    assert "\033[36m1 aocr  (ocr/main.py:88)\033[0m" in report
    assert "\033[33mRUST\033[0m (1 steps)" in report
    assert "\033[33m1 ocr\033[0m" in report


def test_renderer_groups_unavailable_entries_by_surface() -> None:
    gateway_result: Final = CaseResult(
        case=HarnessCase(
            strategy_id="trace_parity",
            strategy_label="Trace parity",
            sdk_function="messages",
            spec=NotImplementedCaseSpec(reason="No messages case is registered."),
            surface="gateway",
        ),
        status=RunStatus.NOT_IMPLEMENTED,
    )

    sections: Final = render_trace_results((_result(_trace((), ())), gateway_result))

    assert tuple(section.title for section in sections) == ("SDK traces", "GATEWAY traces")
    assert "- messages: No messages case is registered." in "\n\n".join(sections[1].blocks)
