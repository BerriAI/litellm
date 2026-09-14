from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import pytest

from ...shared.reporting.models import CaseResult, Coverage, HarnessCase, ResultArtifact, RunStatus
from ...shared.reporting.strategy import ModuleCaseSpec, NotImplementedCaseSpec
from ...shared.tracing.steps import PipelineStep, TraceContract, TraceMapping, mapping
from . import reporting
from .reporting import TRACE_COMPARISON_ARTIFACT, TraceComparisonArtifact, render_trace_results

MAPPINGS: Final = (
    mapping(rust_span="ocr", python_frame=r"ocr/main\.py:\d+ a?ocr$"),
    mapping(rust_span="http_request", python_frame=r"AsyncHTTPHandler\.post$"),
)


def _result(comparison: TraceComparisonArtifact) -> CaseResult:
    case: Final = HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function=comparison.sdk_function,
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface=comparison.surface,
    )
    result: Final = CaseResult(case=case)
    nodeid: Final = f"trace:sdk:{comparison.sdk_function}:{comparison.scenario}:{comparison.mode}"
    result.collected.add(nodeid)
    result.record(
        nodeid,
        RunStatus.PASSED,
        artifacts=(ResultArtifact(TRACE_COMPARISON_ARTIFACT, comparison.model_dump_json()),),
    )
    return result


def _comparison(
    python: tuple[PipelineStep, ...],
    rust: tuple[PipelineStep, ...],
    *,
    mappings: Sequence[TraceMapping] = MAPPINGS,
    rust_error: str | None = None,
) -> TraceComparisonArtifact:
    return TraceComparisonArtifact.from_traces(
        surface="sdk",
        sdk_function="ocr",
        scenario="default",
        mode="sync",
        mappings=mappings,
        contract=TraceContract(),
        python=python,
        rust=rust,
        python_unmatched=796,
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


def test_renderer_shows_matching_python_and_rust_paths() -> None:
    rust: Final = _events(("ocr", 0, None), ("http_request", 1, None))
    python: Final = _events(
        ("ocr", 0, "ocr/main.py:88 aocr"),
        ("http_request", 1, "http_handler.py:673 AsyncHTTPHandler.post"),
    )

    section: Final = render_trace_results((_result(_comparison(python, rust)),))[0]
    report: Final = "\n\n".join(section.blocks)

    assert section.title == "SDK trace comparisons"
    assert "Case: ocr" in report
    assert "PYTHON (2 steps)\n1 aocr  (ocr/main.py:88)\n2   AsyncHTTPHandler.post  (http_handler.py:673)" in report
    assert "RUST (2 steps)\nocr  ->  1 aocr\n  http_request  ->  2 AsyncHTTPHandler.post" in report
    assert "Mapping (identifier -> span)" not in report
    assert "Trace: MATCH" in report
    assert "Same steps, order, and nesting" in report
    assert "Unseen mappings:" not in report


def test_renderer_reports_mappings_that_matched_nothing() -> None:
    events: Final = _events(("ocr", 0, None))

    section: Final = render_trace_results((_result(_comparison(events, events)),))[0]
    report: Final = "\n\n".join(section.blocks)

    assert "Unseen mappings: http_request" in report
    assert "Contract: FAIL" in report


def test_renderer_numbers_repeated_span_occurrences() -> None:
    mappings: Final = (MAPPINGS[0], MAPPINGS[1])
    rust: Final = _events(("ocr", 0, None), ("http_request", 1, None), ("http_request", 1, None))
    python: Final = _events(
        ("ocr", 0, "ocr/main.py:88 aocr"),
        ("http_request", 1, "http_handler.py:673 AsyncHTTPHandler.post"),
        ("http_request", 1, "http_handler.py:673 AsyncHTTPHandler.post"),
    )

    report: Final = "\n\n".join(render_trace_results((_result(_comparison(python, rust, mappings=mappings)),))[0].blocks)

    assert "http_request#2" in report


def test_renderer_accepts_declared_engine_specific_steps() -> None:
    mappings: Final = (
        *MAPPINGS[:1],
        mapping(span="python_prepare", python_frame=r"python_prepare$"),
        mapping(rust_span="rust_prepare"),
    )
    python: Final = _events(("ocr", 0, None), ("python_prepare", 1, "prep.py:1 python_prepare"))
    rust: Final = _events(("ocr", 0, None), ("rust_prepare", 1, None))

    section: Final = render_trace_results((_result(_comparison(python, rust, mappings=mappings)),))[0]
    report: Final = "\n\n".join(section.blocks)

    assert "2   python_prepare  (prep.py:1)  [python only]" in report
    assert "rust_prepare  ->  [rust only]" in report
    assert "Trace: MATCH" in report
    assert "Contract: PASS" in report


def test_unavailable_check_reports_mode_from_nodeid() -> None:
    case: Final = HarnessCase(
        strategy_id="trace_parity",
        strategy_label="Trace parity",
        sdk_function="ocr",
        spec=ModuleCaseSpec(coverage=Coverage.PARTIAL, module="example"),
        surface="sdk",
    )
    result: Final = CaseResult(case=case)
    result.collected.add("trace:sdk:ocr:default:sync")
    result.record("trace:sdk:ocr:default:sync", RunStatus.ERROR)

    section: Final = render_trace_results((result,))[0]
    report: Final = "\n\n".join(section.blocks)

    assert "Case: ocr" in report
    assert "Scenario: default / Mode: sync" in report
    assert "Trace: NOT AVAILABLE\nTest outcome: error" in report
    assert "unknown mode" not in report


def test_renderer_keeps_collected_trace_when_one_engine_errors() -> None:
    python: Final = _events(
        ("ocr", 0, "ocr/main.py:88 aocr"),
        ("http_request", 1, "http_handler.py:673 AsyncHTTPHandler.post"),
    )

    section: Final = render_trace_results(
        (_result(_comparison(python, (), rust_error="rust: native Rust bridge must include the trace-parity feature")),)
    )[0]
    report: Final = "\n\n".join(section.blocks)

    assert "PYTHON (2 steps)\n1 aocr  (ocr/main.py:88)  [python only]" in report
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
    events: Final = _events(("ocr", 0, None))
    modes: Final[tuple[Literal["sync", "async"], ...]] = ("sync", "async")
    for mode in modes:
        nodeid = f"trace:sdk:ocr:default:{mode}"
        result.collected.add(nodeid)
        comparison = TraceComparisonArtifact.from_traces(
            surface="sdk",
            sdk_function="ocr",
            scenario="default",
            mode=mode,
            mappings=MAPPINGS,
            contract=TraceContract(),
            python=events,
            rust=events,
            python_unmatched=0,
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
    assert "Scenario: default / Mode: sync" in report
    assert "Scenario: default / Mode: async" in report


def test_renderer_colors_every_trace_line_in_a_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    rust: Final = _events(("ocr", 0, None), ("http_request", 1, None))
    python: Final = _events(
        ("ocr", 0, "ocr/main.py:88 aocr"),
        ("http_request", 1, "http_handler.py:673 AsyncHTTPHandler.post"),
    )
    monkeypatch.setattr(reporting.sys.stdout, "isatty", lambda: True)
    monkeypatch.delenv("NO_COLOR", raising=False)

    section: Final = render_trace_results((_result(_comparison(python, rust)),))[0]
    report: Final = "\n\n".join(section.blocks)

    assert "\033[36mPYTHON\033[0m (2 steps)" in report
    assert "\033[36m1 aocr  (ocr/main.py:88)\033[0m" in report
    assert "\033[33mRUST\033[0m (2 steps)" in report
    assert "\033[33mocr\033[0m  ->  \033[36m1 aocr\033[0m" in report
    assert "\033[33mhttp_request\033[0m  ->  \033[36m2 AsyncHTTPHandler.post\033[0m" in report


def test_renderer_groups_cases_and_unavailable_entries_by_surface() -> None:
    events: Final = _events(("ocr", 0, None))
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
