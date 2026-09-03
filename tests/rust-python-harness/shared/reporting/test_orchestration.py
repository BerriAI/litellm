from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from .models import CaseResult, Coverage, HarnessCase, HarnessRun, RunStatus, Strategy
from .orchestration import run_strategies
from .rendering import ReportSection, StrategyRenderer, render_outcomes
from .strategy import ModuleCaseSpec, StrategyDefinition, UpdateCallback
from .ui import _HarnessOutputFilter, _final_report


def _run_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    del repo_root, runner_args
    run: Final = HarnessRun.from_cases(cases)
    for case in cases:
        result: Final = run.results[case.key]
        nodeid: Final = f"check:{case.strategy_id}:{case.sdk_function}"
        result.collected.add(nodeid)
        failed: Final = isinstance(case.spec, ModuleCaseSpec) and case.spec.module == "fail"
        result.record(nodeid, RunStatus.FAILED if failed else RunStatus.PASSED)
        if failed:
            run.failures.append((nodeid, "comparison failed"))
        on_update(run)
    run.finished_at = monotonic()
    return int(bool(run.failures)), run


def _strategy(name: str, module: str, *, render: StrategyRenderer = render_outcomes) -> Strategy:
    definition: Final = StrategyDefinition(Path.cwd(), ModuleCaseSpec, _run_cases, render)
    case: Final = HarnessCase(
        strategy_id=name,
        strategy_label=name,
        sdk_function="ocr",
        spec=ModuleCaseSpec(coverage=Coverage.COMPLETE, module=module),
    )
    return Strategy(1, name, name, "", Path.cwd(), (case,), definition)


def test_combines_strategy_reports_and_delegates_rendering() -> None:
    strategies: Final = (_strategy("first", "fail"), _strategy("second", "pass"))

    code, report = run_strategies(strategies, Path.cwd(), lambda _: None)

    assert code == 1
    assert report.results["first:ocr"].status is RunStatus.FAILED
    assert report.results["second:ocr"].status is RunStatus.PASSED
    assert report.completed_tests == 2
    final_report: Final = _final_report(report, code, strategies)
    assert "Status: FAILED" in final_report
    assert "first\n- sdk/ocr: failed, 1/1 tests, complete coverage" in final_report
    assert "second\n- sdk/ocr: passed, 1/1 tests, complete coverage" in final_report
    assert "Failures (showing 1 of 1)" in final_report
    assert "Port confidence" not in final_report
    assert "Slowest tests" not in final_report


def test_strategy_can_replace_the_generic_result_view() -> None:
    def render_custom(_results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
        return (ReportSection("Custom comparison", ("domain-owned diff",)),)

    strategy: Final = _strategy("custom", "pass", render=render_custom)
    code, report = run_strategies((strategy,), Path.cwd(), lambda _: None)

    final_report: Final = _final_report(report, code, (strategy,))
    assert "Custom comparison\ndomain-owned diff" in final_report
    assert "sdk/ocr" not in final_report


def test_harness_output_filter_suppresses_only_ocr_cost_warnings() -> None:
    output_filter: Final = _HarnessOutputFilter()
    ocr_cost_warning: Final = logging.LogRecord(
        "LiteLLM",
        logging.WARNING,
        "/repo/litellm/cost_calculator.py",
        1953,
        "OCR cost: model=%s has no pricing",
        ("example",),
        None,
    )
    other_warning: Final = logging.LogRecord(
        "LiteLLM",
        logging.WARNING,
        "/repo/litellm/main.py",
        1,
        "Provider warning",
        (),
        None,
    )

    assert output_filter.filter(ocr_cost_warning) is False
    assert output_filter.filter(other_warning) is True
