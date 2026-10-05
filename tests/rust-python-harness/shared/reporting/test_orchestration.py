from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from .models import CaseResult, Coverage, HarnessCase, HarnessRun, RunStatus, Strategy
from .orchestration import run_strategies
from .rendering import ReportSection, StrategyRenderer, render_case_outcome
from .strategy import (
    CaseDefinition,
    ModuleCaseSpec,
    NotImplementedCaseSpec,
    StrategyDefinition,
    UpdateCallback,
)
from .ui import HarnessOutputFilter, final_report


def _run_cases(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    del repo_root, runner_args
    run: Final = HarnessRun.from_cases(cases)
    for case in cases:
        _record_case(run, case, on_update)
    run.finished_at = monotonic()
    return int(bool(run.failures)), run


def _record_case(run: HarnessRun, case: HarnessCase, on_update: UpdateCallback) -> None:
    result: Final = run.results[case.key]
    nodeid: Final = f"check:{case.strategy_id}:{case.sdk_function}"
    result.collected.add(nodeid)
    failed: Final = isinstance(case.spec, ModuleCaseSpec) and case.spec.module == "fail"
    result.record(nodeid, RunStatus.FAILED if failed else RunStatus.PASSED)
    if failed:
        run.failures.append((nodeid, "comparison failed"))
    on_update(run)


def _render_test_results(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
    return (ReportSection("Test outcomes", tuple(render_case_outcome(result) for result in results)),)


def _strategy(name: str, module: str, *, render: StrategyRenderer = _render_test_results) -> Strategy:
    case_definition: Final = CaseDefinition("ocr", ModuleCaseSpec(coverage=Coverage.COMPLETE, module=module))
    definition: Final = StrategyDefinition(
        id=name,
        order=1,
        label=name,
        description="Example strategy",
        directory=Path.cwd(),
        runnable_spec=ModuleCaseSpec,
        cases=(case_definition,),
        run=_run_cases,
        render=render,
    )
    case: Final = HarnessCase(
        strategy_id=name,
        strategy_label=name,
        sdk_function="ocr",
        spec=case_definition.spec,
    )
    return Strategy(1, name, name, "", Path.cwd(), (case,), definition)


def test_combines_strategy_reports_and_delegates_rendering() -> None:
    strategies: Final = (_strategy("first", "fail"), _strategy("second", "pass"))

    code, report = run_strategies(strategies, Path.cwd(), lambda _: None)

    assert code == 1
    assert report.results["first:ocr"].status is RunStatus.FAILED
    assert report.results["second:ocr"].status is RunStatus.PASSED
    assert report.completed_checks == 2
    rendered: Final = final_report(report, code, strategies)
    assert "Result: FAILED" in rendered
    assert rendered.count("Test outcomes") == 2
    assert "- ocr: failed, 1/1 checks, complete coverage" in rendered
    assert "- ocr: passed, 1/1 checks, complete coverage" in rendered
    assert "Failures (showing 1 of 1)" in rendered
    assert "Port confidence" not in rendered
    assert "Slowest tests" not in rendered


def test_strategy_can_replace_the_generic_result_view() -> None:
    def render_custom(results: Sequence[CaseResult]) -> tuple[ReportSection, ...]:
        del results
        return (ReportSection("Custom comparison", ("domain-owned diff",)),)

    strategy: Final = _strategy("custom", "pass", render=render_custom)
    code, report = run_strategies((strategy,), Path.cwd(), lambda _: None)

    rendered: Final = final_report(report, code, (strategy,))
    assert "Custom comparison\ndomain-owned diff" in rendered
    assert "sdk/ocr" not in rendered


def test_report_separates_successful_execution_from_incomplete_coverage() -> None:
    runnable: Final = _strategy("mixed", "pass")
    unavailable: Final = HarnessCase(
        strategy_id="mixed",
        strategy_label="mixed",
        sdk_function="messages",
        spec=NotImplementedCaseSpec(reason="No Messages case is registered."),
    )
    code, executed = run_strategies((runnable,), Path.cwd(), lambda _: None)
    unavailable_run: Final = HarnessRun.from_cases((unavailable,))
    combined: Final = HarnessRun(
        results={**executed.results, **unavailable_run.results},
        started_at=executed.started_at,
        finished_at=executed.finished_at,
    )

    rendered: Final = final_report(combined, code, (runnable,))

    assert code == 0
    assert "Result: PASSED" in rendered
    assert "Harness support: 1/2 cases implemented" in rendered
    assert "Cases: 2 selected, 1 not implemented, 0 skipped" in rendered


def test_harness_output_filter_suppresses_expected_harness_warnings() -> None:
    output_filter: Final = HarnessOutputFilter()
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
    loop_warning: Final = logging.LogRecord(
        "LiteLLM",
        logging.WARNING,
        "/repo/litellm/litellm_core_utils/logging_worker.py",
        129,
        "LoggingWorker: event loop changed; carried %d pending and revived %d dequeued logging task(s) onto the new loop",
        (1, 0),
        None,
    )

    assert output_filter.filter(ocr_cost_warning) is False
    assert output_filter.filter(loop_warning) is False
    assert output_filter.filter(other_warning) is True
