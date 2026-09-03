from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final

import pytest

from ..shared.reporting.models import HarnessCase, Strategy
from ..shared.reporting.orchestration import run_strategies
from ..shared.reporting.strategy import SelectorCaseSpec, SuiteCaseSpec
from ..shared.reporting.ui import make_dashboard

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
COVERAGE_ROOT: Final = REPO_ROOT / "target" / "rust-python-harness"


def _coverage_pytest_args(output_root: Path = COVERAGE_ROOT) -> tuple[str, ...]:
    output_root.mkdir(parents=True, exist_ok=True)
    return (
        "--cov=litellm",
        "--cov-context=test",
        f"--cov-report=json:{output_root / 'python.json'}",
        f"--cov-report=xml:{output_root / 'python.xml'}",
        f"--cov-report=html:{output_root / 'python-html'}",
    )


def _grouped_cases(cases: Sequence[HarnessCase]) -> dict[str, tuple[HarnessCase, ...]]:
    grouped: dict[str, list[HarnessCase]] = {}
    for case in cases:
        grouped.setdefault(case.strategy_id, []).append(case)
    return {strategy_id: tuple(grouped_cases) for strategy_id, grouped_cases in grouped.items()}


def run_command(
    args: argparse.Namespace,
    strategies: Sequence[Strategy],
    cases: Sequence[HarnessCase],
) -> int:
    pytest_args = [*args.pytest_arg]
    if args.coverage:
        pytest_args.extend(_coverage_pytest_args())
    grouped: Final = _grouped_cases(cases)
    visible: Final = tuple(
        strategy for strategy in strategies if strategy.id in grouped
    )
    runners: Final = tuple(
        replace(strategy, cases=grouped[strategy.id]) for strategy in visible
    )
    dashboard = make_dashboard(
        visible,
        plain=args.plain,
        confidence_strategies=strategies,
    )
    with dashboard:
        exit_code, run = run_strategies(
            runners,
            REPO_ROOT,
            dashboard.update,
            pytest_args,
        )
        if exit_code != int(pytest.ExitCode.INTERRUPTED):
            dashboard.finish(run, exit_code)
    if args.coverage and (COVERAGE_ROOT / "python.json").exists():
        print(f"Python LOC heatmap: {COVERAGE_ROOT / 'python-html' / 'index.html'}")
        print(f"Machine-readable coverage: {COVERAGE_ROOT / 'python.json'}")
    return exit_code


def _case_detail(case: HarnessCase) -> str:
    spec = case.spec
    if isinstance(spec, SelectorCaseSpec) and spec.selectors:
        return ", ".join(spec.selectors)
    if isinstance(spec, SuiteCaseSpec) and spec.suite is not None:
        return spec.suite
    return "no test configured"


def list_command(
    args: argparse.Namespace,
    strategies: Sequence[Strategy],
    cases: Sequence[HarnessCase],
) -> int:
    del args
    selected_keys: Final = {case.key for case in cases}
    for strategy in strategies:
        selected: Final = tuple(
            case for case in strategy.cases if case.key in selected_keys
        )
        if not selected:
            continue
        print(f"{strategy.id:20} {strategy.label}")
        for case in selected:
            print(
                f"  {case.surface}/{case.sdk_function:12} {case.coverage.value:14} {_case_detail(case)}"
            )
    return 0


def check_command(
    args: argparse.Namespace,
    strategies: Sequence[Strategy],
    cases: Sequence[HarnessCase],
) -> int:
    grouped: Final = _grouped_cases(cases)
    sdk_functions: Final = frozenset(case.sdk_function for case in cases)
    failed = False
    for strategy in strategies:
        if strategy.id not in grouped:
            continue
        check = strategy.definition.check
        if check is None:
            print(f"{strategy.id}: no check defined")
            continue
        for report in check(sdk_functions, REPO_ROOT):
            if args.verbose:
                print(f"{strategy.id}/{report.sdk_function}: {'PASS' if report.passed else 'FAIL'}")
            for line in report.lines:
                print(line)
            failed = failed or not report.passed
    return 1 if failed else 0
