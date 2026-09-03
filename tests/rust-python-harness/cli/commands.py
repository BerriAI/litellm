from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from ..shared.reporting.models import HarnessCase, SdkFunction, Strategy, Surface
from ..shared.reporting.orchestration import run_strategies
from ..shared.reporting.strategy import ModuleCaseSpec, SuiteCaseSpec
from ..shared.reporting.ui import make_dashboard

REPO_ROOT: Final = Path(__file__).resolve().parents[3]


class RunArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verb: Literal["run"]
    strategy: tuple[str, ...]
    sdk_functions: tuple[SdkFunction, ...]
    surface: Surface | None
    interactive: bool
    plain: bool
    runner_arg: tuple[str, ...]


class ListArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verb: Literal["list"]
    strategy: tuple[str, ...]
    sdk_functions: tuple[SdkFunction, ...]
    surface: Surface | None
    interactive: bool


class CheckArgs(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verb: Literal["check"]
    strategy: tuple[str, ...]
    sdk_functions: tuple[SdkFunction, ...]
    surface: Surface | None
    interactive: bool
    verbose: bool


def _grouped_cases(cases: Sequence[HarnessCase]) -> dict[str, tuple[HarnessCase, ...]]:
    grouped: dict[str, list[HarnessCase]] = {}
    for case in cases:
        grouped.setdefault(case.strategy_id, []).append(case)
    return {strategy_id: tuple(grouped_cases) for strategy_id, grouped_cases in grouped.items()}


def run_command(
    args: RunArgs,
    strategies: Sequence[Strategy],
    cases: Sequence[HarnessCase],
) -> int:
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
    )
    with dashboard:
        exit_code, run = run_strategies(
            runners,
            REPO_ROOT,
            dashboard.update,
            args.runner_arg,
        )
        if exit_code != 130:
            dashboard.finish(run, exit_code)
    return exit_code


def _case_detail(case: HarnessCase) -> str:
    spec = case.spec
    if isinstance(spec, SuiteCaseSpec) and spec.suite is not None:
        return spec.suite
    if isinstance(spec, ModuleCaseSpec) and spec.module is not None:
        return spec.module
    return "no test configured"


def list_command(
    args: ListArgs,
    strategies: Sequence[Strategy],
    cases: Sequence[HarnessCase],
) -> int:
    del args
    selected_keys: Final = {case.key for case in cases}
    for strategy in strategies:
        selected = tuple(
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
    args: CheckArgs,
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
