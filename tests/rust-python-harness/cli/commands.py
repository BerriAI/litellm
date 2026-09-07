from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import replace
from pathlib import Path
from typing import Final

from ..shared.reporting.models import HarnessCase, SdkFunction, Strategy, Surface
from ..shared.reporting.orchestration import run_strategies
from ..shared.reporting.ui import make_dashboard

REPO_ROOT: Final = Path(__file__).resolve().parents[3]


def select_cases(
    strategies: Sequence[Strategy],
    sdk_functions: Set[SdkFunction],
    surface: Surface | None = None,
) -> tuple[HarnessCase, ...]:
    return tuple(
        case
        for strategy in strategies
        for case in strategy.cases
        if (not sdk_functions or case.sdk_function in sdk_functions)
        and (surface is None or case.surface == surface)
    )


def run_command(
    strategies: Sequence[Strategy],
    cases: Sequence[HarnessCase],
    runner_args: Sequence[str] = (),
) -> int:
    grouped: Final = {
        strategy.id: tuple(case for case in cases if case.strategy_id == strategy.id)
        for strategy in strategies
    }
    visible: Final = tuple(strategy for strategy in strategies if grouped[strategy.id])
    runners: Final = tuple(replace(strategy, cases=grouped[strategy.id]) for strategy in visible)
    dashboard: Final = make_dashboard(visible)
    with dashboard:
        exit_code, run = run_strategies(runners, REPO_ROOT, dashboard.update, runner_args)
        if exit_code != 130:
            dashboard.finish(run, exit_code)
    return exit_code
