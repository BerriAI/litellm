from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from .models import HarnessRun, Strategy
from .strategy import StrategyRunner, UpdateCallback

__all__ = ["StrategyRunner", "run_strategies"]


def combine_reports(
    reports: Sequence[HarnessRun],
    *,
    timed_reports: Sequence[HarnessRun] | None = None,
) -> HarnessRun:
    duration_sources: Final = reports if timed_reports is None else timed_reports
    return HarnessRun(
        results={key: result for report in reports for key, result in report.results.items()},
        current_nodeid=next((report.current_nodeid for report in reversed(reports) if report.current_nodeid), None),
        failures=[failure for report in reports for failure in report.failures],
        strategy_durations={
            strategy_id: report.duration
            for report in duration_sources
            for strategy_id in {result.case.strategy_id for result in report.results.values()}
        },
        started_at=min((report.started_at for report in reports), default=monotonic()),
        finished_at=(
            max((report.finished_at for report in reports if report.finished_at is not None), default=None)
            if all(report.finished_at is not None for report in reports)
            else None
        ),
    )


def run_strategies(
    strategies: Sequence[Strategy],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    cases: Final = tuple(case for strategy in strategies for case in strategy.cases)

    def execute(
        remaining: tuple[Strategy, ...], reports: tuple[HarnessRun, ...], codes: tuple[int, ...]
    ) -> tuple[int, HarnessRun]:
        if not remaining:
            combined: Final = combine_reports(reports)
            on_update(combined)
            return next((code for code in codes if code), 0), combined
        strategy, *tail = remaining
        selected: Final = tuple(case for case in cases if case.strategy_id == strategy.id)
        pending: Final = HarnessRun.from_cases(
            case for case in cases if case.strategy_id in {later.id for later in tail}
        )
        code, report = strategy.definition.run(
            selected,
            repo_root,
            lambda current: on_update(
                combine_reports(
                    (*reports, current, pending),
                    timed_reports=(*reports, current),
                )
            ),
            runner_args,
        )
        if code in {2, 3, 4}:
            return code, combine_reports((*reports, report, pending))
        return execute(tuple(tail), (*reports, report), (*codes, code))

    return execute(tuple(strategies), (), ())
