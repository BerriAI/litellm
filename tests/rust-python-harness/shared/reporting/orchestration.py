from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from time import monotonic
from typing import Final, Protocol

from .models import HarnessCase, HarnessRun
from .pytest_runner import UpdateCallback


class StrategyRunner(Protocol):
    def __call__(
        self,
        cases: Sequence[HarnessCase],
        repo_root: Path,
        on_update: UpdateCallback,
        pytest_args: Sequence[str] = (),
    ) -> tuple[int, HarnessRun]: ...


def combine_reports(reports: Sequence[HarnessRun]) -> HarnessRun:
    return HarnessRun(
        results={key: result for report in reports for key, result in report.results.items()},
        current_nodeid=next((report.current_nodeid for report in reversed(reports) if report.current_nodeid), None),
        failures=[failure for report in reports for failure in report.failures],
        started_at=min((report.started_at for report in reports), default=monotonic()),
        finished_at=(
            max((report.finished_at for report in reports if report.finished_at is not None), default=None)
            if all(report.finished_at is not None for report in reports)
            else None
        ),
    )


def run_strategies(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    pytest_args: Sequence[str],
    resolve_runner: Callable[[str], StrategyRunner],
) -> tuple[int, HarnessRun]:
    strategy_ids: Final = tuple(dict.fromkeys(case.strategy_id for case in cases))

    def execute(
        remaining: tuple[str, ...], reports: tuple[HarnessRun, ...], codes: tuple[int, ...]
    ) -> tuple[int, HarnessRun]:
        if not remaining:
            combined: Final = combine_reports(reports)
            on_update(combined)
            return next((code for code in codes if code), 0), combined
        strategy_id, *tail = remaining
        selected: Final = tuple(case for case in cases if case.strategy_id == strategy_id)
        pending: Final = HarnessRun.from_cases(case for case in cases if case.strategy_id in tail)
        code, report = resolve_runner(strategy_id)(
            selected,
            repo_root,
            lambda current: on_update(combine_reports((*reports, current, pending))),
            pytest_args,
        )
        if code in {2, 3, 4}:
            return code, combine_reports((*reports, report, pending))
        return execute(tuple(tail), (*reports, report), (*codes, code))

    return execute(strategy_ids, (), ())
