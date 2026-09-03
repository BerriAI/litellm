from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from time import monotonic
from typing import TypeVar

from pydantic import BaseModel

from ..reporting.models import HarnessCase, HarnessRun, RunStatus
from ..reporting.pytest_runner import UpdateCallback
from ..reporting.strategy import SuiteCaseSpec

S = TypeVar("S", bound=BaseModel)

SuiteLoader = Callable[[Path], S]
SuiteExecutor = Callable[[S, Path, Sequence[str]], tuple[str, ...]]


def suite_nodeid(case: HarnessCase) -> str:
    spec = case.spec
    suite = spec.suite if isinstance(spec, SuiteCaseSpec) else None
    return f"suite:{case.strategy_id}:{case.sdk_function}:{suite}"


def run_suites(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    pytest_args: Sequence[str] = (),
    *,
    load: SuiteLoader[S],
    execute: SuiteExecutor[S],
) -> tuple[int, HarnessRun]:
    report = HarnessRun.from_cases(cases)
    for case in cases:
        result = report.results[case.key]
        spec = case.spec
        if not isinstance(spec, SuiteCaseSpec) or spec.suite is None:
            result.finalize()
            continue
        nodeid = suite_nodeid(case)
        result.collected.add(nodeid)
        result.status = RunStatus.RUNNING
        on_update(report)
        try:
            suite = load(repo_root / spec.suite)
            problems = execute(suite, repo_root, pytest_args)
        except (OSError, ValueError) as error:
            result.record(nodeid, RunStatus.ERROR)
            report.failures.append((nodeid, str(error)))
            continue
        result.record(nodeid, RunStatus.FAILED if problems else RunStatus.PASSED)
        report.failures.extend((nodeid, problem) for problem in problems)
        on_update(report)
    report.finished_at = monotonic()
    on_update(report)
    return int(
        any(
            result.status in {RunStatus.ERROR, RunStatus.FAILED, RunStatus.MISSING}
            for result in report.results.values()
        )
    ), report
