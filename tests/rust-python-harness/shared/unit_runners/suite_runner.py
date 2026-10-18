from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Final, TypeVar

from pydantic import BaseModel

from ..reporting.models import HarnessCase, HarnessRun, ResultArtifact, RunStatus, SdkFunction
from ..reporting.strategy import SuiteCaseSpec, UpdateCallback

S = TypeVar("S", bound=BaseModel)


@dataclass(frozen=True, slots=True)
class SuiteExecution:
    problems: tuple[str, ...] = ()
    artifacts: tuple[ResultArtifact, ...] = ()


SuiteExecutor = Callable[[S, Path, Sequence[str]], SuiteExecution]


def suite_nodeid(case: HarnessCase) -> str:
    spec = case.spec
    suite = spec.suite if isinstance(spec, SuiteCaseSpec) else "invalid"
    return f"suite:{case.strategy_id}:{case.sdk_function}:{suite}"


def run_suites(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    runner_args: Sequence[str] = (),
    *,
    suites: Mapping[SdkFunction, S],
    execute: SuiteExecutor[S],
) -> tuple[int, HarnessRun]:
    report = HarnessRun.from_cases(cases)
    for case in cases:
        result = report.results[case.key]
        spec = case.spec
        if not isinstance(spec, SuiteCaseSpec):
            continue
        nodeid = suite_nodeid(case)
        result.collected.add(nodeid)
        result.status = RunStatus.RUNNING
        on_update(report)
        suite = suites.get(case.sdk_function)
        if suite is None:
            result.record(nodeid, RunStatus.ERROR)
            report.failures.append((nodeid, f"no suite registered for {case.sdk_function}"))
            continue
        try:
            execution: Final = execute(suite, repo_root, runner_args)
        except (OSError, ValueError) as error:
            result.record(nodeid, RunStatus.ERROR)
            report.failures.append((nodeid, str(error)))
            continue
        result.record(
            nodeid,
            RunStatus.FAILED if execution.problems else RunStatus.PASSED,
            artifacts=execution.artifacts,
        )
        report.failures.extend((nodeid, problem) for problem in execution.problems)
        on_update(report)
    report.finished_at = monotonic()
    on_update(report)
    return int(
        any(
            result.status in {RunStatus.ERROR, RunStatus.FAILED, RunStatus.MISSING}
            for result in report.results.values()
        )
    ), report
