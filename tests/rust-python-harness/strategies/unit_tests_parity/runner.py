from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from pydantic import BaseModel, ConfigDict

from ...shared.reporting.models import HarnessCase, HarnessRun, RunStatus
from ...shared.reporting.pytest_runner import UpdateCallback
from ...shared.unit_runners.python_runner import BackendSpec, compare_python_runs, run_python_tests

BACKEND: Final = BackendSpec(environment_variable="LITELLM_RUST")


class UnitParitySuite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python_selectors: tuple[str, ...]


def run_suite(suite: UnitParitySuite, repo_root: Path, pytest_args: Sequence[str] = ()) -> tuple[str, ...]:
    if not suite.python_selectors:
        return ("unit parity suites must select Python tests",)
    python: Final = run_python_tests(suite.python_selectors, repo_root, "python", BACKEND, pytest_args)
    rust: Final = run_python_tests(suite.python_selectors, repo_root, "rust", BACKEND, pytest_args)
    return compare_python_runs(python, rust)


def run(
    cases: Sequence[HarnessCase],
    repo_root: Path,
    on_update: UpdateCallback,
    pytest_args: Sequence[str] = (),
) -> tuple[int, HarnessRun]:
    report: Final = HarnessRun.from_cases(cases)
    for case in cases:
        result: Final = report.results[case.key]
        if case.unit_suite is None:
            result.finalize()
            continue
        nodeid: Final = f"parity-suite:{case.unit_suite}"
        result.collected.add(nodeid)
        result.status = RunStatus.RUNNING
        on_update(report)
        try:
            suite: Final = UnitParitySuite.model_validate_json((repo_root / case.unit_suite).read_text())
            problems: Final = run_suite(suite, repo_root, pytest_args)
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


def main(argv: Sequence[str] | None = None) -> int:
    from ...cli import main as harness_main

    return harness_main(argv, strategy_id="unit_tests_parity")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
