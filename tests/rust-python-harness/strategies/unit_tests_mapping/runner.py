from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from pydantic import BaseModel, ConfigDict

from ...shared.reporting.models import HarnessCase, HarnessRun, RunStatus
from ...shared.reporting.pytest_runner import UpdateCallback
from ...shared.unit_runners.python_runner import BackendSpec, run_python_tests
from ...shared.unit_runners.rust_runner import run_rust_tests
from .mapping_validator import TestMapping, validate_mapping


class UnitSuite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python_selectors: tuple[str, ...]
    cargo_manifest: str
    cargo_package: str
    cargo_filter: str
    backend: BackendSpec
    mappings: tuple[TestMapping, ...] = ()


def run_suite(suite: UnitSuite, repo_root: Path, pytest_args: Sequence[str] = ()) -> tuple[str, ...]:
    if not suite.python_selectors or not suite.cargo_filter:
        return ("mapping suites must select Python tests and a focused Cargo filter",)
    python: Final = run_python_tests(
        suite.python_selectors, repo_root, "python", suite.backend, (*pytest_args, "--collect-only")
    )
    inventory: Final = run_rust_tests(
        repo_root / suite.cargo_manifest, suite.cargo_package, suite.cargo_filter, collect_only=True
    )
    mapping: Final = validate_mapping(python.tests, inventory.tests, suite.mappings)
    return (
        *mapping.problems,
        *(("Python test collection failed",) if python.exit_code else ()),
        *(("Rust test collection failed",) if inventory.exit_code else ()),
        *python.problems,
        *((inventory.output,) if inventory.exit_code else ()),
    )


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
        nodeid: Final = f"mapping-suite:{case.unit_suite}"
        result.collected.add(nodeid)
        result.status = RunStatus.RUNNING
        on_update(report)
        try:
            suite: Final = UnitSuite.model_validate_json((repo_root / case.unit_suite).read_text())
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

    return harness_main(argv, strategy_id="unit_tests_mapping")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
