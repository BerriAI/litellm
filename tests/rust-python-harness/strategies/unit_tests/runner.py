from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path
from time import monotonic
from typing import Final

from pydantic import BaseModel, ConfigDict

from ...shared.reporting.models import HarnessCase, HarnessRun, RunStatus
from ...shared.reporting.pytest_runner import UpdateCallback
from .mapping_validator import TestMapping, validate_mapping
from .python_runner import BackendSpec, compare_python_runs, run_python_tests
from .rust_runner import run_rust_tests


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
        return ("unit suites must select Python tests and a focused Cargo filter",)
    python: Final = run_python_tests(suite.python_selectors, repo_root, "python", suite.backend, pytest_args)
    rust_python: Final = run_python_tests(suite.python_selectors, repo_root, "rust", suite.backend, pytest_args)
    inventory: Final = run_rust_tests(
        repo_root / suite.cargo_manifest, suite.cargo_package, suite.cargo_filter, collect_only=True
    )
    mapping: Final = validate_mapping(python.tests, inventory.tests, suite.mappings)
    rust: Final = run_rust_tests(repo_root / suite.cargo_manifest, suite.cargo_package, suite.cargo_filter)
    return (
        *compare_python_runs(python, rust_python),
        *mapping.problems,
        *(("native Rust tests did not all pass",) if set(inventory.tests) != set(rust.tests) else ()),
        *((inventory.output,) if inventory.exit_code else ()),
        *((rust.output,) if rust.exit_code else ()),
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
        nodeid: Final = f"unit-suite:{case.unit_suite}"
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

    return harness_main(argv, strategy_id="unit_tests")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
