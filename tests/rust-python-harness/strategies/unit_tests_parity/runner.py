from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from ...shared.unit_runners.python_runner import BackendSpec, compare_python_runs, run_python_tests
from ...shared.unit_runners.suite_runner import SuiteExecution

BACKEND: Final = BackendSpec(environment_variable="LITELLM_RUST")


class UnitParityExclusion(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    nodeid: str
    reason: str


class UnitParitySuite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python_selectors: tuple[str, ...]
    exclusions: tuple[UnitParityExclusion, ...] = ()


def run_suite(suite: UnitParitySuite, repo_root: Path, pytest_args: Sequence[str] = ()) -> SuiteExecution:
    if not suite.python_selectors:
        return SuiteExecution(problems=("unit parity suites must select Python tests",))
    deselections: Final = tuple(f"--deselect={exclusion.nodeid}" for exclusion in suite.exclusions)
    args: Final = (*pytest_args, *deselections)
    python: Final = run_python_tests(suite.python_selectors, repo_root, "python", BACKEND, args)
    rust: Final = run_python_tests(suite.python_selectors, repo_root, "rust", BACKEND, args)
    return SuiteExecution(problems=compare_python_runs(python, rust))
