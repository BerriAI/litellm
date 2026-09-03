from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from ...shared.unit_runners.python_runner import BackendSpec, compare_python_runs, run_python_tests

BACKEND: Final = BackendSpec(environment_variable="LITELLM_RUST")


class UnitParitySuite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python_selectors: tuple[str, ...]


def run_suite(
    suite: UnitParitySuite, repo_root: Path, pytest_args: Sequence[str] = ()
) -> tuple[str, ...]:
    if not suite.python_selectors:
        return ("unit parity suites must select Python tests",)
    python: Final = run_python_tests(suite.python_selectors, repo_root, "python", BACKEND, pytest_args)
    rust: Final = run_python_tests(suite.python_selectors, repo_root, "rust", BACKEND, pytest_args)
    return compare_python_runs(python, rust)
