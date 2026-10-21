from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from ...shared.unit_runners.rust_runner import run_rust_tests
from ...shared.unit_runners.suite_runner import SuiteExecution


class RustSuite(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    cargo_manifest: str
    cargo_package: str | None = None
    cargo_filter: str


def run_suite(suite: RustSuite, repo_root: Path, pytest_args: Sequence[str] = ()) -> SuiteExecution:
    del pytest_args
    if not suite.cargo_filter:
        return SuiteExecution(problems=("rust suites must configure a focused Cargo filter",))
    inventory = run_rust_tests(
        repo_root / suite.cargo_manifest, suite.cargo_package, suite.cargo_filter, collect_only=True
    )
    rust = run_rust_tests(repo_root / suite.cargo_manifest, suite.cargo_package, suite.cargo_filter)
    return SuiteExecution(
        problems=(
            *(("native Rust tests did not all pass",) if set(inventory.tests) != set(rust.tests) else ()),
            *((inventory.output,) if inventory.exit_code else ()),
            *((rust.output,) if rust.exit_code else ()),
        )
    )
