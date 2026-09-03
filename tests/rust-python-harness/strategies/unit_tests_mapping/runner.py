from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict

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


def run_suite(
    suite: UnitSuite, repo_root: Path, pytest_args: Sequence[str] = ()
) -> tuple[str, ...]:
    if not suite.python_selectors or not suite.cargo_filter:
        return ("mapping suites must select Python tests and a focused Cargo filter",)
    python = run_python_tests(
        suite.python_selectors, repo_root, "python", suite.backend, (*pytest_args, "--collect-only")
    )
    inventory = run_rust_tests(
        repo_root / suite.cargo_manifest, suite.cargo_package, suite.cargo_filter, collect_only=True
    )
    mapping = validate_mapping(python.tests, inventory.tests, suite.mappings)
    return (
        *mapping.problems,
        *(("Python test collection failed",) if python.exit_code else ()),
        *(("Rust test collection failed",) if inventory.exit_code else ()),
        *python.problems,
        *((inventory.output,) if inventory.exit_code else ()),
    )
