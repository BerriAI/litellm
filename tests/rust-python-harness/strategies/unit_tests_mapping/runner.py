from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from ...shared.native_build import ensure_trace_bridge
from ...shared.reporting.models import ResultArtifact
from ...shared.unit_runners.python_runner import collect_python_tests
from ...shared.unit_runners.rust_runner import enumerate_rust_tests
from ...shared.unit_runners.suite_runner import SuiteExecution
from .contracts import UnitTestContract
from .mapping_report import MappingReportArtifact
from .mapping_validator import PythonInventory, RustInventory, audit_mapping

MAPPING_REPORT_ARTIFACT: Final = "mapping_report"


def _audit_problems(artifact: MappingReportArtifact) -> tuple[str, ...]:
    report: Final = artifact.report
    return (
        *(f"mapped Python test does not exist: {nodeid}" for nodeid in report.missing_python_tests),
        *(f"mapped Rust test does not exist: {nodeid}" for nodeid in report.missing_rust_tests),
        *(f"Python test has multiple mappings: {nodeid}" for nodeid in report.duplicate_python_mappings),
        *(f"Rust test has multiple mappings: {nodeid}" for nodeid in report.duplicate_rust_mappings),
        *(f"mapping exclusion does not exist: {nodeid}" for nodeid in report.invalid_mapping_exclusions),
        *(f"Python test is both mapped and excluded: {nodeid}" for nodeid in report.mapped_and_excluded_python_tests),
        *(f"unit parity exclusion does not exist: {nodeid}" for nodeid in report.invalid_unit_parity_exclusions),
    )


def run_suite(
    contract: UnitTestContract,
    repo_root: Path,
    runner_args: Sequence[str] = (),
    *,
    python_inventory: PythonInventory = collect_python_tests,
    rust_inventory: RustInventory = enumerate_rust_tests,
) -> SuiteExecution:
    if contract.mapping.python_functions is not None and contract.mapping.python_functions.trace_module is not None:
        bridge_error: Final = ensure_trace_bridge(repo_root)
        if bridge_error is not None:
            return SuiteExecution(problems=(bridge_error,))
    artifact: Final = MappingReportArtifact(
        report=audit_mapping(
            contract,
            repo_root,
            python_inventory=python_inventory,
            rust_inventory=rust_inventory,
        ),
        detailed=bool(runner_args),
    )
    completeness_problems: Final = (
        tuple(f"Python test has no Rust mapping: {nodeid}" for nodeid in artifact.report.unmapped_python_tests)
        if contract.mapping.require_complete
        else ()
    )
    return SuiteExecution(
        problems=(*_audit_problems(artifact), *completeness_problems),
        artifacts=(ResultArtifact(MAPPING_REPORT_ARTIFACT, artifact.model_dump_json()),),
    )
