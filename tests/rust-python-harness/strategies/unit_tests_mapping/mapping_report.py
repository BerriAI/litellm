from __future__ import annotations

from .mapping_validator import MappingReport


def mapping_report_lines(report: MappingReport) -> tuple[str, ...]:
    return (
        f"{report.mapped_count}/{report.total_count} python tests mapped to rust ({report.percentage}%)",
        f"{len(report.rust_only_tests)} rust-only tests with no python counterpart",
        *(f"unmapped python test: {nodeid}" for nodeid in report.unmapped_python_tests),
        *(f"rust-only test: {nodeid}" for nodeid in report.rust_only_tests),
        *(f"mapped python test does not exist: {nodeid}" for nodeid in report.missing_python_tests),
        *(f"mapped rust test does not exist: {nodeid}" for nodeid in report.missing_rust_tests),
        *(f"python test has multiple mappings: {nodeid}" for nodeid in report.duplicate_python_mappings),
        *(f"unit parity exclusion does not exist: {nodeid}" for nodeid in report.invalid_unit_parity_exclusions),
        *(("mapping contract is valid",) if report.is_valid else ()),
    )
