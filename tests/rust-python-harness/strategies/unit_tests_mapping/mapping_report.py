from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from typing import Final

from pydantic import BaseModel, ConfigDict

from .mapping_validator import MappingReport


class MappingReportArtifact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    report: MappingReport
    detailed: bool = False


def _group_counts(nodeids: Sequence[str], owner: Callable[[str], str]) -> tuple[str, ...]:
    counts: Final = Counter(owner(nodeid) for nodeid in nodeids)
    width: Final = max((len(str(count)) for count in counts.values()), default=1)
    return tuple(
        f"  {count:>{width}}  {name}" for name, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    )


def _python_file(nodeid: str) -> str:
    return nodeid.partition("::")[0]


def _rust_module(nodeid: str) -> str:
    return nodeid.rpartition("::")[0]


def _details(nodeids: Sequence[str], owner: Callable[[str], str]) -> tuple[str, ...]:
    owners: Final = tuple(sorted(frozenset(owner(nodeid) for nodeid in nodeids)))
    return tuple(
        line
        for name in owners
        for line in (
            f"  {name}",
            *(f"    {nodeid.removeprefix(f'{name}::')}" for nodeid in nodeids if owner(nodeid) == name),
        )
    )


def _contract_errors(report: MappingReport) -> tuple[str, ...]:
    return (
        *(f"  Missing Python test: {nodeid}" for nodeid in report.missing_python_tests),
        *(f"  Missing Rust test: {nodeid}" for nodeid in report.missing_rust_tests),
        *(f"  Python test mapped more than once: {nodeid}" for nodeid in report.duplicate_python_mappings),
        *(f"  Rust test mapped more than once: {nodeid}" for nodeid in report.duplicate_rust_mappings),
        *(f"  Missing mapping exclusion: {nodeid}" for nodeid in report.invalid_mapping_exclusions),
        *(f"  Python test is both mapped and excluded: {nodeid}" for nodeid in report.mapped_and_excluded_python_tests),
        *(f"  Missing unit-parity exclusion: {nodeid}" for nodeid in report.invalid_unit_parity_exclusions),
    )


def mapping_report_lines(report: MappingReport, *, detailed: bool = False) -> tuple[str, ...]:
    unmapped_count: Final = len(report.unmapped_python_tests)
    excluded_count: Final = len(report.excluded_python_tests)
    excluded_percentage: Final = (
        0.0 if not report.total_count else round(100.0 * excluded_count / report.total_count, 1)
    )
    unmapped_percentage: Final = (
        0.0 if not report.total_count else round(100.0 * unmapped_count / report.total_count, 1)
    )
    rust_total: Final = len(report.rust_tests)
    rust_only_count: Final = len(report.rust_only_tests)
    rust_mapped_count: Final = rust_total - rust_only_count
    contract_errors: Final = _contract_errors(report)
    detail_lines: Final = (
        (
            "",
            "Unmapped Python test details",
            *_details(report.unmapped_python_tests, _python_file),
            "",
            "Excluded Python test details",
            *_details(report.excluded_python_tests, _python_file),
            "",
            "Rust-only test details",
            *_details(report.rust_only_tests, _rust_module),
        )
        if detailed
        else ()
    )
    return (
        f"Contract: {'PASS' if report.is_valid else 'FAIL'}",
        "",
        "Python coverage",
        f"  Mapped     {report.mapped_count:>3} / {report.total_count} ({report.percentage}%)",
        f"  Excluded   {excluded_count:>3} / {report.total_count} ({excluded_percentage}%)",
        f"  Unmapped   {unmapped_count:>3} / {report.total_count} ({unmapped_percentage}%)",
        "",
        "Rust inventory",
        f"  Mapped     {rust_mapped_count:>3} / {rust_total}",
        f"  Rust-only  {rust_only_count:>3} / {rust_total}",
        "",
        f"Unmapped Python tests by file ({unmapped_count})",
        *_group_counts(report.unmapped_python_tests, _python_file),
        "",
        f"Excluded Python tests by file ({excluded_count})",
        *_group_counts(report.excluded_python_tests, _python_file),
        "",
        f"Rust-only tests by module ({rust_only_count})",
        *_group_counts(report.rust_only_tests, _rust_module),
        *(("", "Contract errors", *contract_errors) if contract_errors else ()),
        *detail_lines,
    )
