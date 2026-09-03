from __future__ import annotations

from pathlib import Path

from ...shared.reporting.strategy import CheckReport
from .mapping_validator import FunctionReport, build_function_report


def function_report_lines(report: FunctionReport) -> tuple[str, ...]:
    lines = [report.sdk_function]
    if report.ledger is None or report.audit is None:
        lines.append("  no ledger yet")
        return tuple(lines)
    ledger, audit = report.ledger, report.audit
    lines.append(
        f"  {ledger.mapped_count}/{ledger.total_count} python tests mapped to rust "
        f"({ledger.percentage}%)"
    )
    lines.append(f"  {len(ledger.rust_only_tests)} rust-only tests with no python counterpart")
    if audit.is_clean:
        lines.append("  ledger is in sync with the live test files")
        return tuple(lines)
    for label, items in (
        ("ledger references a python test that no longer exists", audit.missing_python_tests),
        ("python test exists but is not tracked in the ledger", audit.stale_python_tests),
        ("ledger references a rust test that no longer exists", audit.missing_rust_tests),
        ("rust test exists but is not tracked in the ledger", audit.stale_rust_tests),
    ):
        for item in items:
            lines.append(f"  {label}: {item}")
    return tuple(lines)


def check_ledgers(
    sdk_functions: frozenset[str], repo_root: Path
) -> tuple[CheckReport, ...]:
    reports = tuple(
        build_function_report(function, repo_root) for function in sorted(sdk_functions)
    )
    return tuple(
        CheckReport(
            sdk_function=report.sdk_function,
            lines=function_report_lines(report),
            passed=report.is_clean,
        )
        for report in reports
    )
