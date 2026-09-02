from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from .ledger import LEDGER_PATH, TestLedger, load_ledger

REPO_ROOT = Path(__file__).parent.parent.parent

_RUST_TEST_PATTERN = re.compile(
    r"#\[(?:test|tokio::test)\][^\n]*\n(?:[^\n]*\n)*?\s*(?:async\s+)?fn\s+(\w+)\s*\("
)


@dataclass(frozen=True, slots=True)
class AuditReport:
    missing_python_tests: tuple[str, ...]
    stale_python_tests: tuple[str, ...]
    missing_rust_tests: tuple[str, ...]
    stale_rust_tests: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not (
            self.missing_python_tests
            or self.stale_python_tests
            or self.missing_rust_tests
            or self.stale_rust_tests
        )


class LedgerDriftError(Exception):
    pass


def _enumerate_python_tests(repo_root: Path, relative_path: str) -> frozenset[str]:
    source = (repo_root / relative_path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=relative_path)

    module_level: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_"):
            module_level.append(node.name)
        elif isinstance(node, ast.ClassDef):
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and child.name.startswith(
                    "test_"
                ):
                    module_level.append(f"{node.name}::{child.name}")

    return frozenset(module_level)


def _enumerate_rust_tests(repo_root: Path, relative_path: str) -> frozenset[str]:
    source = (repo_root / relative_path).read_text(encoding="utf-8")
    return frozenset(match.group(1) for match in _RUST_TEST_PATTERN.finditer(source))


def _ledger_python_tests_by_file(ledger: TestLedger) -> dict[str, set[str]]:
    grouping: dict[str, set[str]] = {path: set() for path in ledger.python_scope}
    for entry in ledger.entries:
        grouping.setdefault(entry.python_file, set()).add(entry.python_test)
    return grouping


def _ledger_rust_tests_by_file(ledger: TestLedger) -> dict[str, set[str]]:
    grouping: dict[str, set[str]] = {path: set() for path in ledger.rust_scope}
    for entry in ledger.entries:
        if entry.status == "mapped":
            grouping.setdefault(entry.rust_file, set()).add(entry.rust_test)
    for rust_only in ledger.rust_only_tests:
        grouping.setdefault(rust_only.rust_file, set()).add(rust_only.rust_test)
    return grouping


def audit_ledger(ledger: TestLedger, repo_root: Path = REPO_ROOT) -> AuditReport:
    missing_python: list[str] = []
    stale_python: list[str] = []
    for python_file, ledger_tests in _ledger_python_tests_by_file(ledger).items():
        actual_tests = _enumerate_python_tests(repo_root, python_file)
        for missing in sorted(ledger_tests - actual_tests):
            missing_python.append(f"{python_file}:{missing}")
        for stale in sorted(actual_tests - ledger_tests):
            stale_python.append(f"{python_file}:{stale}")

    missing_rust: list[str] = []
    stale_rust: list[str] = []
    for rust_file, ledger_tests in _ledger_rust_tests_by_file(ledger).items():
        actual_tests = _enumerate_rust_tests(repo_root, rust_file)
        for missing in sorted(ledger_tests - actual_tests):
            missing_rust.append(f"{rust_file}:{missing}")
        for stale in sorted(actual_tests - ledger_tests):
            stale_rust.append(f"{rust_file}:{stale}")

    return AuditReport(
        missing_python_tests=tuple(missing_python),
        stale_python_tests=tuple(stale_python),
        missing_rust_tests=tuple(missing_rust),
        stale_rust_tests=tuple(stale_rust),
    )


def run_audit(ledger_path: Path = LEDGER_PATH, repo_root: Path = REPO_ROOT) -> AuditReport:
    ledger = load_ledger(ledger_path)
    report = audit_ledger(ledger, repo_root)
    if report.is_clean:
        return report

    lines: list[str] = []
    for label, items in (
        ("ledger references a python test that no longer exists", report.missing_python_tests),
        ("python test exists but is not tracked in the ledger", report.stale_python_tests),
        ("ledger references a rust test that no longer exists", report.missing_rust_tests),
        ("rust test exists but is not tracked in the ledger", report.stale_rust_tests),
    ):
        for item in items:
            lines.append(f"{label}: {item}")
    raise LedgerDriftError("\n".join(lines))
