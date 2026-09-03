from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ...shared.parity.ledger import TestLedger, load_ledger
from .python_runner import enumerate_python_tests
from .rust_runner import enumerate_rust_tests

REPO_ROOT = Path(__file__).resolve().parents[4]
LEDGER_ROOT = Path(__file__).parent / "ledgers"


def ledger_path_for(sdk_function: str) -> Path:
    return LEDGER_ROOT / sdk_function / f"{sdk_function}_test_ledger.json"


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
        actual_tests = enumerate_python_tests(repo_root, python_file)
        for missing in sorted(ledger_tests - actual_tests):
            missing_python.append(f"{python_file}:{missing}")
        for stale in sorted(actual_tests - ledger_tests):
            stale_python.append(f"{python_file}:{stale}")

    missing_rust: list[str] = []
    stale_rust: list[str] = []
    for rust_file, ledger_tests in _ledger_rust_tests_by_file(ledger).items():
        actual_tests = enumerate_rust_tests(repo_root, rust_file)
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


@dataclass(frozen=True, slots=True)
class FunctionReport:
    sdk_function: str
    ledger: TestLedger | None
    audit: AuditReport | None

    @property
    def has_ledger(self) -> bool:
        return self.ledger is not None

    @property
    def is_clean(self) -> bool:
        return self.audit is None or self.audit.is_clean


def build_function_report(sdk_function: str, repo_root: Path = REPO_ROOT) -> FunctionReport:
    path = ledger_path_for(sdk_function)
    if not path.exists():
        return FunctionReport(sdk_function=sdk_function, ledger=None, audit=None)
    ledger = load_ledger(path)
    return FunctionReport(
        sdk_function=sdk_function, ledger=ledger, audit=audit_ledger(ledger, repo_root)
    )
