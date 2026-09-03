from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from ...shared.parity.ledger import RustTestIdentity, RustTestScope, TestLedger, load_ledger
from .python_runner import enumerate_python_tests
from .rust_runner import enumerate_rust_tests

REPO_ROOT: Final = Path(__file__).resolve().parents[4]
LEDGER_DIRECTORY: Final = Path("tests/rust-python-harness/strategies/unit_tests/ledgers")
RustInventory: TypeAlias = Callable[[Path, tuple[RustTestScope, ...]], frozenset[RustTestIdentity]]


def ledger_path_for(sdk_function: str, repo_root: Path = REPO_ROOT) -> Path:
    return repo_root / LEDGER_DIRECTORY / sdk_function / f"{sdk_function}_test_ledger.json"


@dataclass(frozen=True, slots=True)
class AuditReport:
    missing_python_tests: tuple[str, ...]
    stale_python_tests: tuple[str, ...]
    missing_rust_tests: tuple[str, ...]
    stale_rust_tests: tuple[str, ...]

    @property
    def is_clean(self) -> bool:
        return not (
            self.missing_python_tests or self.stale_python_tests or self.missing_rust_tests or self.stale_rust_tests
        )


def _ledger_python_tests_by_file(ledger: TestLedger) -> dict[str, set[str]]:
    return {
        path: {entry.python_test for entry in ledger.entries if entry.python_file == path}
        for path in ledger.python_scope
    }


def audit_ledger(
    ledger: TestLedger, repo_root: Path = REPO_ROOT, *, rust_inventory: RustInventory = enumerate_rust_tests
) -> AuditReport:
    python_tests: Final = tuple(
        (python_file, ledger_tests, enumerate_python_tests(repo_root, python_file))
        for python_file, ledger_tests in _ledger_python_tests_by_file(ledger).items()
    )
    missing_python: Final = tuple(
        f"{python_file}:{missing}"
        for python_file, ledger_tests, actual_tests in python_tests
        for missing in sorted(ledger_tests - actual_tests)
    )
    stale_python: Final = tuple(
        f"{python_file}:{stale}"
        for python_file, ledger_tests, actual_tests in python_tests
        for stale in sorted(actual_tests - ledger_tests)
    )

    rust_files: Final = frozenset(entry.rust_file for entry in ledger.entries if entry.status == "mapped") | frozenset(
        entry.rust_file for entry in ledger.rust_only_tests
    )
    missing_files: Final = tuple(sorted(path for path in rust_files if not (repo_root / path).is_file()))
    if missing_files:
        raise ValueError(f"Rust source locators no longer exist: {', '.join(missing_files)}")
    actual_rust: Final = rust_inventory(repo_root, ledger.rust_scope)

    return AuditReport(
        missing_python_tests=missing_python,
        stale_python_tests=stale_python,
        missing_rust_tests=tuple(sorted(identity.key for identity in ledger.rust_tests - actual_rust)),
        stale_rust_tests=tuple(sorted(identity.key for identity in actual_rust - ledger.rust_tests)),
    )


@dataclass(frozen=True, slots=True)
class FunctionReport:
    sdk_function: str
    ledger: TestLedger | None
    audit: AuditReport | None
    error: str | None = None

    @property
    def has_ledger(self) -> bool:
        return self.ledger is not None

    @property
    def passes_validation(self) -> bool:
        return self.passes_audit and self.ledger is not None and self.ledger.unresolved_portable_count == 0

    @property
    def passes_audit(self) -> bool:
        return self.error is None and self.ledger is not None and self.audit is not None and self.audit.is_clean


def build_function_report(
    sdk_function: str, repo_root: Path = REPO_ROOT, *, rust_inventory: RustInventory = enumerate_rust_tests
) -> FunctionReport:
    path: Final = ledger_path_for(sdk_function, repo_root)
    if not path.exists():
        return FunctionReport(sdk_function=sdk_function, ledger=None, audit=None)
    try:
        ledger: Final = load_ledger(path)
        if ledger.sdk_function != sdk_function:
            raise ValueError(f"{path}: sdk_function must be {sdk_function!r}")
        return FunctionReport(
            sdk_function=sdk_function,
            ledger=ledger,
            audit=audit_ledger(ledger, repo_root, rust_inventory=rust_inventory),
        )
    except (ValueError, OSError, SyntaxError) as exc:
        return FunctionReport(sdk_function=sdk_function, ledger=None, audit=None, error=str(exc))
