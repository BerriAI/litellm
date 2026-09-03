from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict

from ...shared.parity.ledger import TestLedger, load_ledger
from .python_runner import enumerate_python_tests
from .rust_runner import enumerate_rust_tests


class TestMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    python: str
    rust: str


@dataclass(frozen=True, slots=True)
class MappingReport:
    pairs: tuple[TestMapping, ...]
    problems: tuple[str, ...]


def _name(node: str) -> str:
    return node.rsplit("::", 1)[-1].split("[", 1)[0]


def validate_mapping(
    python_tests: Sequence[str],
    rust_tests: Sequence[str],
    annotations: Sequence[TestMapping] = (),
) -> MappingReport:
    explicit_problems: Final = (
        *(f"missing Python counterpart: {pair.python}" for pair in annotations if pair.python not in python_tests),
        *(f"missing Rust counterpart: {pair.rust}" for pair in annotations if pair.rust not in rust_tests),
        *(
            f"ambiguous Python annotation: {name}"
            for name, count in Counter(p.python for p in annotations).items()
            if count > 1
        ),
        *(
            f"ambiguous Rust annotation: {name}"
            for name, count in Counter(p.rust for p in annotations).items()
            if count > 1
        ),
    )
    explicit_python: Final = {pair.python for pair in annotations}
    candidates: Final = {
        python: tuple(rust for rust in rust_tests if _name(python) == _name(rust))
        for python in python_tests
        if python not in explicit_python
    }
    pairs: Final = (
        *annotations,
        *(TestMapping(python=python, rust=matches[0]) for python, matches in candidates.items() if len(matches) == 1),
    )
    problems: Final = (
        *explicit_problems,
        *(f"missing Rust counterpart: {python}" for python, matches in candidates.items() if not matches),
        *(
            f"ambiguous Rust counterparts: {python}: {matches}"
            for python, matches in candidates.items()
            if len(matches) > 1
        ),
        *(
            f"ambiguous Python counterparts: {rust}"
            for rust, count in Counter(pair.rust for pair in pairs).items()
            if count > 1
        ),
        *(f"missing Python counterpart: {rust}" for rust in rust_tests if rust not in {pair.rust for pair in pairs}),
        *(("no Python tests collected",) if not python_tests else ()),
        *(("no Rust tests collected",) if not rust_tests else ()),
    )
    return MappingReport(pairs, problems)


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
