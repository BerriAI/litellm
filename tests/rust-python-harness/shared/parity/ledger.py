from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass(frozen=True, slots=True)
class LedgerEntry:
    python_file: str
    python_test: str
    status: str
    rust_file: str
    rust_test: str
    justification: str
    reason: str


@dataclass(frozen=True, slots=True)
class RustOnlyEntry:
    rust_file: str
    rust_test: str
    reason: str


@dataclass(frozen=True, slots=True)
class TestLedger:
    sdk_function: str
    python_scope: tuple[str, ...]
    rust_scope: tuple[str, ...]
    entries: tuple[LedgerEntry, ...]
    rust_only_tests: tuple[RustOnlyEntry, ...]

    @property
    def mapped_count(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "mapped")

    @property
    def total_count(self) -> int:
        return len(self.entries)

    @property
    def percentage(self) -> float:
        if self.total_count == 0:
            return 0.0
        return round(100.0 * self.mapped_count / self.total_count, 1)


def _require_string(value: Any, field: str, source: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source}: {field} must be a non-empty string")
    return value


def _require_string_list(value: Any, field: str, source: Path) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{source}: {field} must be a list of non-empty strings")
    return tuple(value)


def _load_entry(data: Any, index: int, source: Path) -> LedgerEntry:
    if not isinstance(data, dict):
        raise ValueError(f"{source}: entries[{index}] must be an object")
    python_file = _require_string(data.get("python_file"), f"entries[{index}].python_file", source)
    python_test = _require_string(data.get("python_test"), f"entries[{index}].python_test", source)
    status = data.get("status")
    if status not in ("mapped", "unmapped"):
        raise ValueError(f"{source}: entries[{index}].status must be 'mapped' or 'unmapped'")

    if status == "mapped":
        rust_file = _require_string(data.get("rust_file"), f"entries[{index}].rust_file", source)
        rust_test = _require_string(data.get("rust_test"), f"entries[{index}].rust_test", source)
        justification = _require_string(
            data.get("justification"), f"entries[{index}].justification", source
        )
        return LedgerEntry(
            python_file=python_file,
            python_test=python_test,
            status=status,
            rust_file=rust_file,
            rust_test=rust_test,
            justification=justification,
            reason="",
        )

    reason = _require_string(data.get("reason"), f"entries[{index}].reason", source)
    return LedgerEntry(
        python_file=python_file,
        python_test=python_test,
        status=status,
        rust_file="",
        rust_test="",
        justification="",
        reason=reason,
    )


def _load_rust_only_entry(data: Any, index: int, source: Path) -> RustOnlyEntry:
    if not isinstance(data, dict):
        raise ValueError(f"{source}: rust_only_tests[{index}] must be an object")
    return RustOnlyEntry(
        rust_file=_require_string(data.get("rust_file"), f"rust_only_tests[{index}].rust_file", source),
        rust_test=_require_string(data.get("rust_test"), f"rust_only_tests[{index}].rust_test", source),
        reason=_require_string(data.get("reason"), f"rust_only_tests[{index}].reason", source),
    )


def load_ledger(path: Path) -> TestLedger:
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)

    sdk_function = _require_string(data.get("sdk_function"), "sdk_function", path)
    python_scope = _require_string_list(data.get("python_scope"), "python_scope", path)
    rust_scope = _require_string_list(data.get("rust_scope"), "rust_scope", path)

    entries_data = data.get("entries")
    if not isinstance(entries_data, list):
        raise ValueError(f"{path}: entries must be a list")
    entries = tuple(
        _load_entry(entry, index, path) for index, entry in enumerate(entries_data)
    )

    rust_only_data = data.get("rust_only_tests")
    if not isinstance(rust_only_data, list):
        raise ValueError(f"{path}: rust_only_tests must be a list")
    rust_only_tests = tuple(
        _load_rust_only_entry(entry, index, path) for index, entry in enumerate(rust_only_data)
    )

    return TestLedger(
        sdk_function=sdk_function,
        python_scope=python_scope,
        rust_scope=rust_scope,
        entries=entries,
        rust_only_tests=rust_only_tests,
    )
