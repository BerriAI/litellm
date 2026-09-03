from __future__ import annotations

from itertools import groupby
from pathlib import Path
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Self

NonEmptyString: TypeAlias = Annotated[
    str, StringConstraints(min_length=1, pattern=r"\S")
]
LEDGER_CONFIG: Final = ConfigDict(extra="forbid", frozen=True, strict=True)


class MappedLedgerEntry(BaseModel):
    model_config = LEDGER_CONFIG

    python_file: NonEmptyString
    python_test: NonEmptyString
    status: Literal["mapped"]
    rust_file: NonEmptyString
    rust_test: NonEmptyString
    justification: NonEmptyString


class UnmappedLedgerEntry(BaseModel):
    model_config = LEDGER_CONFIG

    python_file: NonEmptyString
    python_test: NonEmptyString
    status: Literal["python_only", "unresolved_portable"]
    reason: NonEmptyString


LedgerEntry: TypeAlias = Annotated[
    MappedLedgerEntry | UnmappedLedgerEntry, Field(discriminator="status")
]


class RustOnlyEntry(BaseModel):
    model_config = LEDGER_CONFIG

    rust_file: NonEmptyString
    rust_test: NonEmptyString
    reason: NonEmptyString


def _require_unique(targets: tuple[str, ...], field: str) -> None:
    duplicates: Final = tuple(
        target
        for target, group in groupby(sorted(targets))
        if sum(1 for _ in group) > 1
    )
    if duplicates:
        raise ValueError(f"{field} contains duplicates: {', '.join(duplicates)}")


class TestLedger(BaseModel):
    model_config = LEDGER_CONFIG

    schema_version: Literal[1]
    sdk_function: NonEmptyString
    python_scope: tuple[NonEmptyString, ...]
    rust_scope: tuple[NonEmptyString, ...]
    entries: tuple[LedgerEntry, ...]
    rust_only_tests: tuple[RustOnlyEntry, ...]

    @model_validator(mode="after")
    def validate_mapping_structure(self) -> Self:
        _require_unique(self.python_scope, "python_scope")
        _require_unique(self.rust_scope, "rust_scope")
        _require_unique(
            tuple(f"{entry.python_file}::{entry.python_test}" for entry in self.entries),
            "Python test identities",
        )
        mapped_targets: Final = tuple(
            f"{entry.rust_file}::{entry.rust_test}" for entry in self.entries if entry.status == "mapped"
        )
        rust_only_targets: Final = tuple(f"{entry.rust_file}::{entry.rust_test}" for entry in self.rust_only_tests)
        _require_unique(mapped_targets, "Rust mapping targets")
        _require_unique(rust_only_targets, "Rust-only test identities")
        overlap: Final = tuple(sorted(frozenset(mapped_targets) & frozenset(rust_only_targets)))
        if overlap:
            raise ValueError(f"Rust tests cannot be both mapped and rust-only: {', '.join(overlap)}")
        unscoped_python: Final = tuple(
            sorted(frozenset(entry.python_file for entry in self.entries) - frozenset(self.python_scope))
        )
        if unscoped_python:
            raise ValueError(f"entries reference files outside python_scope: {', '.join(unscoped_python)}")
        unscoped_rust: Final = tuple(
            sorted(
                (
                    frozenset(entry.rust_file for entry in self.entries if entry.status == "mapped")
                    | frozenset(entry.rust_file for entry in self.rust_only_tests)
                )
                - frozenset(self.rust_scope)
            )
        )
        if unscoped_rust:
            raise ValueError(f"entries reference files outside rust_scope: {', '.join(unscoped_rust)}")
        return self

    @property
    def mapped_count(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "mapped")

    @property
    def python_only_count(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "python_only")

    @property
    def unresolved_portable_count(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "unresolved_portable")

    @property
    def portable_count(self) -> int:
        return self.mapped_count + self.unresolved_portable_count

    @property
    def total_count(self) -> int:
        return len(self.entries)

    @property
    def percentage(self) -> float:
        if self.portable_count == 0:
            return 0.0
        return round(100.0 * self.mapped_count / self.portable_count, 1)


def load_ledger(path: Path) -> TestLedger:
    return TestLedger.model_validate_json(path.read_text(encoding="utf-8"))
