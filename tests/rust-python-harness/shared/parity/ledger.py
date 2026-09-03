from __future__ import annotations

from itertools import groupby
from pathlib import Path
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator
from typing_extensions import Self

NonEmptyString: TypeAlias = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
LEDGER_CONFIG: Final = ConfigDict(extra="forbid", frozen=True, strict=True)


class RustTarget(BaseModel):
    model_config = LEDGER_CONFIG

    package: NonEmptyString
    name: NonEmptyString
    kind: Literal["lib", "bin", "test"]

    @property
    def key(self) -> str:
        return f"{self.package}/{self.kind}/{self.name}"


class RustTestIdentity(BaseModel):
    model_config = LEDGER_CONFIG

    target: RustTarget
    name: NonEmptyString

    @property
    def key(self) -> str:
        return f"{self.target.key}::{self.name}"


class RustTestScope(BaseModel):
    model_config = LEDGER_CONFIG

    target: RustTarget
    features: tuple[NonEmptyString, ...]
    default_features: bool
    modules: Annotated[tuple[NonEmptyString, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def validate_scope(self) -> Self:
        _require_unique(self.features, "Rust features")
        _require_unique(self.modules, "Rust modules")
        if any(module.endswith("::") for module in self.modules):
            raise ValueError("Rust modules must omit the trailing :: separator")
        overlaps: Final = tuple(
            f"{outer} includes {inner}"
            for outer in self.modules
            for inner in self.modules
            if inner.startswith(f"{outer}::")
        )
        if overlaps:
            raise ValueError(f"Rust modules overlap: {', '.join(overlaps)}")
        return self

    def contains(self, identity: RustTestIdentity) -> bool:
        return identity.target == self.target and any(
            identity.name.startswith(f"{module}::") for module in self.modules
        )


class MappedLedgerEntry(BaseModel):
    model_config = LEDGER_CONFIG

    python_file: NonEmptyString
    python_test: NonEmptyString
    status: Literal["mapped"]
    rust_file: NonEmptyString
    rust: RustTestIdentity
    justification: NonEmptyString


class UnmappedLedgerEntry(BaseModel):
    model_config = LEDGER_CONFIG

    python_file: NonEmptyString
    python_test: NonEmptyString
    status: Literal["python_only", "unresolved_portable"]
    reason: NonEmptyString


LedgerEntry: TypeAlias = Annotated[MappedLedgerEntry | UnmappedLedgerEntry, Field(discriminator="status")]


class RustOnlyEntry(BaseModel):
    model_config = LEDGER_CONFIG

    rust_file: NonEmptyString
    rust: RustTestIdentity
    reason: NonEmptyString


def _require_unique(targets: tuple[str, ...], field: str) -> None:
    duplicates: Final = tuple(target for target, group in groupby(sorted(targets)) if sum(1 for _ in group) > 1)
    if duplicates:
        raise ValueError(f"{field} contains duplicates: {', '.join(duplicates)}")


class TestLedger(BaseModel):
    model_config = LEDGER_CONFIG

    sdk_function: NonEmptyString
    python_scope: tuple[NonEmptyString, ...]
    rust_scope: tuple[RustTestScope, ...]
    entries: tuple[LedgerEntry, ...]
    rust_only_tests: tuple[RustOnlyEntry, ...]

    @model_validator(mode="after")
    def validate_mapping_structure(self) -> Self:
        _require_unique(self.python_scope, "python_scope")
        _require_unique(tuple(scope.target.key for scope in self.rust_scope), "rust_scope targets")
        _require_unique(
            tuple(f"{entry.python_file}::{entry.python_test}" for entry in self.entries),
            "Python test identities",
        )
        mapped_targets: Final = tuple(entry.rust.key for entry in self.entries if entry.status == "mapped")
        rust_only_targets: Final = tuple(entry.rust.key for entry in self.rust_only_tests)
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
                identity.key
                for identity in self.rust_tests
                if not any(scope.contains(identity) for scope in self.rust_scope)
            )
        )
        if unscoped_rust:
            raise ValueError(f"entries reference tests outside rust_scope: {', '.join(unscoped_rust)}")
        return self

    @property
    def rust_tests(self) -> frozenset[RustTestIdentity]:
        return frozenset(entry.rust for entry in self.entries if entry.status == "mapped") | frozenset(
            entry.rust for entry in self.rust_only_tests
        )

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
