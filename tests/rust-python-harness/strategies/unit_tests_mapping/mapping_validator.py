from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...shared.unit_runners.python_runner import enumerate_python_tests
from ...shared.unit_runners.rust_runner import enumerate_rust_tests


class _MappingModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class TestMapping(_MappingModel):
    python: str
    rust: str

    @field_validator("python", "rust")
    @classmethod
    def validate_nodeid(cls, value: str) -> str:
        stripped: Final = value.strip()
        if "::" not in stripped:
            raise ValueError("must be a source path and test name separated by '::'")
        return stripped


class UnitParityExclusionSpec(_MappingModel):
    nodeid: str
    reason: str

    @field_validator("nodeid", "reason")
    @classmethod
    def validate_fields(cls, value: str) -> str:
        stripped: Final = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped


class MappingSuite(_MappingModel):
    python_scope: tuple[str, ...]
    unit_parity_scope: tuple[str, ...]
    unit_parity_exclusions: tuple[UnitParityExclusionSpec, ...] = ()
    rust_scope: tuple[str, ...]
    cargo_manifest: str
    cargo_filter: str
    cargo_package: str | None = None
    mappings: tuple[TestMapping, ...]

    @field_validator("python_scope", "unit_parity_scope", "rust_scope")
    @classmethod
    def validate_scope(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        stripped: Final = tuple(item.strip() for item in value)
        if any(not item for item in stripped):
            raise ValueError("must contain only non-empty source paths")
        return stripped

    @field_validator("cargo_manifest", "cargo_filter")
    @classmethod
    def validate_cargo_fields(cls, value: str) -> str:
        stripped: Final = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped

    @model_validator(mode="after")
    def validate_unit_parity_scope(self) -> MappingSuite:
        unknown: Final = frozenset(self.unit_parity_scope) - frozenset(self.python_scope)
        if unknown:
            raise ValueError(f"unit_parity_scope must be contained in python_scope: {sorted(unknown)}")
        return self


@dataclass(frozen=True, slots=True)
class MappingReport:
    python_tests: tuple[str, ...]
    rust_tests: tuple[str, ...]
    mapped_python_tests: tuple[str, ...]
    unmapped_python_tests: tuple[str, ...]
    rust_only_tests: tuple[str, ...]
    missing_python_tests: tuple[str, ...]
    missing_rust_tests: tuple[str, ...]
    duplicate_python_mappings: tuple[str, ...]
    invalid_unit_parity_exclusions: tuple[str, ...]

    @property
    def mapped_count(self) -> int:
        return len(self.mapped_python_tests)

    @property
    def total_count(self) -> int:
        return len(self.python_tests)

    @property
    def percentage(self) -> float:
        return 0.0 if not self.total_count else round(100.0 * self.mapped_count / self.total_count, 1)

    @property
    def is_valid(self) -> bool:
        return not (
            self.missing_python_tests
            or self.missing_rust_tests
            or self.duplicate_python_mappings
            or self.invalid_unit_parity_exclusions
        )


def _python_inventory(suite: MappingSuite, repo_root: Path) -> frozenset[str]:
    return frozenset(
        f"{source}::{test}" for source in suite.python_scope for test in enumerate_python_tests(repo_root, source)
    )


def _rust_inventory(suite: MappingSuite, repo_root: Path) -> frozenset[str]:
    return frozenset(
        f"{source}::{test}" for source in suite.rust_scope for test in enumerate_rust_tests(repo_root, source)
    )


def audit_mapping(suite: MappingSuite, repo_root: Path) -> MappingReport:
    python_tests: Final = _python_inventory(suite, repo_root)
    unit_parity_tests: Final = frozenset(
        f"{source}::{test}" for source in suite.unit_parity_scope for test in enumerate_python_tests(repo_root, source)
    )
    rust_tests: Final = _rust_inventory(suite, repo_root)
    mapped_python: Final = frozenset(mapping.python for mapping in suite.mappings)
    mapped_rust: Final = frozenset(mapping.rust for mapping in suite.mappings)
    duplicate_python: Final = tuple(
        sorted(nodeid for nodeid, count in Counter(mapping.python for mapping in suite.mappings).items() if count > 1)
    )
    return MappingReport(
        python_tests=tuple(sorted(python_tests)),
        rust_tests=tuple(sorted(rust_tests)),
        mapped_python_tests=tuple(sorted(python_tests & mapped_python)),
        unmapped_python_tests=tuple(sorted(python_tests - mapped_python)),
        rust_only_tests=tuple(sorted(rust_tests - mapped_rust)),
        missing_python_tests=tuple(sorted(mapped_python - python_tests)),
        missing_rust_tests=tuple(sorted(mapped_rust - rust_tests)),
        duplicate_python_mappings=duplicate_python,
        invalid_unit_parity_exclusions=tuple(
            sorted(
                exclusion.nodeid
                for exclusion in suite.unit_parity_exclusions
                if exclusion.nodeid not in unit_parity_tests
            )
        ),
    )
