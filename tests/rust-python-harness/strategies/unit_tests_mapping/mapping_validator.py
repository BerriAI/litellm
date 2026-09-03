from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, TypeAlias

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from ...shared.unit_runners.python_runner import collect_python_tests
from ...shared.unit_runners.rust_runner import RustTestIdentity, RustTestScope, enumerate_rust_tests

PythonInventory: TypeAlias = Callable[[Sequence[str], Path], frozenset[str]]
RustInventory: TypeAlias = Callable[[Path, tuple[RustTestScope, ...]], frozenset[RustTestIdentity]]


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
    python_selectors: tuple[str, ...]
    unit_parity_selectors: tuple[str, ...]
    unit_parity_exclusions: tuple[UnitParityExclusionSpec, ...] = ()
    rust_scope: tuple[RustTestScope, ...]
    cargo_manifest: str
    cargo_filter: str
    cargo_package: str | None = None
    mappings: tuple[TestMapping, ...]

    @field_validator("python_selectors", "unit_parity_selectors")
    @classmethod
    def validate_selectors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
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
        unknown: Final = frozenset(self.unit_parity_selectors) - frozenset(self.python_selectors)
        if unknown:
            raise ValueError(f"unit_parity_selectors must be contained in python_selectors: {sorted(unknown)}")
        targets: Final = tuple(scope.target.key for scope in self.rust_scope)
        duplicate_targets: Final = tuple(target for target, count in Counter(targets).items() if count > 1)
        if duplicate_targets:
            raise ValueError(f"rust_scope contains duplicate targets: {sorted(duplicate_targets)}")
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
    duplicate_rust_mappings: tuple[str, ...]
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
            or self.duplicate_rust_mappings
            or self.invalid_unit_parity_exclusions
        )


def _selected(nodeid: str, selectors: Sequence[str]) -> bool:
    source: Final = nodeid.partition("::")[0]
    return any(source == selector or source.startswith(f"{selector.rstrip('/')}/") for selector in selectors)


def audit_mapping(
    suite: MappingSuite,
    repo_root: Path,
    *,
    python_inventory: PythonInventory = collect_python_tests,
    rust_inventory: RustInventory = enumerate_rust_tests,
) -> MappingReport:
    python_tests: Final = python_inventory(suite.python_selectors, repo_root)
    unit_parity_tests: Final = frozenset(
        nodeid for nodeid in python_tests if _selected(nodeid, suite.unit_parity_selectors)
    )
    rust_tests: Final = frozenset(identity.key for identity in rust_inventory(repo_root, suite.rust_scope))
    mapped_python: Final = frozenset(mapping.python for mapping in suite.mappings)
    mapped_rust: Final = frozenset(mapping.rust for mapping in suite.mappings)
    duplicate_python: Final = tuple(
        sorted(nodeid for nodeid, count in Counter(mapping.python for mapping in suite.mappings).items() if count > 1)
    )
    duplicate_rust: Final = tuple(
        sorted(nodeid for nodeid, count in Counter(mapping.rust for mapping in suite.mappings).items() if count > 1)
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
        duplicate_rust_mappings=duplicate_rust,
        invalid_unit_parity_exclusions=tuple(
            sorted(
                exclusion.nodeid
                for exclusion in suite.unit_parity_exclusions
                if exclusion.nodeid not in unit_parity_tests
            )
        ),
    )
