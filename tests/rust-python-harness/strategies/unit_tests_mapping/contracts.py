from __future__ import annotations

from collections import Counter
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Self

from ...shared.unit_runners.rust_runner import RustTestIdentity, RustTestScope


class _ContractModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def _clean_unique(values: tuple[str, ...], field: str) -> tuple[str, ...]:
    cleaned: Final = tuple(value.strip().rstrip("/") for value in values)
    if not cleaned or any(not value for value in cleaned):
        raise ValueError(f"{field} must contain non-empty paths")
    duplicates: Final = tuple(value for value, count in Counter(cleaned).items() if count > 1)
    if duplicates:
        raise ValueError(f"{field} contains duplicates: {sorted(duplicates)}")
    return cleaned


def _selector_contains(parent: str, child: str) -> bool:
    return child == parent or child.startswith(f"{parent}/")


class TestMapping(_ContractModel):
    python: str
    rust: RustTestIdentity

    @field_validator("python")
    @classmethod
    def validate_python_nodeid(cls, value: str) -> str:
        stripped: Final = value.strip()
        if "::" not in stripped:
            raise ValueError("must be a source path and test name separated by '::'")
        return stripped


class UnitParityExclusionSpec(_ContractModel):
    nodeid: str
    reason: str

    @field_validator("nodeid", "reason")
    @classmethod
    def validate_fields(cls, value: str) -> str:
        stripped: Final = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped


class MappingSpec(_ContractModel):
    python_selectors: tuple[str, ...]
    rust_scope: tuple[RustTestScope, ...]
    mappings: tuple[TestMapping, ...]

    @field_validator("python_selectors")
    @classmethod
    def validate_python_selectors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _clean_unique(value, "python_selectors")

    @model_validator(mode="after")
    def validate_rust_scope(self) -> Self:
        if not self.rust_scope:
            raise ValueError("rust_scope must not be empty")
        targets: Final = tuple(scope.target.key for scope in self.rust_scope)
        duplicates: Final = tuple(target for target, count in Counter(targets).items() if count > 1)
        if duplicates:
            raise ValueError(f"rust_scope contains duplicate targets: {sorted(duplicates)}")
        return self


class UnitParitySpec(_ContractModel):
    python_selectors: tuple[str, ...]
    exclusions: tuple[UnitParityExclusionSpec, ...] = ()

    @field_validator("python_selectors")
    @classmethod
    def validate_python_selectors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _clean_unique(value, "unit parity python_selectors")

    @model_validator(mode="after")
    def validate_exclusions(self) -> Self:
        nodeids: Final = tuple(exclusion.nodeid for exclusion in self.exclusions)
        duplicates: Final = tuple(nodeid for nodeid, count in Counter(nodeids).items() if count > 1)
        if duplicates:
            raise ValueError(f"unit parity exclusions contain duplicate nodeids: {sorted(duplicates)}")
        return self


class RustUnitSpec(_ContractModel):
    cargo_manifest: str
    cargo_filter: str
    cargo_package: str | None = None

    @field_validator("cargo_manifest", "cargo_filter")
    @classmethod
    def validate_required_fields(cls, value: str) -> str:
        stripped: Final = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped

    @field_validator("cargo_package")
    @classmethod
    def validate_package(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped: Final = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string when provided")
        return stripped


class UnitTestContract(_ContractModel):
    mapping: MappingSpec
    unit_parity: UnitParitySpec
    rust: RustUnitSpec

    @model_validator(mode="after")
    def validate_unit_parity_scope(self) -> Self:
        unknown: Final = tuple(
            selector
            for selector in self.unit_parity.python_selectors
            if not any(_selector_contains(parent, selector) for parent in self.mapping.python_selectors)
        )
        if unknown:
            raise ValueError(f"unit parity selectors must be contained in mapping selectors: {sorted(unknown)}")
        return self
