from __future__ import annotations

from collections import Counter
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Self

from ...shared.tracing.pytest_usage import PythonFunctionReference
from ...shared.unit_runners.rust_runner import RustTarget, RustTestIdentity, RustTestScope


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


class RustTestFamily(_ContractModel):
    kind: Literal["family"] = "family"
    target: RustTarget
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        stripped: Final = value.strip()
        if not stripped or stripped.endswith("::"):
            raise ValueError("must be a non-empty Rust test base name")
        return stripped

    @property
    def key(self) -> str:
        return f"{self.target.key}::{self.name}::case_*"

    def contains(self, identity: RustTestIdentity) -> bool:
        return identity.target == self.target and identity.name.startswith(f"{self.name}::case_")


class TestMapping(_ContractModel):
    python: str
    rust: RustTestIdentity | RustTestFamily

    @field_validator("python")
    @classmethod
    def validate_python_nodeid(cls, value: str) -> str:
        stripped: Final = value.strip()
        if "::" not in stripped:
            raise ValueError("must be a source path and test name separated by '::'")
        return stripped


class PythonFunctionDiscoverySpec(_ContractModel):
    functions: tuple[PythonFunctionReference, ...] = ()
    trace_module: str | None = None
    trace_spans: tuple[str, ...] = ()
    search_roots: tuple[str, ...]
    exclude_roots: tuple[str, ...] = ()
    includes: tuple[str, ...] = ()
    exclusions: tuple[str, ...] = ()

    @field_validator("search_roots")
    @classmethod
    def validate_search_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return _clean_unique(value, "python function search_roots")

    @field_validator("exclude_roots")
    @classmethod
    def validate_exclude_roots(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            return ()
        return _clean_unique(value, "python function exclude_roots")

    @model_validator(mode="after")
    def validate_functions(self) -> Self:
        if bool(self.functions) == bool(self.trace_module):
            raise ValueError("python function discovery needs exactly one function list or trace module")
        if self.trace_module is not None and not self.trace_spans:
            raise ValueError("trace-derived Python function discovery needs trace_spans")
        if not self.functions:
            return self
        keys: Final = tuple(f"{function.module}:{function.qualname}" for function in self.functions)
        duplicates: Final = tuple(key for key, count in Counter(keys).items() if count > 1)
        if duplicates:
            raise ValueError(f"python function discovery contains duplicates: {sorted(duplicates)}")
        return self


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


class MappingExclusionSpec(_ContractModel):
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
    python_selectors: tuple[str, ...] = ()
    python_functions: PythonFunctionDiscoverySpec | None = None
    rust_scope: tuple[RustTestScope, ...] = ()
    rust_targets: tuple[RustTarget, ...] = ()
    mappings: tuple[TestMapping, ...]
    exclusions: tuple[MappingExclusionSpec, ...] = ()
    require_complete: bool = False

    @field_validator("python_selectors")
    @classmethod
    def validate_python_selectors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            return ()
        return _clean_unique(value, "python_selectors")

    @model_validator(mode="after")
    def validate_rust_scope(self) -> Self:
        if bool(self.python_selectors) == bool(self.python_functions):
            raise ValueError("mapping needs exactly one Python selector or function-discovery scope")
        targets: Final = tuple(scope.target.key for scope in self.rust_scope)
        duplicates: Final = tuple(target for target, count in Counter(targets).items() if count > 1)
        if duplicates:
            raise ValueError(f"rust_scope contains duplicate targets: {sorted(duplicates)}")
        target_names: Final = tuple(target.name for target in self.rust_targets)
        duplicate_names: Final = tuple(name for name, count in Counter(target_names).items() if count > 1)
        if duplicate_names:
            raise ValueError(f"rust_targets contains duplicate names: {sorted(duplicate_names)}")
        exclusion_nodeids: Final = tuple(exclusion.nodeid for exclusion in self.exclusions)
        duplicate_exclusions: Final = tuple(nodeid for nodeid, count in Counter(exclusion_nodeids).items() if count > 1)
        if duplicate_exclusions:
            raise ValueError(f"mapping exclusions contain duplicate nodeids: {sorted(duplicate_exclusions)}")
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
        if not self.mapping.python_selectors:
            return self
        unknown: Final = tuple(
            selector
            for selector in self.unit_parity.python_selectors
            if not any(_selector_contains(parent, selector) for parent in self.mapping.python_selectors)
        )
        if unknown:
            raise ValueError(f"unit parity selectors must be contained in mapping selectors: {sorted(unknown)}")
        return self
