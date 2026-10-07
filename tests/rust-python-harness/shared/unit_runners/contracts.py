from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from typing_extensions import Self

from ..reporting.models import SdkFunction


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
    unit_parity: UnitParitySpec
    rust: RustUnitSpec


OCR_CONTRACT: Final = UnitTestContract(
    unit_parity=UnitParitySpec(
        python_selectors=(
            "tests/test_litellm/llms/azure_ai/test_azure_document_intelligence_ocr_transformation.py",
            "tests/test_litellm/llms/mistral/ocr",
            "tests/test_litellm/llms/ocr",
            "tests/test_litellm/ocr",
        ),
        exclusions=(
            UnitParityExclusionSpec(
                nodeid="tests/test_litellm/ocr/test_rust_bridge.py::test_rust_toggles_flag",
                reason="This test asserts the process-level backend flag selected by the parity runner.",
            ),
        ),
    ),
    rust=RustUnitSpec(
        cargo_manifest="litellm-rust/Cargo.toml",
        cargo_filter="ocr",
    ),
)

UNIT_TEST_CONTRACTS: Final[Mapping[SdkFunction, UnitTestContract]] = MappingProxyType({"ocr": OCR_CONTRACT})
