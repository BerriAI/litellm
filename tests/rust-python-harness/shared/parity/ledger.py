from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class _LedgerModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")


class _PythonTestEntry(_LedgerModel):
    python_file: str
    python_test: str

    @field_validator("python_file", "python_test")
    @classmethod
    def validate_python_test_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped


class MappedLedgerEntry(_PythonTestEntry):
    status: Literal["mapped"]
    rust_file: str
    rust_test: str
    justification: str
    reason: Literal[""] = ""

    @field_validator("rust_file", "rust_test", "justification")
    @classmethod
    def validate_mapping_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped


class UnmappedLedgerEntry(_PythonTestEntry):
    status: Literal["unmapped"]
    reason: str
    rust_file: Literal[""] = ""
    rust_test: Literal[""] = ""
    justification: Literal[""] = ""

    @field_validator("reason")
    @classmethod
    def validate_reason(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped


LedgerEntry = Annotated[MappedLedgerEntry | UnmappedLedgerEntry, Field(discriminator="status")]


class RustOnlyEntry(_LedgerModel):
    rust_file: str
    rust_test: str
    reason: str

    @field_validator("rust_file", "rust_test", "reason")
    @classmethod
    def validate_fields(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped


class TestLedger(_LedgerModel):
    sdk_function: str
    python_scope: tuple[str, ...]
    rust_scope: tuple[str, ...]
    entries: tuple[LedgerEntry, ...]
    rust_only_tests: tuple[RustOnlyEntry, ...]

    @field_validator("sdk_function")
    @classmethod
    def validate_sdk_function(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("must be a non-empty string")
        return stripped

    @field_validator("python_scope", "rust_scope")
    @classmethod
    def validate_scope(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        stripped = tuple(item.strip() for item in value)
        if any(not item for item in stripped):
            raise ValueError("must contain only non-empty strings")
        return stripped

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


def load_ledger(path: Path) -> TestLedger:
    return TestLedger.model_validate_json(path.read_text(encoding="utf-8"))
