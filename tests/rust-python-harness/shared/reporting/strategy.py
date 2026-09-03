from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

from .models import Coverage, HarnessCase, HarnessRun

if TYPE_CHECKING:
    from .pytest_runner import UpdateCallback

_INERT_COVERAGE: Final = frozenset({Coverage.PLANNED, Coverage.NOT_APPLICABLE})


class SelectorCaseSpec(BaseModel):
    """A pytest-driven strategy cell configured with pytest selectors."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: Coverage
    selectors: tuple[str, ...] = ()
    note: str = ""

    @property
    def configured(self) -> bool:
        return bool(self.selectors)

    @model_validator(mode="after")
    def _validate_inert_and_blank_selectors(self) -> SelectorCaseSpec:
        if self.coverage in _INERT_COVERAGE and self.selectors:
            raise ValueError(f"{self.coverage.value} case cannot configure tests")
        if any(not selector.strip() for selector in self.selectors):
            raise ValueError("case selectors cannot be blank")
        return self


class SuiteCaseSpec(BaseModel):
    """A JSON-suite-driven strategy cell configured with a suite path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: Coverage
    suite: str | None = None
    note: str = ""

    @property
    def configured(self) -> bool:
        return self.suite is not None

    @model_validator(mode="after")
    def _validate_inert_coverage(self) -> SuiteCaseSpec:
        if self.coverage in _INERT_COVERAGE and self.suite is not None:
            raise ValueError(f"{self.coverage.value} case cannot configure tests")
        return self


CaseSpec: TypeAlias = SelectorCaseSpec | SuiteCaseSpec


@dataclass(frozen=True, slots=True)
class CheckReport:
    sdk_function: str
    lines: tuple[str, ...]
    passed: bool


class StrategyRunner(Protocol):
    def __call__(
        self,
        cases: Sequence[HarnessCase],
        repo_root: Path,
        on_update: UpdateCallback,
        pytest_args: Sequence[str] = (),
    ) -> tuple[int, HarnessRun]: ...


class StrategyChecker(Protocol):
    def __call__(
        self, sdk_functions: frozenset[str], repo_root: Path
    ) -> tuple[CheckReport, ...]: ...


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    directory: Path
    case_spec: type[SelectorCaseSpec] | type[SuiteCaseSpec]
    run: StrategyRunner
    check: StrategyChecker | None = None
