from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

from .models import Coverage, HarnessCase, HarnessRun
from .rendering import StrategyRenderer

_INERT_COVERAGE: Final = frozenset({Coverage.PLANNED, Coverage.NOT_APPLICABLE})
UpdateCallback: TypeAlias = Callable[[HarnessRun], None]


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


class ModuleCaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage: Coverage
    module: str | None = None
    note: str = ""

    @property
    def configured(self) -> bool:
        return self.module is not None

    @model_validator(mode="after")
    def _validate_inert_coverage(self) -> ModuleCaseSpec:
        if self.coverage in _INERT_COVERAGE and self.module is not None:
            raise ValueError(f"{self.coverage.value} case cannot configure a module")
        if self.module is not None and not self.module.strip():
            raise ValueError("case module cannot be blank")
        return self


CaseSpec: TypeAlias = SuiteCaseSpec | ModuleCaseSpec


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
        runner_args: Sequence[str] = (),
    ) -> tuple[int, HarnessRun]: ...


class StrategyChecker(Protocol):
    def __call__(
        self, sdk_functions: frozenset[str], repo_root: Path
    ) -> tuple[CheckReport, ...]: ...


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    directory: Path
    case_spec: type[SuiteCaseSpec] | type[ModuleCaseSpec]
    run: StrategyRunner
    render: StrategyRenderer
    check: StrategyChecker | None = None
