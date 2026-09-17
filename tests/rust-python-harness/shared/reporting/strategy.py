from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, StringConstraints

from .models import CaseDisposition, Coverage, HarnessCase, HarnessRun, SdkFunction, Surface
from .rendering import StrategyRenderer

UpdateCallback: TypeAlias = Callable[[HarnessRun], None]
NonBlankString: TypeAlias = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SuiteCaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disposition: Literal[CaseDisposition.RUNNABLE] = CaseDisposition.RUNNABLE
    coverage: Coverage
    suite: NonBlankString
    note: str = ""


class ModuleCaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disposition: Literal[CaseDisposition.RUNNABLE] = CaseDisposition.RUNNABLE
    coverage: Coverage
    module: NonBlankString
    note: str = ""


class NotImplementedCaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disposition: Literal[CaseDisposition.NOT_IMPLEMENTED] = CaseDisposition.NOT_IMPLEMENTED
    coverage: None = None
    reason: NonBlankString


class SkippedCaseSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    disposition: Literal[CaseDisposition.SKIPPED] = CaseDisposition.SKIPPED
    coverage: None = None
    reason: NonBlankString


RunnableCaseSpec: TypeAlias = SuiteCaseSpec | ModuleCaseSpec
UnavailableCaseSpec: TypeAlias = NotImplementedCaseSpec | SkippedCaseSpec
CaseSpec: TypeAlias = RunnableCaseSpec | UnavailableCaseSpec


@dataclass(frozen=True, slots=True)
class CaseDefinition:
    sdk_function: SdkFunction
    spec: CaseSpec
    surface: Surface | None = None


@dataclass(frozen=True, slots=True)
class RunnerArgumentDefinition:
    option: str
    help: str
    metavar: str = "ARG"


class StrategyRunner(Protocol):
    def __call__(
        self,
        cases: Sequence[HarnessCase],
        repo_root: Path,
        on_update: UpdateCallback,
        runner_args: Sequence[str] = (),
    ) -> tuple[int, HarnessRun]: ...


@dataclass(frozen=True, slots=True)
class StrategyDefinition:
    id: str
    order: int
    label: str
    description: str
    directory: Path
    runnable_spec: type[SuiteCaseSpec] | type[ModuleCaseSpec]
    cases: tuple[CaseDefinition, ...]
    run: StrategyRunner
    render: StrategyRenderer
    surfaces: tuple[Surface, ...] = ()
    runner_argument: RunnerArgumentDefinition | None = None
