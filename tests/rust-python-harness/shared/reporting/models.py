from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, Final, Literal, TypeAlias, assert_never

if TYPE_CHECKING:
    from .strategy import CaseSpec, StrategyDefinition


class Coverage(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"


class CaseDisposition(str, Enum):
    RUNNABLE = "runnable"
    NOT_IMPLEMENTED = "not_implemented"
    SKIPPED = "skipped"


class RunStatus(str, Enum):
    NOT_RUN = "not_run"
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    MISSING = "missing"
    NOT_IMPLEMENTED = "not_implemented"


SdkFunction: TypeAlias = Literal["ocr", "messages", "responses", "count_tokens", "chat_completions", "transcription"]
Surface: TypeAlias = Literal["sdk", "gateway"]
SURFACES: Final[tuple[Surface, ...]] = ("sdk", "gateway")
SDK_FUNCTIONS: Final[tuple[SdkFunction, ...]] = (
    "ocr",
    "messages",
    "responses",
    "count_tokens",
    "chat_completions",
    "transcription",
)


@dataclass(frozen=True)
class HarnessCase:
    strategy_id: str
    strategy_label: str
    sdk_function: SdkFunction
    spec: CaseSpec
    surface: Surface | None = None

    @property
    def key(self) -> str:
        return (
            f"{self.strategy_id}:{self.sdk_function}"
            if self.surface in {None, "sdk"}
            else f"{self.strategy_id}:gateway:{self.sdk_function}"
        )

    @property
    def display_name(self) -> str:
        return self.sdk_function if self.surface is None else f"{self.surface}/{self.sdk_function}"

    @property
    def coverage(self) -> Coverage | None:
        return self.spec.coverage


@dataclass(frozen=True)
class Strategy:
    order: int
    id: str
    label: str
    description: str
    directory: Path
    cases: tuple[HarnessCase, ...]
    definition: StrategyDefinition


@dataclass
class CaseResult:
    case: HarnessCase
    status: RunStatus = RunStatus.NOT_RUN
    collected: set[str] = field(default_factory=set)
    completed: set[str] = field(default_factory=set)
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0
    outcomes: dict[str, RunStatus] = field(default_factory=dict)
    durations: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, tuple[ResultArtifact, ...]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.collected)

    @property
    def duration(self) -> float:
        return sum(self.durations.values())

    def record(
        self,
        nodeid: str,
        status: RunStatus,
        duration: float = 0.0,
        artifacts: tuple[ResultArtifact, ...] = (),
    ) -> None:
        """Record a terminal outcome, allowing teardown errors to replace a pass."""
        self.outcomes[nodeid] = status
        self.add_duration(nodeid, duration)
        if artifacts:
            self.artifacts[nodeid] = artifacts
        self.completed = set(self.outcomes)
        values = tuple(self.outcomes.values())
        self.passed = values.count(RunStatus.PASSED)
        self.failed = values.count(RunStatus.FAILED)
        self.skipped = values.count(RunStatus.SKIPPED)
        self.errors = values.count(RunStatus.ERROR)
        self.finalize()

    def add_duration(self, nodeid: str, duration: float) -> None:
        self.durations[nodeid] = self.durations.get(nodeid, 0.0) + duration

    def set_initial_status(self) -> None:
        disposition: Final = self.case.spec.disposition
        match disposition:
            case CaseDisposition.RUNNABLE:
                self.status = RunStatus.QUEUED
                return
            case CaseDisposition.NOT_IMPLEMENTED:
                self.status = RunStatus.NOT_IMPLEMENTED
                return
            case CaseDisposition.SKIPPED:
                self.status = RunStatus.SKIPPED
                return
        assert_never(disposition)

    def finalize(self) -> None:
        if self.status in {RunStatus.NOT_IMPLEMENTED, RunStatus.SKIPPED} and not self.collected:
            return
        if not self.collected:
            self.status = RunStatus.MISSING
        elif self.errors:
            self.status = RunStatus.ERROR
        elif self.failed:
            self.status = RunStatus.FAILED
        elif self.passed and len(self.completed) == len(self.collected):
            self.status = RunStatus.PASSED
        elif self.skipped and len(self.completed) == len(self.collected):
            self.status = RunStatus.SKIPPED


@dataclass(frozen=True, slots=True)
class ResultArtifact:
    kind: str
    body: str


@dataclass
class HarnessRun:
    results: dict[str, CaseResult]
    current_nodeid: str | None = None
    failures: list[tuple[str, str]] = field(default_factory=list)
    strategy_durations: dict[str, float] = field(default_factory=dict)
    started_at: float = field(default_factory=monotonic)
    finished_at: float | None = None

    @property
    def duration(self) -> float:
        return (self.finished_at or monotonic()) - self.started_at

    @property
    def unique_checks(self) -> int:
        return len(
            {nodeid for result in self.results.values() for nodeid in result.collected}
        )

    @property
    def completed_checks(self) -> int:
        return len(
            {nodeid for result in self.results.values() for nodeid in result.completed}
        )

    @classmethod
    def from_cases(cls, cases: Iterable[HarnessCase]) -> HarnessRun:
        results = {case.key: CaseResult(case=case) for case in cases}
        for result in results.values():
            result.set_initial_status()
        return cls(results=results)
