from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from time import monotonic
from typing import Iterable


class Coverage(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    PLANNED = "planned"
    NOT_APPLICABLE = "not_applicable"


class RunStatus(str, Enum):
    NOT_RUN = "not_run"
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"
    MISSING = "missing"
    PLANNED = "planned"
    NOT_APPLICABLE = "not_applicable"


SDK_FUNCTIONS = ("ocr", "messages", "responses", "count_tokens")


@dataclass(frozen=True)
class HarnessCase:
    strategy_id: str
    strategy_label: str
    sdk_function: str
    coverage: Coverage
    selectors: tuple[str, ...]
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.strategy_id}:{self.sdk_function}"


@dataclass(frozen=True)
class Strategy:
    order: int
    id: str
    label: str
    description: str
    directory: Path
    cases: tuple[HarnessCase, ...]


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

    @property
    def total(self) -> int:
        return len(self.collected)

    @property
    def duration(self) -> float:
        return sum(self.durations.values())

    def record(self, nodeid: str, status: RunStatus, duration: float = 0.0) -> None:
        """Record a terminal outcome, allowing teardown errors to replace a pass."""
        self.outcomes[nodeid] = status
        self.durations[nodeid] = self.durations.get(nodeid, 0.0) + duration
        self.completed = set(self.outcomes)
        values = tuple(self.outcomes.values())
        self.passed = values.count(RunStatus.PASSED)
        self.failed = values.count(RunStatus.FAILED)
        self.skipped = values.count(RunStatus.SKIPPED)
        self.errors = values.count(RunStatus.ERROR)
        self.finalize()

    def set_initial_status(self) -> None:
        if self.case.coverage is Coverage.NOT_APPLICABLE:
            self.status = RunStatus.NOT_APPLICABLE
        elif not self.case.selectors:
            self.status = RunStatus.PLANNED
        else:
            self.status = RunStatus.QUEUED

    def finalize(self) -> None:
        if self.status in {RunStatus.NOT_APPLICABLE, RunStatus.PLANNED}:
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


@dataclass
class HarnessRun:
    results: dict[str, CaseResult]
    current_nodeid: str | None = None
    failures: list[tuple[str, str]] = field(default_factory=list)
    started_at: float = field(default_factory=monotonic)
    finished_at: float | None = None

    @property
    def duration(self) -> float:
        return (self.finished_at or monotonic()) - self.started_at

    @property
    def unique_tests(self) -> int:
        return len(
            {nodeid for result in self.results.values() for nodeid in result.collected}
        )

    @property
    def completed_tests(self) -> int:
        return len(
            {nodeid for result in self.results.values() for nodeid in result.completed}
        )

    @classmethod
    def from_cases(cls, cases: Iterable[HarnessCase]) -> "HarnessRun":
        results = {case.key: CaseResult(case=case) for case in cases}
        for result in results.values():
            result.set_initial_status()
        return cls(results=results)
