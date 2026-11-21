from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sqlite3
import sys
import time
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.harness.evaluation import (
    EvalProgress,
    EvalRun,
    EvalTarget,
    EvalTask,
    PredicateCheck,
    aagent_eval,
)
from litellm.harness.types import Harness, Text, ToolCall, ToolResult


class _FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)


class TaskRow(_FrozenModel):
    id: str
    prompt: str
    expected: str


class TargetInfo(_FrozenModel):
    label: str
    harness: str
    model: str
    telemetry_complete: bool
    agent: str = "Agent"


class RunManifest(_FrozenModel):
    run_id: str
    created_at: float
    tasks_path: str
    config_path: str
    fingerprint: str
    trials: int
    total: int
    fixture: bool
    targets: tuple[TargetInfo, ...]
    tasks: tuple[TaskRow, ...]


class CaseRow(_FrozenModel):
    run_id: str
    task_id: str
    target: str
    trial: int
    status: str
    detail: str
    duration: float
    started_at: float
    text: str | None
    agent_cost: float | None
    judge_cost: float | None
    model_calls: int | None
    tool_calls: int | None


class EventRow(_FrozenModel):
    run_id: str
    sequence: int
    kind: str
    name: str | None = None
    call_id: str | None = None
    arguments: str | None = None
    output: str | None = None
    is_error: bool = False
    at: float


class TargetSummary(_FrozenModel):
    target: str
    completed: int
    total: int
    passed: int
    failed: int
    errors: int
    agent_cost: float | None
    judge_cost: float | None


@dataclass(frozen=True, slots=True)
class SuiteConfig:
    tasks: tuple[EvalTask, ...]
    targets: tuple[EvalTarget, ...]
    task_rows: tuple[TaskRow, ...]
    task_file: Path
    config_file: Path
    fingerprint: str
    fixture: bool
    telemetry_complete: frozenset[str]
    agent_name: str = "Agent"


def _default_task(row: TaskRow) -> EvalTask:
    return EvalTask(
        id=row.id,
        prompt=row.prompt,
        check=PredicateCheck(lambda result: result.text.strip().casefold() == row.expected.strip().casefold()),
    )


def _require_target(value: object) -> EvalTarget:
    if not isinstance(value, EvalTarget):
        raise TypeError("targets must contain SDK EvalTarget instances")
    return value


def load_config(tasks: Path, config: Path) -> SuiteConfig:
    task_file: Final = tasks.resolve()
    config_file: Final = config.resolve()
    with task_file.open(newline="", encoding="utf-8-sig") as source:
        reader: Final = csv.DictReader(source)
        if reader.fieldnames is None or not {"id", "prompt", "expected"}.issubset(reader.fieldnames):
            raise ValueError("tasks.csv needs id, prompt, expected columns")
        rows: Final = tuple(TaskRow.model_validate(row) for row in reader)
    if not rows or any(not row.id.strip() or not row.prompt.strip() for row in rows):
        raise ValueError("CSV must contain tasks with non-empty ids and prompts")
    if len(frozenset(row.id for row in rows)) != len(rows):
        raise ValueError("CSV task ids must be unique")
    spec: Final = importlib.util.spec_from_file_location("lens_evals", config_file)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load Python evaluation configuration")
    module: Final = importlib.util.module_from_spec(spec)
    import_path: Final = tuple(sys.path)
    sys.path.insert(0, str(config_file.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = import_path
    target_values: Final = TypeAdapter(tuple[object, ...]).validate_python(getattr(module, "targets", ()))
    if not target_values:
        raise ValueError("evals.py must define targets = [EvalTarget(...), ...]")
    target_items: Final = tuple(_require_target(target) for target in target_values)
    if len(frozenset(target.name for target in target_items)) != len(target_items):
        raise ValueError("Target labels must be unique")
    make_task: Final = getattr(module, "make_task", _default_task)
    if not callable(make_task):
        raise TypeError("make_task must be callable")
    task_values: Final = tuple(make_task(row) for row in rows)
    if any(not isinstance(task, EvalTask) for task in task_values):
        raise TypeError("make_task must return an SDK EvalTask")
    task_items: Final = tuple(task for task in task_values if isinstance(task, EvalTask))
    if tuple(task.id for task in task_items) != tuple(row.id for row in rows):
        raise ValueError("make_task must preserve each CSV task id")
    explicit_telemetry: Final = TypeAdapter(frozenset[str]).validate_python(
        getattr(module, "telemetry_complete", frozenset[str]())
    )
    known: Final = (
        frozenset(
            target.name for target in target_items if target.runner is None and target.harness is Harness.TOOL_LOOP
        )
        | explicit_telemetry
    )
    fingerprint: Final = hashlib.sha256(task_file.read_bytes() + b"\x00" + config_file.read_bytes()).hexdigest()
    agent_name: Final = TypeAdapter(str).validate_python(getattr(module, "agent_name", "Agent"))
    if not agent_name.strip():
        raise ValueError("agent_name must be a non-empty string")
    return SuiteConfig(
        task_items,
        target_items,
        rows,
        task_file,
        config_file,
        fingerprint,
        getattr(module, "fixture", False) is True,
        known,
        agent_name,
    )


def _json_rows(cursor: sqlite3.Cursor) -> tuple[str, ...]:
    rows: Final = TypeAdapter(tuple[tuple[str], ...]).validate_python(cursor.fetchall())
    return tuple(row[0] for row in rows)


def _harness_name(target: EvalTarget) -> str:
    if target.runner is None:
        return target.harness.value
    name: Final[object] = getattr(target.runner, "__name__", None)
    return name if isinstance(name, str) and name else "custom"


class RunStore:
    def __init__(self, directory: Path, manifest: RunManifest) -> None:
        self.directory = directory
        self.manifest = manifest
        self.run_id = manifest.run_id
        self._connection = sqlite3.connect(directory / "run.sqlite3")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._connection.execute("CREATE TABLE IF NOT EXISTS cases (id TEXT PRIMARY KEY, data TEXT NOT NULL)")
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS events (id TEXT, sequence INTEGER, data TEXT NOT NULL, PRIMARY KEY(id, sequence))"
        )
        self._connection.commit()

    @classmethod
    def create(cls, root: Path, config: SuiteConfig, trials: int = 1) -> RunStore:
        if trials < 1:
            raise ValueError("trials must be at least one")
        run_id: Final = "eval_" + uuid.uuid4().hex[:12]
        directory: Final = root.resolve() / run_id
        directory.mkdir(parents=True, mode=0o700)
        manifest: Final = RunManifest(
            run_id=run_id,
            created_at=time.time(),
            tasks_path=str(config.task_file),
            config_path=str(config.config_file),
            fingerprint=config.fingerprint,
            trials=trials,
            total=len(config.tasks) * len(config.targets) * trials,
            fixture=config.fixture,
            targets=tuple(
                TargetInfo(
                    label=target.name,
                    harness=_harness_name(target),
                    model=target.model,
                    telemetry_complete=target.name in config.telemetry_complete,
                    agent=config.agent_name,
                )
                for target in config.targets
            ),
            tasks=config.task_rows,
        )
        (directory / "manifest.json").write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return cls(directory, manifest)

    @classmethod
    def open(cls, root: Path, run_id: str) -> RunStore:
        if not run_id.startswith("eval_") or not run_id.removeprefix("eval_").isalnum():
            raise ValueError("Invalid run id")
        directory: Final = root.resolve() / run_id
        manifest: Final = RunManifest.model_validate_json((directory / "manifest.json").read_text(encoding="utf-8"))
        return cls(directory, manifest)

    @classmethod
    def list_runs(cls, root: Path) -> tuple[RunManifest, ...]:
        return tuple(
            sorted(
                (
                    RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
                    for path in root.glob("eval_*/manifest.json")
                ),
                key=lambda manifest: manifest.created_at,
                reverse=True,
            )
        )

    def close(self) -> None:
        self._connection.close()

    def rows(self) -> tuple[CaseRow, ...]:
        return tuple(
            CaseRow.model_validate_json(value)
            for value in _json_rows(self._connection.execute("SELECT data FROM cases ORDER BY rowid"))
        )

    def events(self, run_id: str) -> tuple[EventRow, ...]:
        return tuple(
            EventRow.model_validate_json(value)
            for value in _json_rows(
                self._connection.execute("SELECT data FROM events WHERE id = ? ORDER BY sequence", (run_id,))
            )
        )

    def completed(self) -> frozenset[tuple[str, str, int]]:
        return frozenset((row.task_id, row.target, row.trial) for row in self.rows())

    @property
    def completed_count(self) -> int:
        rows: Final = TypeAdapter(tuple[tuple[int], ...]).validate_python(
            self._connection.execute("SELECT COUNT(*) FROM cases").fetchall()
        )
        return rows[0][0]

    def save_progress(self, progress: EvalProgress) -> None:
        if progress.event is None:
            self._connection.execute("DELETE FROM events WHERE id = ?", (progress.run_id,))
            self._connection.commit()
            return
        events: Final = self.events(progress.run_id)
        event: Final = progress.event
        row: Final = EventRow(
            run_id=progress.run_id,
            sequence=len(events),
            kind=type(event).__name__,
            name=event.name if isinstance(event, ToolCall) else None,
            call_id=event.id if isinstance(event, (ToolCall, ToolResult)) else None,
            arguments=(
                json.dumps(dict(event.input), sort_keys=True, separators=(",", ":"))
                if isinstance(event, ToolCall)
                else None
            ),
            output=(
                event.output if isinstance(event, ToolResult) else event.delta if isinstance(event, Text) else None
            ),
            is_error=event.is_error if isinstance(event, ToolResult) else False,
            at=time.time(),
        )
        self._connection.execute(
            "INSERT INTO events (id, sequence, data) VALUES (?, ?, ?)",
            (row.run_id, row.sequence, row.model_dump_json()),
        )
        self._connection.commit()

    def save_result(self, result: EvalRun) -> CaseRow:
        target: Final = next(target for target in self.manifest.targets if target.label == result.target)
        event_rows: Final = self.events(result.run_id)
        row: Final = CaseRow(
            run_id=result.run_id,
            task_id=result.task_id,
            target=result.target,
            trial=result.trial,
            status=result.status.value,
            detail=result.detail,
            duration=result.duration,
            started_at=time.time() - result.duration,
            text=result.result.text if result.result is not None else None,
            agent_cost=result.cost if target.telemetry_complete else None,
            judge_cost=result.check.judge_cost if result.check is not None else 0.0,
            model_calls=(result.usage.calls if target.telemetry_complete and result.usage is not None else None),
            tool_calls=(sum(event.kind == "ToolCall" for event in event_rows) if target.telemetry_complete else None),
        )
        self._connection.execute(
            "INSERT INTO cases (id, data) VALUES (?, ?)",
            (row.run_id, row.model_dump_json()),
        )
        self._connection.commit()
        return row

    def summary(self) -> tuple[TargetSummary, ...]:
        rows: Final = self.rows()
        return tuple(
            _summary(target, rows, len(self.manifest.tasks) * self.manifest.trials) for target in self.manifest.targets
        )


def _summary(target: TargetInfo, rows: tuple[CaseRow, ...], total: int) -> TargetSummary:
    selected: Final = tuple(row for row in rows if row.target == target.label)
    return TargetSummary(
        target=target.label,
        completed=len(selected),
        total=total,
        passed=sum(row.status == "pass" for row in selected),
        failed=sum(row.status == "fail" for row in selected),
        errors=sum(row.status == "error" for row in selected),
        agent_cost=(
            sum(row.agent_cost for row in selected if row.agent_cost is not None)
            if selected and all(row.agent_cost is not None for row in selected)
            else None
        ),
        judge_cost=(
            sum(row.judge_cost for row in selected if row.judge_cost is not None)
            if selected and all(row.judge_cost is not None for row in selected)
            else None
        ),
    )


def repeated_calls(events: tuple[EventRow, ...]) -> tuple[tuple[str, str, int], ...]:
    counts: Final = Counter(
        (event.name or "tool", event.arguments or "{}") for event in events if event.kind == "ToolCall"
    )
    return tuple((name, arguments, count) for (name, arguments), count in counts.items() if count > 1)


async def run_suite(
    store: RunStore,
    config: SuiteConfig,
    *,
    max_concurrency: int = 2,
    on_progress: Callable[[EvalProgress], None] | None = None,
    on_result: Callable[[CaseRow], None] | None = None,
) -> None:
    if config.fingerprint != store.manifest.fingerprint:
        raise ValueError("CSV or evals.py changed since this run. Restore them to resume, or start a new run.")

    def progress(event: EvalProgress) -> None:
        store.save_progress(event)
        if on_progress is not None:
            on_progress(event)

    def result(event: EvalRun) -> None:
        row: Final = store.save_result(event)
        if on_result is not None:
            on_result(row)

    await aagent_eval(
        config.tasks,
        config.targets,
        trials=store.manifest.trials,
        max_concurrency=max_concurrency,
        suite_id=store.run_id,
        completed=store.completed(),
        on_progress=progress,
        on_result=result,
    )
