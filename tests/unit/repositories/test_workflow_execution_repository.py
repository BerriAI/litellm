import asyncio
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Final
from uuid import uuid4

import pytest

from litellm.models.workflow_definition import WorkflowSnapshot
from litellm.models.workflow_execution import (
    EngineRun,
    ExecutionError,
    RunAdmission,
    RunInput,
    RunPage,
    RunState,
    StepTrace,
)
from litellm.repositories import workflow_execution_repository
from litellm.repositories.workflow_definition_repository import WorkflowDefinitionRepository
from litellm.repositories.workflow_execution_repository import WorkflowExecutionRepository
from tests.unit.proxy.workflow_definition_rows import DefinitionRows, Rows, bundle

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW


@pytest.fixture(autouse=True)
def freeze_repository_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(workflow_execution_repository, "datetime", FixedDatetime)


def pending_run(
    run_id: str = "run-1",
    *,
    snapshot_id: str = "snapshot-1",
    workflow_id: str = "workflow-1",
    key_hash: str = "owner-key",
    team_id: str = "team",
    deadline: datetime = NOW + timedelta(seconds=90),
) -> EngineRun:
    return EngineRun(
        run_id=run_id,
        session_id=f"session-{run_id}",
        workflow_id=workflow_id,
        snapshot_id=snapshot_id,
        team_id=team_id,
        created_by=key_hash,
        status=RunState.PENDING,
        created_at=NOW,
        updated_at=NOW,
        input=RunInput(
            messages=({"role": "user", "content": "plan"},), end_user_id="end-user", deadline=deadline, draft=True
        ),
        request_digest="request-digest",
    )


def _data(data: Mapping[str, object]) -> dict[str, object]:
    from prisma import Json

    return {key: value.data if isinstance(value, Json) else value for key, value in data.items()}


class RunRows(Rows):
    fail_create = False

    async def create(self, data: Mapping[str, object], include=None):
        if self.fail_create:
            raise RuntimeError("Injected run write failure")
        return await super().create(
            data={
                "run_id": str(uuid4()),
                "session_id": str(uuid4()),
                "status": RunState.PENDING.value,
                "output": None,
                "metadata": None,
                "lease_token": None,
                "lease_expires_at": None,
                "cancel_requested": False,
                **_data(data),
            }
        )

    async def update_many(self, data: Mapping[str, object], where: Mapping[str, object]) -> int:
        return await super().update_many(data=_data(data), where=where)


class EventRows(Rows):
    async def create(self, data: Mapping[str, object], include=None):
        return await super().create(data={"event_id": str(uuid4()), **_data(data)})


class ExecutionRows:
    """Small model-action fake shared by endpoint and runner behavior tests."""

    def __init__(self, runs: tuple[EngineRun, ...] = ()) -> None:
        self.litellm_workflowrun = RunRows()
        self.litellm_workflowevent = EventRows()
        self.litellm_workflowsnapshot = Rows()
        self.litellm_spendlogs = Rows()
        self.litellm_teamtable = Rows(({"team_id": "team", "updated_at": NOW},))
        self.db = self
        self.writer_db = self
        self.lock = asyncio.Lock()
        self.runs = runs

    @property
    def runs(self) -> tuple[EngineRun, ...]:
        return tuple(EngineRun.model_validate(row) for row in self.litellm_workflowrun.rows)

    @runs.setter
    def runs(self, runs: tuple[EngineRun, ...]) -> None:
        self.litellm_workflowrun.rows = tuple({"engine_owned": True, **run.model_dump()} for run in runs)

    @property
    def events(self) -> tuple[StepTrace, ...]:
        return tuple(StepTrace.model_validate(row["data"]["trace"]) for row in self.litellm_workflowevent.rows)

    @events.setter
    def events(self, events: tuple[StepTrace, ...]) -> None:
        self.litellm_workflowevent.rows = tuple(
            {
                "run_id": step.run_id,
                "event_type": "liteflow.step",
                "step_name": step.node_id,
                "sequence_number": index,
                "data": {"trace": step.model_dump(mode="json")},
            }
            for index, step in enumerate(events)
        )

    @asynccontextmanager
    async def tx(self) -> AsyncGenerator["ExecutionRows"]:
        async with self.lock:
            runs = self.litellm_workflowrun.rows
            events = self.litellm_workflowevent.rows
            snapshots = self.litellm_workflowsnapshot.rows
            try:
                yield self
            except BaseException:
                self.litellm_workflowrun.rows = runs
                self.litellm_workflowevent.rows = events
                self.litellm_workflowsnapshot.rows = snapshots
                raise


async def admit(repository: WorkflowExecutionRepository, key: str | None = "key", **overrides):
    original = pending_run()
    return await repository.admit(
        **{
            "workflow_id": original.workflow_id,
            "snapshot_id": original.snapshot_id,
            "team_id": original.team_id,
            "key_hash": original.created_by,
            "input": original.input,
            "digest": original.request_digest,
            "idempotency_key": key,
            **overrides,
        }
    )


def trace(state: RunState = RunState.RUNNING, attempt: int = 1) -> StepTrace:
    return StepTrace(
        step_id="step-1",
        run_id="run-1",
        node_id="node-1",
        state=state,
        input=None,
        attempts=attempt,
        deadline=NOW + timedelta(seconds=30),
        started_at=NOW,
    )


@pytest.mark.asyncio
async def test_concurrent_idempotent_admission_pins_one_snapshot_and_rejects_different_input() -> None:
    database = ExecutionRows()
    repository = WorkflowExecutionRepository(database)
    first, replay = await asyncio.gather(admit(repository), admit(repository, snapshot_id="edited"))
    assert isinstance(first, RunAdmission) and isinstance(replay, RunAdmission)
    assert sorted((first.created, replay.created)) == [False, True]
    assert first.run == replay.run
    assert len(database.runs) == 1
    assert await repository.find_idempotent("workflow-1", "team", "owner-key", "key") == first.run
    assert await admit(repository, digest="different") == ExecutionError(
        "idempotency_conflict", "Idempotency key was used with different input"
    )


@pytest.mark.asyncio
async def test_null_idempotency_keys_create_distinct_runs_and_nullable_output_round_trips() -> None:
    repository = WorkflowExecutionRepository(ExecutionRows())
    first, second = await admit(repository, None), await admit(repository, None)
    assert isinstance(first, RunAdmission) and isinstance(second, RunAdmission)
    assert first.run.run_id != second.run.run_id
    assert first.run.output is None and first.run.metadata is None
    assert await repository.claim(first.run.run_id, "lease", NOW + timedelta(seconds=15)) is not None
    assert await repository.finish(first.run.run_id, "lease", RunState.COMPLETED, {"value": None}, None)
    finished = await repository.get(first.run.run_id)
    assert finished is not None and finished.output == {"value": None} and finished.metadata == {"error": None}


@pytest.mark.asyncio
async def test_concurrent_admission_never_exceeds_team_capacity_across_workflows() -> None:
    database = ExecutionRows(tuple(pending_run(f"run-{index}") for index in range(63)))
    repository = WorkflowExecutionRepository(database)
    outcomes = await asyncio.gather(admit(repository, "one"), admit(repository, "two", workflow_id="other"))
    assert sum(isinstance(outcome, RunAdmission) for outcome in outcomes) == 1
    assert sum(isinstance(outcome, ExecutionError) and outcome.code == "capacity_exceeded" for outcome in outcomes) == 1
    assert len(database.runs) == 64


@pytest.mark.asyncio
async def test_draft_snapshot_commits_with_admitted_run_only_and_rolls_back_failed_write() -> None:
    definitions = DefinitionRows()
    store = WorkflowDefinitionRepository(definitions.database)
    draft = await store.create(bundle(), "team", "author")
    snapshot = await store.prepare_draft(draft.workflow_id, "team", draft.etag)
    assert isinstance(snapshot, WorkflowSnapshot)
    database = ExecutionRows()
    database.litellm_workflowsnapshot = definitions.litellm_workflowsnapshot
    repository = WorkflowExecutionRepository(database)
    assert database.litellm_workflowsnapshot.rows == ()
    database.litellm_workflowrun.fail_create = True
    with pytest.raises(RuntimeError, match="Injected run write failure"):
        await admit(
            repository, draft_snapshot=snapshot, workflow_id=draft.workflow_id, snapshot_id=snapshot.snapshot_id
        )
    assert database.litellm_workflowsnapshot.rows == () and database.runs == ()
    database.litellm_workflowrun.fail_create = False
    admitted = await admit(
        repository, draft_snapshot=snapshot, workflow_id=draft.workflow_id, snapshot_id=snapshot.snapshot_id
    )
    assert isinstance(admitted, RunAdmission) and len(database.litellm_workflowsnapshot.rows) == 1
    next_snapshot = await store.prepare_draft(draft.workflow_id, "team", draft.etag)
    replay = await admit(
        repository, draft_snapshot=next_snapshot, workflow_id=draft.workflow_id, snapshot_id=next_snapshot.snapshot_id
    )
    assert replay == RunAdmission(admitted.run, False)
    assert len(database.litellm_workflowsnapshot.rows) == 1
    database.runs = tuple(pending_run(f"full-{index}") for index in range(64))
    refused = await admit(
        repository,
        "new-key",
        draft_snapshot=next_snapshot,
        workflow_id=draft.workflow_id,
        snapshot_id=next_snapshot.snapshot_id,
    )
    assert isinstance(refused, ExecutionError) and refused.code == "capacity_exceeded"
    assert len(database.litellm_workflowsnapshot.rows) == 1


@pytest.mark.asyncio
async def test_stale_worker_cannot_checkpoint_renew_or_finish_after_reclaim() -> None:
    expired = pending_run().model_copy(
        update={
            "status": RunState.RUNNING,
            "lease_token": "old",
            "lease_expires_at": NOW - timedelta(seconds=1),
        }
    )
    database = ExecutionRows((expired,))
    repository = WorkflowExecutionRepository(database)
    assert await repository.recover(1) == (expired,)
    assert await repository.claim("run-1", "new", NOW + timedelta(seconds=15)) is not None
    assert await repository.claim("run-1", "other", NOW + timedelta(seconds=15)) is None
    assert not await repository.renew("run-1", "old", NOW + timedelta(seconds=20))
    assert not await repository.checkpoint("run-1", "old", trace())
    assert not await repository.finish("run-1", "old", RunState.COMPLETED, "forged", None)
    assert database.events == ()


@pytest.mark.asyncio
async def test_cancellation_fences_new_dispatch_but_preserves_completed_receipts() -> None:
    database = ExecutionRows((pending_run(),))
    repository = WorkflowExecutionRepository(database)
    assert await repository.claim("run-1", "lease", NOW + timedelta(seconds=15)) is not None
    assert await repository.checkpoint("run-1", "lease", trace())
    await repository.cancel("run-1")
    assert not await repository.checkpoint("run-1", "lease", trace(attempt=2))
    completed = trace(RunState.COMPLETED).model_copy(update={"output": "paid-result", "response_cost": 0.25})
    assert await repository.checkpoint("run-1", "lease", completed)
    assert await repository.steps("run-1") == (completed.model_copy(update={"request_ids": ()}),)


@pytest.mark.asyncio
async def test_attempt_receipts_aggregate_once_and_remain_scoped_to_run_principal() -> None:
    database = ExecutionRows((pending_run(),))
    repository = WorkflowExecutionRepository(database)
    assert await repository.claim("run-1", "lease", NOW + timedelta(seconds=15)) is not None
    first = trace(RunState.FAILED).model_copy(
        update={
            "call_id": "call-1",
            "request_id": "failed",
            "response_cost": 0.125,
            "prompt_tokens": 4,
            "completion_tokens": 2,
        }
    )
    second = trace(RunState.COMPLETED, 2).model_copy(update={"call_id": "call-2", "output": "done"})
    await asyncio.gather(*(repository.checkpoint("run-1", "lease", step) for step in (first, first, second, second)))
    database.litellm_spendlogs.rows = (
        {
            "request_id": "paid",
            "litellm_call_id": "call-2",
            "api_key": "owner-key",
            "team_id": "team",
            "session_id": "session-run-1",
            "spend": 0.25,
            "prompt_tokens": 6,
            "completion_tokens": 3,
        },
        {
            "request_id": "foreign",
            "litellm_call_id": "call-2",
            "api_key": "other-key",
            "team_id": "team",
            "session_id": "session-run-1",
            "spend": 99,
            "prompt_tokens": 99,
            "completion_tokens": 99,
        },
    )
    (step,) = await repository.steps("run-1")
    assert step.attempts == 2 and step.output == "done"
    assert (step.prompt_tokens, step.completion_tokens, step.response_cost) == (10, 5, 0.375)
    assert step.request_ids == ("failed", "paid")
    assert len(database.events) == 4
    assert tuple(row["sequence_number"] for row in database.litellm_workflowevent.rows) == (0, 1, 2, 3)


@pytest.mark.asyncio
async def test_history_cursor_uses_timestamp_tiebreaker_and_principal_scope() -> None:
    database = ExecutionRows(
        (
            pending_run("run-c"),
            pending_run("run-a"),
            pending_run("run-b"),
            pending_run("foreign", team_id="other"),
            pending_run("foreign-key", key_hash="other"),
        )
    )
    repository = WorkflowExecutionRepository(database)
    first = await repository.list_runs("workflow-1", "team", "owner-key", RunState.PENDING, 2, None)
    assert isinstance(first, RunPage) and first.next_cursor is not None
    assert tuple(run.run_id for run in first.runs) == ("run-c", "run-b")
    tail = await repository.list_runs("workflow-1", "team", "owner-key", RunState.PENDING, 2, first.next_cursor)
    assert isinstance(tail, RunPage) and tail.next_cursor is None
    assert tuple(run.run_id for run in tail.runs) == ("run-a",)
    invalid = await repository.list_runs("workflow-1", "team", None, None, 2, "not-a-cursor")
    assert isinstance(invalid, ExecutionError) and invalid.code == "invalid_cursor"
