import re
from collections.abc import AsyncGenerator, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm.models.workflow_execution import (
    EngineRun,
    ExecutionError,
    RunAdmission,
    RunInput,
    RunPage,
    RunState,
    StepTrace,
)
from litellm.repositories.workflow_execution_repository import ExecutionQueries, WorkflowExecutionRepository

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
_STRING: Final = TypeAdapter(str)
_INTEGER: Final = TypeAdapter(int)
_JSON: Final[TypeAdapter[JsonValue]] = TypeAdapter(JsonValue)
_TRACE: Final = TypeAdapter(dict[str, StepTrace])


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
            messages=({"role": "user", "content": "plan"},),
            end_user_id="end-user",
            deadline=deadline,
            draft=True,
        ),
        request_digest="request-digest",
    )


@dataclass(frozen=True, slots=True)
class Query:
    sql: str
    args: tuple[object, ...]
    in_transaction: bool


class ExecutionRows:
    def __init__(self, runs: tuple[EngineRun, ...] = ()) -> None:
        self.runs = runs
        self.keys: tuple[tuple[str, str | None], ...] = ()
        self.events: tuple[StepTrace, ...] = ()
        self.queries: tuple[Query, ...] = ()
        self.in_transaction = False

    @asynccontextmanager
    async def tx(self) -> AsyncGenerator[ExecutionQueries]:
        self.in_transaction = True
        try:
            yield self
        finally:
            self.in_transaction = False

    def _rows(self, runs: tuple[EngineRun, ...]) -> tuple[Mapping[str, object], ...]:
        return tuple(run.model_dump() for run in runs)

    def _replace(self, run: EngineRun) -> tuple[Mapping[str, object], ...]:
        self.runs = tuple(run if current.run_id == run.run_id else current for current in self.runs)
        return self._rows((run,))

    async def query_raw(self, query: str, *args: object) -> object:
        self.queries = (*self.queries, Query(query, args, self.in_transaction))
        parameters: Final = frozenset(int(match.group(1)) for match in re.finditer(r"\$(\d+)", query))
        assert parameters == frozenset(range(1, len(args) + 1)), (query, args)
        if "pg_advisory_xact_lock" in query:
            assert query.startswith("SELECT true AS acquired FROM "), "Prisma cannot decode a PostgreSQL void column"
            return ({"acquired": True},)
        if "idempotency_key =" in query:
            owner, team, workflow, key = args if "created_by = $1" in query else (args[2], args[1], args[0], args[3])
            return self._rows(
                tuple(
                    run
                    for run in self.runs
                    if key is not None
                    and (run.created_by, run.team_id, run.workflow_id) == (owner, team, workflow)
                    and (run.run_id, key) in self.keys
                )
            )
        if query.startswith('INSERT INTO "LiteLLM_WorkflowRun"'):
            run: Final = EngineRun.model_validate(
                {
                    "run_id": args[0],
                    "session_id": args[1],
                    "created_by": args[2],
                    "workflow_id": args[3],
                    "snapshot_id": args[4],
                    "team_id": args[5],
                    "input": RunInput.model_validate_json(_STRING.validate_python(args[6])),
                    "request_digest": args[7],
                    "status": RunState.PENDING,
                    "created_at": NOW,
                    "updated_at": NOW,
                    "output": None,
                    "metadata": None,
                    "lease_token": None,
                    "lease_expires_at": None,
                }
            )
            self.runs = (*self.runs, run)
            self.keys = (*self.keys, (run.run_id, None if args[8] is None else _STRING.validate_python(args[8])))
            return self._rows((run,))
        if "LIMIT 64" in query:
            return tuple(
                {"run_id": run.run_id}
                for run in self.runs
                if run.team_id == args[0] and run.status in {RunState.PENDING, RunState.RUNNING}
            )[:64]
        if "ORDER BY created_at DESC, run_id DESC" in query:
            boundary: Final = None if args[4] is None else datetime.fromisoformat(_STRING.validate_python(args[4]))
            matching: Final = tuple(
                run
                for run in self.runs
                if run.workflow_id == args[0]
                and run.team_id == args[1]
                and (args[2] is None or run.created_by == args[2])
                and (args[3] is None or run.status.value == args[3])
                and (boundary is None or (run.created_at, run.run_id) < (boundary, _STRING.validate_python(args[5])))
            )
            return self._rows(
                tuple(sorted(matching, key=lambda run: (run.created_at, run.run_id), reverse=True))[
                    : _INTEGER.validate_python(args[6])
                ]
            )
        if query.startswith("SELECT trace FROM"):
            latest: Final = {(step.node_id, step.attempts): step for step in self.events if step.run_id == args[0]}
            return tuple(
                {"trace": step.model_dump(mode="json")}
                for step in sorted(latest.values(), key=lambda step: (step.started_at, step.node_id))
            )
        if query.startswith('INSERT INTO "LiteLLM_WorkflowEvent"'):
            payload: Final = _TRACE.validate_json(_STRING.validate_python(args[3]))
            self.events = (*self.events, payload["trace"])
            return ({"event_id": args[0]},)
        assert query.startswith(
            (
                'SELECT * FROM "LiteLLM_WorkflowRun" WHERE run_id',
                'SELECT run_id FROM "LiteLLM_WorkflowRun" WHERE run_id',
                'UPDATE "LiteLLM_WorkflowRun"',
            )
        ), query
        current: Final = next((run for run in self.runs if run.run_id == args[0]), None)
        if current is None:
            return ()
        if query.startswith('SELECT * FROM "LiteLLM_WorkflowRun" WHERE run_id'):
            return self._rows((current,))
        if "SET status = 'running'" in query:
            if current.status is not RunState.PENDING:
                return ()
            return self._replace(
                current.model_copy(
                    update={
                        "status": RunState.RUNNING,
                        "lease_token": args[1],
                        "lease_expires_at": datetime.fromisoformat(_STRING.validate_python(args[2])),
                    }
                )
            )
        if "SET cancel_requested = true" in query:
            return self._replace(current.model_copy(update={"cancel_requested": True}))
        owned: Final = (
            current.status is RunState.RUNNING
            and current.lease_token == args[1]
            and current.lease_expires_at is not None
            and current.lease_expires_at > NOW
        )
        if "FOR UPDATE" in query:
            return ({"run_id": current.run_id},) if owned else ()
        if "SET status = $3" in query:
            if not owned:
                return ()
            return self._replace(
                current.model_copy(
                    update={
                        "status": RunState(args[2]),
                        "output": _JSON.validate_json(_STRING.validate_python(args[3])),
                        "metadata": {"error": args[4]},
                        "lease_token": None,
                        "lease_expires_at": None,
                    }
                )
            )
        raise AssertionError(f"Unexpected database operation: {query}")


@pytest.mark.asyncio
async def test_idempotent_admission_pins_snapshot_and_rejects_different_input() -> None:
    database: Final = ExecutionRows()
    repository: Final = WorkflowExecutionRepository(database)
    original: Final = pending_run()
    key: Final = "key'; SELECT secret"
    first: Final = await repository.admit(
        workflow_id=original.workflow_id,
        snapshot_id="snapshot-1",
        team_id=original.team_id,
        key_hash=original.created_by,
        input=original.input,
        digest=original.request_digest,
        idempotency_key=key,
    )
    replay: Final = await repository.admit(
        workflow_id=original.workflow_id,
        snapshot_id="snapshot-2",
        team_id=original.team_id,
        key_hash=original.created_by,
        input=original.input,
        digest=original.request_digest,
        idempotency_key=key,
    )
    conflict: Final = await repository.admit(
        workflow_id=original.workflow_id,
        snapshot_id="snapshot-2",
        team_id=original.team_id,
        key_hash=original.created_by,
        input=original.input,
        digest="different",
        idempotency_key=key,
    )
    assert isinstance(first, RunAdmission)
    assert replay == RunAdmission(first.run, False)
    assert first.created and first.run.snapshot_id == "snapshot-1"
    assert conflict == ExecutionError("idempotency_conflict", "Idempotency key was used with different input")
    assert len(database.runs) == 1
    assert all(query.in_transaction for query in database.queries)
    assert database.queries[0].args == (original.team_id,)
    assert database.queries[0].sql.startswith("SELECT true AS acquired FROM pg_advisory_xact_lock(")
    assert all("key'; SELECT secret" not in query.sql for query in database.queries)
    insert: Final = next(query for query in database.queries if query.sql.startswith("INSERT"))
    assert "timezone('UTC', NOW()), timezone('UTC', NOW())" in insert.sql
    assert insert.args[2:6] == (original.created_by, original.workflow_id, "snapshot-1", original.team_id)
    assert RunInput.model_validate_json(_STRING.validate_python(insert.args[6])) == original.input
    assert insert.args[7:] == (original.request_digest, key)


@pytest.mark.asyncio
async def test_null_idempotency_and_nullable_database_columns_round_trip_without_coalescing_runs() -> None:
    database: Final = ExecutionRows()
    repository: Final = WorkflowExecutionRepository(database)
    original: Final = pending_run()
    first: Final = await repository.admit(
        workflow_id=original.workflow_id,
        snapshot_id=original.snapshot_id,
        team_id=original.team_id,
        key_hash=original.created_by,
        input=original.input,
        digest=original.request_digest,
        idempotency_key=None,
    )
    second: Final = await repository.admit(
        workflow_id=original.workflow_id,
        snapshot_id=original.snapshot_id,
        team_id=original.team_id,
        key_hash=original.created_by,
        input=original.input,
        digest=original.request_digest,
        idempotency_key=None,
    )
    assert isinstance(first, RunAdmission) and isinstance(second, RunAdmission)
    assert first.created and second.created and first.run.run_id != second.run.run_id
    assert (first.run.output, first.run.metadata, first.run.lease_token, first.run.lease_expires_at) == (None,) * 4
    lookup: Final = next(query for query in database.queries if "idempotency_key =" in query.sql)
    assert lookup.args == (original.created_by, original.team_id, original.workflow_id, None)
    assert "idempotency_key = $4::text" in lookup.sql
    assert await repository.claim(first.run.run_id, "lease", NOW + timedelta(seconds=15)) is not None
    claim: Final = database.queries[-1]
    assert claim.args == (first.run.run_id, "lease", (NOW + timedelta(seconds=15)).isoformat())
    assert "lease_expires_at = timezone('UTC', $3::text::timestamptz)" in claim.sql
    assert await repository.finish(first.run.run_id, "lease", RunState.COMPLETED, {"value": None}, None)
    finish: Final = database.queries[-1]
    assert finish.args == (first.run.run_id, "lease", "completed", '{"value": null}', None)
    assert "$4::jsonb" in finish.sql and "$5::text" in finish.sql
    finished: Final = await repository.get(first.run.run_id)
    assert finished is not None
    assert (finished.output, finished.metadata, finished.lease_token) == ({"value": None}, {"error": None}, None)


@pytest.mark.asyncio
async def test_history_cursor_uses_timestamp_and_id_with_lookahead_and_scoped_parameters() -> None:
    database: Final = ExecutionRows(
        (
            pending_run("run-c"),
            pending_run("run-a"),
            pending_run("run-b"),
            pending_run("foreign-team", team_id="other-team"),
            pending_run("foreign-workflow", workflow_id="other-workflow"),
            pending_run("foreign-key", key_hash="other-key"),
        )
    )
    repository: Final = WorkflowExecutionRepository(database)
    first: Final = await repository.list_runs("workflow-1", "team", "owner-key", RunState.PENDING, 2, None)
    assert isinstance(first, RunPage)
    assert tuple(run.run_id for run in first.runs) == ("run-c", "run-b")
    assert first.count == 2 and first.next_cursor is not None
    assert database.queries[-1].args == ("workflow-1", "team", "owner-key", "pending", None, None, 3)
    database.runs = (*database.runs, pending_run("run-d"))
    tail: Final = await repository.list_runs("workflow-1", "team", "owner-key", RunState.PENDING, 2, first.next_cursor)
    assert isinstance(tail, RunPage)
    assert tuple(run.run_id for run in tail.runs) == ("run-a",)
    assert tail.count == 1 and tail.next_cursor is None
    query: Final = database.queries[-1]
    assert query.args == ("workflow-1", "team", "owner-key", "pending", NOW.isoformat(), "run-b", 3)
    assert "(created_at, run_id) < (timezone('UTC', $5::text::timestamptz), $6::text)" in query.sql
    assert "ORDER BY created_at DESC, run_id DESC LIMIT $7" in query.sql
    assert "workflow_id = $1 AND team_id = $2" in query.sql
    assert "created_by = $3::text" in query.sql and "status = $4::text" in query.sql
    assert "engine_owned = true" in query.sql


@pytest.mark.asyncio
async def test_invalid_cursor_never_queries_and_unfiltered_history_passes_sql_nulls() -> None:
    database: Final = ExecutionRows()
    repository: Final = WorkflowExecutionRepository(database)
    assert await repository.list_runs("workflow-1", "team", None, None, 1, "not-a-cursor") == ExecutionError(
        "invalid_cursor", "Invalid run cursor"
    )
    assert database.queries == ()
    assert await repository.list_runs("workflow-1", "team", None, None, 1, None) == RunPage(
        runs=(), count=0, next_cursor=None
    )
    query: Final = database.queries[-1]
    assert query.args == ("workflow-1", "team", None, None, None, None, 2)
    assert "($3::text IS NULL" in query.sql and "($4::text IS NULL" in query.sql and "($5::text IS NULL" in query.sql


@pytest.mark.asyncio
async def test_lost_lease_cannot_write_trace_or_terminal_result() -> None:
    database: Final = ExecutionRows((pending_run(),))
    repository: Final = WorkflowExecutionRepository(database)
    assert await repository.claim("run-1", "owner-lease", NOW + timedelta(seconds=15)) is not None
    step: Final = StepTrace(
        step_id="step-1",
        run_id="run-1",
        node_id="node-1",
        state=RunState.RUNNING,
        input=None,
        attempts=1,
        deadline=NOW + timedelta(seconds=10),
        started_at=NOW,
    )
    assert not await repository.checkpoint("run-1", "stale-lease", step)
    assert not await repository.finish("run-1", "stale-lease", RunState.COMPLETED, "forged", None)
    assert database.events == ()
    current: Final = await repository.get("run-1")
    assert current is not None and current.status is RunState.RUNNING and current.output is None
    ownership: Final = database.queries[1]
    assert ownership.in_transaction and ownership.args == ("run-1", "stale-lease")
    assert "lease_token = $2" in ownership.sql
    assert "lease_expires_at > timezone('UTC', NOW()) FOR UPDATE" in ownership.sql


@pytest.mark.asyncio
async def test_attempt_receipts_aggregate_without_counting_duplicate_checkpoints_twice() -> None:
    database: Final = ExecutionRows((pending_run(),))
    repository: Final = WorkflowExecutionRepository(database)
    assert await repository.claim("run-1", "lease", NOW + timedelta(seconds=15)) is not None
    first: Final = StepTrace(
        step_id="step-1",
        run_id="run-1",
        node_id="node-1",
        state=RunState.FAILED,
        input=None,
        attempts=1,
        deadline=NOW + timedelta(seconds=10),
        started_at=NOW,
        finished_at=NOW,
        prompt_tokens=4,
        completion_tokens=2,
        response_cost=0.125,
        request_id="failed-receipt",
        call_id="call-1",
        error="Provider failed after billing",
    )
    second: Final = first.model_copy(
        update={
            "state": RunState.COMPLETED,
            "attempts": 2,
            "output": {"done": True},
            "error": None,
            "prompt_tokens": 6,
            "completion_tokens": 3,
            "response_cost": 0.25,
            "request_id": "successful-receipt",
            "call_id": "call-2",
        }
    )
    for checkpoint in (first, first, second, second):
        assert await repository.checkpoint("run-1", "lease", checkpoint)
    traces: Final = await repository.steps("run-1")
    assert traces == (
        second.model_copy(
            update={
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "response_cost": 0.375,
                "request_ids": ("failed-receipt", "successful-receipt"),
            }
        ),
    )
    assert len(database.events) == 4
    query: Final = database.queries[-1]
    assert query.args == ("run-1",)
    assert "DISTINCT ON (step_name, data->'trace'->>'attempts')" in query.sql
