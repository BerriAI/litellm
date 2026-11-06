import base64
import json
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from itertools import chain
from typing import Final, Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter, ValidationError

from litellm.models.workflow_execution import (
    EngineRun,
    ExecutionError,
    RunAdmission,
    RunInput,
    RunPage,
    RunState,
    RunView,
    StepTrace,
)


class ExecutionQueries(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...


class ExecutionDatabase(ExecutionQueries, Protocol):
    def tx(self) -> AbstractAsyncContextManager[ExecutionQueries]: ...


class _Cursor(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: datetime
    run_id: str


class _TraceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace: StepTrace


_RUNS: Final = TypeAdapter(tuple[EngineRun, ...])
_VIEWS: Final = TypeAdapter(tuple[RunView, ...])
_TRACES: Final = TypeAdapter(tuple[_TraceRow, ...])
_ROWS: Final = TypeAdapter(tuple[dict[str, object], ...])


def _cursor(value: str | None) -> _Cursor | ExecutionError | None:
    if value is None:
        return None
    try:
        return _Cursor.model_validate_json(base64.urlsafe_b64decode(value.encode("ascii")))
    except (ValueError, UnicodeError, ValidationError):
        return ExecutionError("invalid_cursor", "Invalid run cursor")


def _encoded_cursor(run: RunView) -> str:
    return base64.urlsafe_b64encode(
        _Cursor(created_at=run.created_at, run_id=run.run_id).model_dump_json().encode()
    ).decode("ascii")


async def _first_run(database: ExecutionQueries, query: str, *args: object) -> EngineRun | None:
    rows: Final = _RUNS.validate_python(await database.query_raw(query, *args))
    return rows[0] if rows else None


def _aggregate_attempts(attempts: tuple[StepTrace, ...]) -> StepTrace:
    latest: Final = max(attempts, key=lambda step: step.attempts)
    costs: Final = tuple(step.response_cost for step in attempts if step.response_cost is not None)
    ids: Final = chain.from_iterable(
        step.request_ids or ((step.request_id,) if step.request_id is not None else ()) for step in attempts
    )
    return latest.model_copy(
        update={
            "prompt_tokens": sum(step.prompt_tokens for step in attempts),
            "completion_tokens": sum(step.completion_tokens for step in attempts),
            "response_cost": sum(costs) if costs else None,
            "request_ids": tuple(dict.fromkeys(ids)),
        }
    )


class WorkflowExecutionRepository:
    def __init__(self, database: ExecutionDatabase) -> None:
        self._database: Final = database

    async def find_idempotent(
        self, workflow_id: str, team_id: str, key_hash: str, idempotency_key: str
    ) -> EngineRun | None:
        return await _first_run(
            self._database,
            'SELECT * FROM "LiteLLM_WorkflowRun" WHERE engine_owned = true AND workflow_id = $1 '
            "AND team_id = $2 AND created_by = $3 AND idempotency_key = $4",
            workflow_id,
            team_id,
            key_hash,
            idempotency_key,
        )

    async def admit(
        self,
        *,
        workflow_id: str,
        snapshot_id: str,
        team_id: str,
        key_hash: str,
        input: RunInput,
        digest: str,
        idempotency_key: str | None,
    ) -> RunAdmission | ExecutionError:
        async with self._database.tx() as transaction:
            await transaction.query_raw(
                "SELECT true AS acquired FROM pg_advisory_xact_lock(hashtextextended($1, 0))", team_id
            )
            existing: Final = await _first_run(
                transaction,
                'SELECT * FROM "LiteLLM_WorkflowRun" WHERE engine_owned = true AND created_by = $1 '
                "AND team_id = $2 AND workflow_id = $3 AND idempotency_key = $4::text",
                key_hash,
                team_id,
                workflow_id,
                idempotency_key,
            )
            if existing is not None:
                return (
                    RunAdmission(existing, False)
                    if existing.request_digest == digest
                    else ExecutionError("idempotency_conflict", "Idempotency key was used with different input")
                )
            capacity: Final = _ROWS.validate_python(
                await transaction.query_raw(
                    'SELECT run_id FROM "LiteLLM_WorkflowRun" WHERE engine_owned = true AND team_id = $1 '
                    "AND status IN ('pending', 'running') LIMIT 64",
                    team_id,
                )
            )
            if len(capacity) >= 64:
                return ExecutionError("capacity_exceeded", "The team's workflow execution queue is full")
            created: Final = await _first_run(
                transaction,
                'INSERT INTO "LiteLLM_WorkflowRun" '
                "(run_id, session_id, workflow_type, status, created_by, created_at, updated_at, "
                "workflow_id, snapshot_id, team_id, engine_owned, input, request_digest, idempotency_key) "
                "VALUES ($1, $2, 'liteflow', 'pending', $3, timezone('UTC', NOW()), timezone('UTC', NOW()), $4, $5, $6, true, $7::jsonb, $8, $9) "
                "RETURNING *",
                str(uuid4()),
                str(uuid4()),
                key_hash,
                workflow_id,
                snapshot_id,
                team_id,
                input.model_dump_json(),
                digest,
                idempotency_key,
            )
            if created is None:
                return ExecutionError("storage_error", "Workflow admission did not create a run")
            return RunAdmission(created, True)

    async def get(self, run_id: str) -> EngineRun | None:
        return await _first_run(
            self._database,
            'SELECT * FROM "LiteLLM_WorkflowRun" WHERE run_id = $1 AND engine_owned = true',
            run_id,
        )

    async def recover(self, limit: int) -> tuple[EngineRun, ...]:
        return _RUNS.validate_python(
            await self._database.query_raw(
                'SELECT * FROM "LiteLLM_WorkflowRun" WHERE engine_owned = true AND '
                "(status = 'pending' OR (status = 'running' AND lease_expires_at < timezone('UTC', NOW()))) "
                "ORDER BY created_at, run_id LIMIT $1",
                limit,
            )
        )

    async def claim(self, run_id: str, token: str, lease_expires_at: datetime) -> EngineRun | None:
        return await _first_run(
            self._database,
            "UPDATE \"LiteLLM_WorkflowRun\" SET status = 'running', lease_token = $2, "
            "lease_expires_at = timezone('UTC', $3::text::timestamptz), updated_at = timezone('UTC', NOW()) "
            "WHERE run_id = $1 AND engine_owned = true AND "
            "(status = 'pending' OR (status = 'running' AND lease_expires_at < timezone('UTC', NOW()))) RETURNING *",
            run_id,
            token,
            lease_expires_at.isoformat(),
        )

    async def renew(self, run_id: str, token: str, lease_expires_at: datetime) -> bool:
        rows: Final = _ROWS.validate_python(
            await self._database.query_raw(
                "UPDATE \"LiteLLM_WorkflowRun\" SET lease_expires_at = timezone('UTC', $3::text::timestamptz) "
                "WHERE run_id = $1 AND lease_token = $2 AND status = 'running' "
                "AND lease_expires_at > timezone('UTC', NOW()) RETURNING run_id",
                run_id,
                token,
                lease_expires_at.isoformat(),
            )
        )
        return bool(rows)

    async def checkpoint(self, run_id: str, token: str, step: StepTrace) -> bool:
        async with self._database.tx() as transaction:
            owned: Final = _ROWS.validate_python(
                await transaction.query_raw(
                    'SELECT run_id FROM "LiteLLM_WorkflowRun" WHERE run_id = $1 AND lease_token = $2 '
                    "AND status = 'running' AND lease_expires_at > timezone('UTC', NOW()) FOR UPDATE",
                    run_id,
                    token,
                )
            )
            if not owned:
                return False
            await transaction.query_raw(
                'INSERT INTO "LiteLLM_WorkflowEvent" '
                "(event_id, run_id, event_type, step_name, sequence_number, data, created_at) "
                "SELECT $1, $2, 'liteflow.step', $3, COALESCE(MAX(sequence_number) + 1, 0), $4::jsonb, timezone('UTC', NOW()) "
                'FROM "LiteLLM_WorkflowEvent" WHERE run_id = $2 RETURNING event_id',
                str(uuid4()),
                run_id,
                step.node_id,
                json.dumps({"trace": step.model_dump(mode="json")}, allow_nan=False),
            )
            return True

    async def steps(self, run_id: str) -> tuple[StepTrace, ...]:
        rows: Final = _TRACES.validate_python(
            await self._database.query_raw(
                "SELECT trace FROM ("
                "SELECT CASE WHEN receipts.ids IS NULL THEN latest.trace ELSE latest.trace || "
                "jsonb_build_object('request_id', CASE WHEN cardinality(receipts.ids) = 1 THEN receipts.ids[1] END, "
                "'request_ids', to_jsonb(receipts.ids), 'response_cost', receipts.cost, "
                "'prompt_tokens', receipts.prompt_tokens, 'completion_tokens', receipts.completion_tokens) END AS trace, "
                "latest.step_name FROM ("
                "SELECT DISTINCT ON (step_name, data->'trace'->>'attempts') data->'trace' AS trace, step_name "
                "FROM \"LiteLLM_WorkflowEvent\" WHERE run_id = $1 AND event_type = 'liteflow.step' "
                "ORDER BY step_name, data->'trace'->>'attempts', sequence_number DESC"
                ') latest JOIN "LiteLLM_WorkflowRun" run ON run.run_id = $1 AND run.engine_owned = true '
                "LEFT JOIN LATERAL ("
                "SELECT array_agg(log.request_id ORDER BY log.request_id) AS ids, SUM(log.spend) AS cost, "
                "SUM(log.prompt_tokens) AS prompt_tokens, SUM(log.completion_tokens) AS completion_tokens "
                "FROM \"LiteLLM_SpendLogs\" log WHERE log.litellm_call_id = latest.trace->>'call_id' "
                "AND log.api_key = run.created_by AND log.team_id = run.team_id AND log.session_id = run.session_id"
                ") receipts ON true"
                ") latest ORDER BY (trace->>'started_at')::timestamptz, step_name",
                run_id,
            )
        )
        node_ids: Final = dict.fromkeys(row.trace.node_id for row in rows)
        return tuple(
            _aggregate_attempts(tuple(row.trace for row in rows if row.trace.node_id == node_id))
            for node_id in node_ids
        )

    async def finish(self, run_id: str, token: str, state: RunState, output: JsonValue, error: str | None) -> bool:
        rows: Final = _ROWS.validate_python(
            await self._database.query_raw(
                'UPDATE "LiteLLM_WorkflowRun" SET status = $3, output = $4::jsonb, '
                "metadata = jsonb_build_object('error', $5::text), lease_token = NULL, lease_expires_at = NULL, "
                "updated_at = timezone('UTC', NOW()) WHERE run_id = $1 AND lease_token = $2 AND status = 'running' "
                "AND lease_expires_at > timezone('UTC', NOW()) RETURNING run_id",
                run_id,
                token,
                state.value,
                json.dumps(output, allow_nan=False),
                error,
            )
        )
        return bool(rows)

    async def cancel(self, run_id: str) -> None:
        await self._database.query_raw(
            "UPDATE \"LiteLLM_WorkflowRun\" SET cancel_requested = true, updated_at = timezone('UTC', NOW()) "
            "WHERE run_id = $1 AND engine_owned = true AND status IN ('pending', 'running') RETURNING run_id",
            run_id,
        )

    async def list_runs(
        self,
        workflow_id: str,
        team_id: str,
        key_hash: str | None,
        state: RunState | None,
        limit: int,
        cursor: str | None,
    ) -> RunPage | ExecutionError:
        position: Final = _cursor(cursor)
        if isinstance(position, ExecutionError):
            return position
        rows: Final = _VIEWS.validate_python(
            await self._database.query_raw(
                'SELECT * FROM "LiteLLM_WorkflowRun" WHERE engine_owned = true '
                "AND workflow_id = $1 AND team_id = $2 "
                "AND ($3::text IS NULL OR created_by = $3::text) AND ($4::text IS NULL OR status = $4::text) "
                "AND ($5::text IS NULL OR (created_at, run_id) < (timezone('UTC', $5::text::timestamptz), $6::text)) "
                "ORDER BY created_at DESC, run_id DESC LIMIT $7",
                workflow_id,
                team_id,
                key_hash,
                state.value if state is not None else None,
                position.created_at.isoformat() if position is not None else None,
                position.run_id if position is not None else None,
                limit + 1,
            )
        )
        page: Final = rows[:limit]
        return RunPage(
            runs=page,
            count=len(page),
            next_cursor=_encoded_cursor(page[-1]) if len(rows) > limit else None,
        )
