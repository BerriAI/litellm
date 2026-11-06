import json
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    TypeVar,
    cast,  # noqa: TID251  # PrismaWrapper forwards transaction operations dynamically
)

from pydantic import JsonValue, TypeAdapter

from litellm._uuid import uuid
from litellm.models.workflow import (
    WorkflowCheckpoint,
    WorkflowConflict,
    WorkflowDraft,
    WorkflowExecution,
    WorkflowRunState,
    WorkflowSnapshot,
    WorkflowStep,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


class WorkflowTransaction(Protocol):
    async def query_raw(self, query: str, *args: object) -> object: ...

    async def execute_raw(self, query: str, *args: object) -> int: ...


class WorkflowDatabase(WorkflowTransaction, Protocol):
    def tx(self) -> AbstractAsyncContextManager[WorkflowTransaction]: ...


_DRAFTS: Final = TypeAdapter(tuple[WorkflowDraft, ...])
_SNAPSHOTS: Final = TypeAdapter(tuple[WorkflowSnapshot, ...])
_EXECUTIONS: Final = TypeAdapter(tuple[WorkflowExecution, ...])
_CHECKPOINTS: Final = TypeAdapter(tuple[WorkflowCheckpoint, ...])
_STEPS: Final = TypeAdapter(tuple[WorkflowStep, ...])
_DRAFT_COLUMNS: Final = "workflow_id, team_id, name, etag, bundle_json, contributors, published_snapshot_id"
_SNAPSHOT_COLUMNS: Final = "snapshot_id, workflow_id, team_id, name, bundle_json, published_by, created_at, origin"
_EXECUTION_COLUMNS: Final = (
    "run_id, snapshot_id, team_id, name, principal, dispatch_key, request_digest, request_json, state, "
    "deadline, lease_owner, lease_until, cancel_requested, checkpoints, response_json, error"
)
_STEP_COLUMNS: Final = (
    "step_id, run_id, node_id, state, input, output, attempts, deadline, started_at, finished_at, error, "
    "prompt_tokens, completion_tokens, response_cost, request_id"
)


_RowT: Final = TypeVar("_RowT")


def _first(rows: Sequence[_RowT]) -> _RowT | None:
    return rows[0] if rows else None


class WorkflowRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._database: Final = cast(  # cast-ok: Prisma exposes raw SQL and transaction methods dynamically
            WorkflowDatabase, prisma_client.writer_db
        )

    async def create(self, team_id: str, name: str, bundle_json: str, author: str) -> WorkflowDraft | WorkflowConflict:
        rows: Final = _DRAFTS.validate_python(
            await self._database.query_raw(
                'INSERT INTO "LiteLLM_WorkflowDefinition" '
                "(workflow_id, team_id, name, etag, bundle_json, contributors) "
                "VALUES ($1, $2, $3, $4, $5, ARRAY[$6]::text[]) ON CONFLICT (team_id, name) DO NOTHING "
                f"RETURNING {_DRAFT_COLUMNS}",
                str(uuid.uuid4()),
                team_id,
                name,
                str(uuid.uuid4()),
                bundle_json,
                author,
            )
        )
        return rows[0] if rows else WorkflowConflict(message="Workflow already exists")

    async def get(self, team_id: str, name: str) -> WorkflowDraft | None:
        return _first(
            _DRAFTS.validate_python(
                await self._database.query_raw(
                    f'SELECT {_DRAFT_COLUMNS} FROM "LiteLLM_WorkflowDefinition" WHERE team_id = $1 AND name = $2',
                    team_id,
                    name,
                )
            )
        )

    async def replace(
        self, team_id: str, name: str, etag: str, bundle_json: str, author: str
    ) -> WorkflowDraft | WorkflowConflict:
        rows: Final = _DRAFTS.validate_python(
            await self._database.query_raw(
                'UPDATE "LiteLLM_WorkflowDefinition" SET etag = $4, bundle_json = $5, '
                "contributors = ARRAY(SELECT DISTINCT unnest(contributors || ARRAY[$6]::text[])) "
                f"WHERE team_id = $1 AND name = $2 AND etag = $3 RETURNING {_DRAFT_COLUMNS}",
                team_id,
                name,
                etag,
                str(uuid.uuid4()),
                bundle_json,
                author,
            )
        )
        return rows[0] if rows else WorkflowConflict(message="Draft changed or does not exist", status_code=412)

    async def get_id(self, workflow_id: str) -> WorkflowDraft | None:
        return _first(
            _DRAFTS.validate_python(
                await self._database.query_raw(
                    f'SELECT {_DRAFT_COLUMNS} FROM "LiteLLM_WorkflowDefinition" WHERE workflow_id = $1',
                    workflow_id,
                )
            )
        )

    async def list(self, team_id: str, after: str | None, limit: int) -> tuple[WorkflowDraft, ...]:
        return _DRAFTS.validate_python(
            await self._database.query_raw(
                f'SELECT {_DRAFT_COLUMNS} FROM "LiteLLM_WorkflowDefinition" '
                "WHERE team_id = $1 AND workflow_id > $2 ORDER BY workflow_id LIMIT $3",
                team_id,
                after or "",
                limit,
            )
        )

    async def publish(self, team_id: str, name: str, etag: str, publisher: str) -> WorkflowSnapshot | WorkflowConflict:
        async with self._database.tx() as tx:
            drafts: Final = _DRAFTS.validate_python(
                await tx.query_raw(
                    f'SELECT {_DRAFT_COLUMNS} FROM "LiteLLM_WorkflowDefinition" '
                    "WHERE team_id = $1 AND name = $2 FOR UPDATE",
                    team_id,
                    name,
                )
            )
            if not drafts or drafts[0].etag != etag:
                return WorkflowConflict(message="Draft changed or does not exist", status_code=412)
            if publisher in drafts[0].contributors:
                return WorkflowConflict(message="Draft contributors cannot publish their own draft", status_code=403)
            rows: Final = _SNAPSHOTS.validate_python(
                await tx.query_raw(
                    'INSERT INTO "LiteLLM_WorkflowSnapshot" '
                    "(snapshot_id, workflow_id, team_id, name, bundle_json, published_by) "
                    f"VALUES ($1, $2, $3, $4, $5, $6) RETURNING {_SNAPSHOT_COLUMNS}",
                    str(uuid.uuid4()),
                    drafts[0].workflow_id,
                    team_id,
                    name,
                    drafts[0].bundle_json,
                    publisher,
                )
            )
            await tx.execute_raw(
                'UPDATE "LiteLLM_WorkflowDefinition" SET published_snapshot_id = $3, etag = $4 '
                "WHERE team_id = $1 AND name = $2",
                team_id,
                name,
                rows[0].snapshot_id,
                str(uuid.uuid4()),
            )
            return rows[0]

    async def snapshot(self, snapshot_id: str) -> WorkflowSnapshot | None:
        return _first(
            _SNAPSHOTS.validate_python(
                await self._database.query_raw(
                    f'SELECT {_SNAPSHOT_COLUMNS} FROM "LiteLLM_WorkflowSnapshot" WHERE snapshot_id = $1',
                    snapshot_id,
                )
            )
        )

    async def published(self, team_id: str, name: str) -> WorkflowSnapshot | None:
        return _first(
            _SNAPSHOTS.validate_python(
                await self._database.query_raw(
                    f'SELECT {_SNAPSHOT_COLUMNS} FROM "LiteLLM_WorkflowSnapshot" WHERE snapshot_id = '
                    '(SELECT published_snapshot_id FROM "LiteLLM_WorkflowDefinition" WHERE team_id = $1 AND name = $2)',
                    team_id,
                    name,
                )
            )
        )

    async def pin_draft(self, workflow_id: str, etag: str, requested_by: str) -> WorkflowSnapshot | WorkflowConflict:
        rows: Final = _SNAPSHOTS.validate_python(
            await self._database.query_raw(
                'INSERT INTO "LiteLLM_WorkflowSnapshot" '
                "(snapshot_id, workflow_id, team_id, name, bundle_json, published_by, origin) "
                "SELECT $3, workflow_id, team_id, name, bundle_json, $4, 'draft_test' "
                'FROM "LiteLLM_WorkflowDefinition" WHERE workflow_id = $1 AND etag = $2 '
                f"RETURNING {_SNAPSHOT_COLUMNS}",
                workflow_id,
                etag,
                str(uuid.uuid4()),
                requested_by,
            )
        )
        return rows[0] if rows else WorkflowConflict(message="Draft changed or does not exist", status_code=412)

    async def by_dispatch_key(self, dispatch_key: str) -> WorkflowExecution | None:
        return _first(
            _EXECUTIONS.validate_python(
                await self._database.query_raw(
                    f'SELECT {_EXECUTION_COLUMNS} FROM "LiteLLM_WorkflowExecution" WHERE dispatch_key = $1',
                    dispatch_key,
                )
            )
        )

    async def execution(self, run_id: str) -> WorkflowExecution | None:
        return _first(
            _EXECUTIONS.validate_python(
                await self._database.query_raw(
                    f'SELECT {_EXECUTION_COLUMNS} FROM "LiteLLM_WorkflowExecution" WHERE run_id = $1',
                    run_id,
                )
            )
        )

    async def dispatch(self, execution: WorkflowExecution) -> WorkflowExecution:
        async with self._database.tx() as tx:
            await tx.query_raw(
                "SELECT 1 AS locked FROM pg_advisory_xact_lock(hashtextextended($1, 0))", execution.dispatch_key
            )
            existing: Final = _EXECUTIONS.validate_python(
                await tx.query_raw(
                    f'SELECT {_EXECUTION_COLUMNS} FROM "LiteLLM_WorkflowExecution" WHERE dispatch_key = $1',
                    execution.dispatch_key,
                )
            )
            if existing:
                return existing[0]
            await tx.execute_raw(
                'INSERT INTO "LiteLLM_WorkflowRun" '
                "(run_id, session_id, workflow_type, created_by, input, metadata, engine_owned, updated_at) "
                "VALUES ($1, $1, 'liteflow', $2, $3::jsonb, $4::jsonb, true, NOW())",
                execution.run_id,
                execution.principal,
                execution.request_json,
                json.dumps(
                    {
                        "snapshot_id": execution.snapshot_id,
                        "team_id": execution.team_id,
                        "workflow_name": execution.name,
                    }
                ),
            )
            rows: Final = _EXECUTIONS.validate_python(
                await tx.query_raw(
                    'INSERT INTO "LiteLLM_WorkflowExecution" '
                    "(run_id, snapshot_id, team_id, name, principal, dispatch_key, request_digest, request_json, deadline) "
                    f"VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9::timestamptz AT TIME ZONE 'UTC') RETURNING {_EXECUTION_COLUMNS}",
                    execution.run_id,
                    execution.snapshot_id,
                    execution.team_id,
                    execution.name,
                    execution.principal,
                    execution.dispatch_key,
                    execution.request_digest,
                    execution.request_json,
                    execution.deadline,
                )
            )
            return rows[0]

    async def lease_next(self, owner: str) -> WorkflowExecution | None:
        return _first(
            _EXECUTIONS.validate_python(
                await self._database.query_raw(
                    "UPDATE \"LiteLLM_WorkflowExecution\" SET lease_owner = $1, lease_until = NOW() + INTERVAL '30 seconds', "
                    "state = 'running' WHERE run_id = (SELECT run_id FROM \"LiteLLM_WorkflowExecution\" "
                    "WHERE state IN ('pending', 'running') AND (lease_until IS NULL OR lease_until < NOW()) "
                    f"ORDER BY deadline FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING {_EXECUTION_COLUMNS}",
                    owner,
                )
            )
        )

    async def renew(self, run_id: str, owner: str) -> bool:
        return bool(
            await self._database.execute_raw(
                "UPDATE \"LiteLLM_WorkflowExecution\" SET lease_until = NOW() + INTERVAL '30 seconds' "
                "WHERE run_id = $1 AND lease_owner = $2 AND lease_until > NOW() AND state = 'running'",
                run_id,
                owner,
            )
        )

    async def steps(self, run_id: str) -> tuple[WorkflowStep, ...]:
        return _STEPS.validate_python(
            await self._database.query_raw(
                f'SELECT {_STEP_COLUMNS} FROM "LiteLLM_WorkflowStep" WHERE run_id = $1 ORDER BY started_at, step_id',
                run_id,
            )
        )

    async def start_step(
        self, run_id: str, node_id: str, input: JsonValue, deadline: datetime, owner: str
    ) -> WorkflowStep | WorkflowConflict:
        rows: Final = _STEPS.validate_python(
            await self._database.query_raw(
                'INSERT INTO "LiteLLM_WorkflowStep" (step_id, run_id, node_id, input, deadline) '
                "SELECT $1,$2,$3,$4::jsonb,$5::timestamptz AT TIME ZONE 'UTC' WHERE EXISTS (SELECT 1 FROM \"LiteLLM_WorkflowExecution\" "
                "WHERE run_id=$2 AND lease_owner=$6 AND lease_until>NOW() AND NOT cancel_requested) "
                "ON CONFLICT (run_id, node_id) DO UPDATE SET node_id = EXCLUDED.node_id "
                f"RETURNING {_STEP_COLUMNS}",
                str(uuid.uuid4()),
                run_id,
                node_id,
                json.dumps(input, allow_nan=False),
                deadline,
                owner,
            )
        )
        return rows[0] if rows else WorkflowConflict(message="Execution lease lost")

    async def attempt_step(self, step_id: str, owner: str) -> WorkflowStep | WorkflowConflict:
        rows: Final = _STEPS.validate_python(
            await self._database.query_raw(
                'UPDATE "LiteLLM_WorkflowStep" SET attempts = attempts + 1 '
                'WHERE step_id = $1 AND EXISTS (SELECT 1 FROM "LiteLLM_WorkflowExecution" e '
                'WHERE e.run_id="LiteLLM_WorkflowStep".run_id AND e.lease_owner=$2 '
                f"AND e.lease_until>NOW() AND NOT e.cancel_requested) RETURNING {_STEP_COLUMNS}",
                step_id,
                owner,
            )
        )
        return rows[0] if rows else WorkflowConflict(message="Execution lease lost")

    async def _finish_step(self, database: WorkflowTransaction, step: WorkflowStep, owner: str) -> bool:
        return bool(
            await database.execute_raw(
                "UPDATE \"LiteLLM_WorkflowStep\" SET state=$2, output=$3::jsonb, finished_at=$4::timestamptz AT TIME ZONE 'UTC', error=$5, "
                "prompt_tokens=$6, completion_tokens=$7, response_cost=$8, request_id=$9 WHERE step_id=$1 "
                'AND EXISTS (SELECT 1 FROM "LiteLLM_WorkflowExecution" e '
                'WHERE e.run_id="LiteLLM_WorkflowStep".run_id AND e.lease_owner=$10 '
                "AND e.lease_until>NOW() AND NOT e.cancel_requested)",
                step.step_id,
                step.state.value,
                json.dumps(step.output, allow_nan=False),
                step.finished_at,
                step.error,
                step.prompt_tokens,
                step.completion_tokens,
                step.response_cost,
                step.request_id,
                owner,
            )
        )

    async def finish_step(self, step: WorkflowStep, owner: str) -> bool:
        return await self._finish_step(self._database, step, owner)

    async def checkpoint(self, run_id: str, checkpoint: WorkflowCheckpoint, step: WorkflowStep, owner: str) -> bool:
        async with self._database.tx() as tx:
            rows: Final = _EXECUTIONS.validate_python(
                await tx.query_raw(
                    f'SELECT {_EXECUTION_COLUMNS} FROM "LiteLLM_WorkflowExecution" '
                    "WHERE run_id = $1 AND lease_owner=$2 AND lease_until>NOW() AND NOT cancel_requested FOR UPDATE",
                    run_id,
                    owner,
                )
            )
            if not rows or not await self._finish_step(tx, step, owner):
                return False
            if any(item.node_path == checkpoint.node_path for item in rows[0].checkpoints):
                return True
            checkpoints: Final = (*rows[0].checkpoints, checkpoint)
            await tx.execute_raw(
                'UPDATE "LiteLLM_WorkflowExecution" SET checkpoints = $2::jsonb WHERE run_id = $1',
                run_id,
                _CHECKPOINTS.dump_json(checkpoints).decode(),
            )
            await tx.execute_raw(
                'INSERT INTO "LiteLLM_WorkflowEvent" '
                "(event_id, run_id, event_type, step_name, sequence_number, data) "
                "VALUES ($1,$2,'node.completed',$3,$4,$5::jsonb)",
                str(uuid.uuid4()),
                run_id,
                checkpoint.node_path,
                len(checkpoints) - 1,
                checkpoint.model_dump_json(),
            )
            return True

    async def finish(
        self,
        run_id: str,
        state: WorkflowRunState,
        response_json: str | None = None,
        error: str | None = None,
        *,
        owner: str,
    ) -> bool:
        async with self._database.tx() as tx:
            updated: Final = _EXECUTIONS.validate_python(
                await tx.query_raw(
                    "UPDATE \"LiteLLM_WorkflowExecution\" SET state = CASE WHEN cancel_requested THEN 'cancelled' ELSE $2 END, "
                    "response_json = CASE WHEN cancel_requested THEN NULL ELSE $3 END, error = $4 "
                    f"WHERE run_id = $1 AND state IN ($5,$6) AND lease_owner=$7 AND lease_until>NOW() RETURNING {_EXECUTION_COLUMNS}",
                    run_id,
                    state.value,
                    response_json,
                    error,
                    WorkflowRunState.RUNNING.value,
                    WorkflowRunState.PENDING.value,
                    owner,
                )
            )
            if not updated:
                return False
            await tx.execute_raw(
                'UPDATE "LiteLLM_WorkflowRun" SET status = $2, output = $3::jsonb, updated_at = NOW() WHERE run_id = $1',
                run_id,
                updated[0].state.value,
                json.dumps({"response": json.loads(updated[0].response_json) if updated[0].response_json else None}),
            )
            return True

    async def cancel(self, run_id: str) -> None:
        await self._database.execute_raw(
            'UPDATE "LiteLLM_WorkflowExecution" SET cancel_requested = true WHERE run_id = $1 AND state IN ($2,$3)',
            run_id,
            WorkflowRunState.PENDING.value,
            WorkflowRunState.RUNNING.value,
        )
