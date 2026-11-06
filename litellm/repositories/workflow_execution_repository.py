import base64
from collections.abc import Mapping
from datetime import datetime, timezone
from itertools import chain
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from litellm.models.workflow_definition import WorkflowSnapshot
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
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import (
    SpendLogsRepository,
    TeamRepository,
    WorkflowEventRepository,
    WorkflowRunRepository,
)
from litellm.repositories.workflow_definition_repository import persist_snapshot

if TYPE_CHECKING:
    from prisma import Prisma
    from prisma import models as prisma_models


class _Cursor(BaseModel):
    model_config = ConfigDict(frozen=True)

    created_at: datetime
    run_id: str


class _TraceRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    trace: StepTrace


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


def _owned(run_id: str, token: str) -> Mapping[str, object]:
    return {
        "run_id": run_id,
        "engine_owned": True,
        "lease_token": token,
        "status": RunState.RUNNING.value,
        "lease_expires_at": {"gt": datetime.now(timezone.utc)},
    }


async def _lock_team(table: TableActions["prisma_models.LiteLLM_TeamTable"], team_id: str) -> bool:
    # A no-op model update serializes team admission until commit. Compare the
    # observed timestamp so a concurrent team edit never has its audit time reset.
    for _ in range(3):
        team = await table.find_unique(where={"team_id": team_id})
        if team is None:
            return False
        if await table.update_many(
            where={"team_id": team_id, "updated_at": team.updated_at}, data={"updated_at": team.updated_at}
        ):
            return True
    return False


class WorkflowExecutionRepository(WorkflowRunRepository):
    async def find_idempotent(
        self, workflow_id: str, team_id: str, key_hash: str, idempotency_key: str
    ) -> EngineRun | None:
        row: Final = await self.table.find_first(
            where={
                "engine_owned": True,
                "workflow_id": workflow_id,
                "team_id": team_id,
                "created_by": key_hash,
                "idempotency_key": idempotency_key,
            }
        )
        return None if row is None else EngineRun.model_validate(row, from_attributes=True)

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
        draft_snapshot: WorkflowSnapshot | None = None,
    ) -> RunAdmission | ExecutionError:
        from prisma import Json

        database: Final[Prisma] = self.writer_db
        async with database.tx() as transaction:
            if not await _lock_team(TeamRepository.transaction_table(transaction), team_id):
                return ExecutionError("storage_error", "Workflow team is unavailable. Retry the request")
            if idempotency_key is not None:
                existing: Final = await self.transaction_table(transaction).find_first(
                    where={
                        "engine_owned": True,
                        "created_by": key_hash,
                        "team_id": team_id,
                        "workflow_id": workflow_id,
                        "idempotency_key": idempotency_key,
                    }
                )
                if existing is not None:
                    return (
                        RunAdmission(EngineRun.model_validate(existing, from_attributes=True), False)
                        if existing.request_digest == digest
                        else ExecutionError("idempotency_conflict", "Idempotency key was used with different input")
                    )
            capacity: Final = await self.transaction_table(transaction).count(
                where={"engine_owned": True, "team_id": team_id, "status": {"in": ["pending", "running"]}},
                take=64,
            )
            if capacity >= 64:
                return ExecutionError("capacity_exceeded", "The team's workflow execution queue is full")
            if draft_snapshot is not None:
                await persist_snapshot(transaction, draft_snapshot)
            now: Final = datetime.now(timezone.utc)
            data: Final = {
                "created_at": now,
                "updated_at": now,
                "workflow_type": "liteflow",
                "workflow_id": workflow_id,
                "snapshot_id": snapshot_id,
                "team_id": team_id,
                "created_by": key_hash,
                "engine_owned": True,
                "input": Json(input.model_dump(mode="json")),
                "request_digest": digest,
                "idempotency_key": idempotency_key,
            }
            created: Final = await self.transaction_table(transaction).create(data=data)
            return RunAdmission(EngineRun.model_validate(created, from_attributes=True), True)

    async def get(self, run_id: str) -> EngineRun | None:
        row: Final = await self.table.find_first(where={"run_id": run_id, "engine_owned": True})
        return None if row is None else EngineRun.model_validate(row, from_attributes=True)

    async def recover(self, limit: int) -> tuple[EngineRun, ...]:
        rows: Final = await self.table.find_many(
            where={
                "engine_owned": True,
                "OR": [
                    {"status": RunState.PENDING.value},
                    {"status": RunState.RUNNING.value, "lease_expires_at": {"lt": datetime.now(timezone.utc)}},
                ],
            },
            order=[{"created_at": "asc"}, {"run_id": "asc"}],
            take=limit,
        )
        return tuple(EngineRun.model_validate(row, from_attributes=True) for row in rows)

    async def claim(self, run_id: str, token: str, lease_expires_at: datetime) -> EngineRun | None:
        database: Final[Prisma] = self.writer_db
        async with database.tx() as transaction:
            claimed: Final = await self.transaction_table(transaction).update_many(
                where={
                    "run_id": run_id,
                    "engine_owned": True,
                    "OR": [
                        {"status": RunState.PENDING.value},
                        {"status": RunState.RUNNING.value, "lease_expires_at": {"lt": datetime.now(timezone.utc)}},
                    ],
                },
                data={"status": RunState.RUNNING.value, "lease_token": token, "lease_expires_at": lease_expires_at},
            )
            if not claimed:
                return None
            row: Final = await self.transaction_table(transaction).find_unique(where={"run_id": run_id})
            return None if row is None else EngineRun.model_validate(row, from_attributes=True)

    async def renew(self, run_id: str, token: str, lease_expires_at: datetime) -> bool:
        return bool(
            await self.table.update_many(where=_owned(run_id, token), data={"lease_expires_at": lease_expires_at})
        )

    async def checkpoint(self, run_id: str, token: str, step: StepTrace) -> bool:
        from prisma import Json

        database: Final[Prisma] = self.writer_db
        async with database.tx() as transaction:
            where: Final = dict(_owned(run_id, token))
            if step.state is RunState.RUNNING:
                where["cancel_requested"] = False
            # The conditional update locks the run while allocating its event
            # sequence. Cancellation fences dispatch but still accepts receipts.
            if not await self.transaction_table(transaction).update_many(where=where, data={"lease_token": token}):
                return False
            last: Final = await WorkflowEventRepository.transaction_table(transaction).find_first(
                where={"run_id": run_id}, order={"sequence_number": "desc"}
            )
            data: Final = {
                "created_at": datetime.now(timezone.utc),
                "run_id": run_id,
                "event_type": "liteflow.step",
                "step_name": step.node_id,
                "sequence_number": 0 if last is None else last.sequence_number + 1,
                "data": Json({"trace": step.model_dump(mode="json")}),
            }
            await WorkflowEventRepository.transaction_table(transaction).create(data=data)
            return True

    async def steps(self, run_id: str) -> tuple[StepTrace, ...]:
        run: Final = await self.get(run_id)
        if run is None:
            return ()
        events: Final = await WorkflowEventRepository(self._prisma_client, use_writer=True).table.find_many(
            where={"run_id": run_id, "event_type": "liteflow.step"}, order={"sequence_number": "asc"}
        )
        latest: Final[dict[tuple[str, int], StepTrace]] = {}  # mutable-ok: local attempt index
        for event in events:
            step = _TraceRow.model_validate(event.data).trace
            latest[(step.node_id, step.attempts)] = step
        call_ids: Final = [step.call_id for step in latest.values() if step.call_id is not None]
        receipts: Final[dict[str, list[prisma_models.LiteLLM_SpendLogs]]] = {}  # mutable-ok: local receipt grouping
        if call_ids:
            logs: Final = await SpendLogsRepository(self._prisma_client, use_writer=True).table.find_many(
                where={
                    "litellm_call_id": {"in": call_ids},
                    "api_key": run.created_by,
                    "team_id": run.team_id,
                    "session_id": run.session_id,
                },
                order={"request_id": "asc"},
            )
            for log in logs:
                if log.litellm_call_id is not None:
                    receipts.setdefault(log.litellm_call_id, []).append(log)
        attempts: Final[dict[str, list[StepTrace]]] = {}  # mutable-ok: local attempt grouping before immutable result
        for step in sorted(latest.values(), key=lambda trace: (trace.started_at, trace.node_id)):
            billed = receipts.get(step.call_id) if step.call_id is not None else None
            if billed:
                step = step.model_copy(
                    update={
                        "request_id": billed[0].request_id if len(billed) == 1 else None,
                        "request_ids": tuple(log.request_id for log in billed),
                        "response_cost": sum(log.spend for log in billed),
                        "prompt_tokens": sum(log.prompt_tokens for log in billed),
                        "completion_tokens": sum(log.completion_tokens for log in billed),
                    }
                )
            attempts.setdefault(step.node_id, []).append(step)
        return tuple(_aggregate_attempts(tuple(node_attempts)) for node_attempts in attempts.values())

    async def finish(self, run_id: str, token: str, state: RunState, output: JsonValue, error: str | None) -> bool:
        from prisma import Json

        return bool(
            await self.table.update_many(
                where=_owned(run_id, token),
                data={
                    "status": state.value,
                    "output": Json(output),
                    "metadata": Json({"error": error}),
                    "lease_token": None,
                    "lease_expires_at": None,
                },
            )
        )

    async def cancel(self, run_id: str) -> None:
        await self.table.update_many(
            where={"run_id": run_id, "engine_owned": True, "status": {"in": ["pending", "running"]}},
            data={"cancel_requested": True},
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
        where: Final[dict[str, object]] = {  # mutable-ok: build a local Prisma filter from optional fields
            "engine_owned": True,
            "workflow_id": workflow_id,
            "team_id": team_id,
        }
        if key_hash is not None:
            where["created_by"] = key_hash
        if state is not None:
            where["status"] = state.value
        if position is not None:
            where["OR"] = [
                {"created_at": {"lt": position.created_at}},
                {"created_at": position.created_at, "run_id": {"lt": position.run_id}},
            ]
        rows: Final = await self.table.find_many(
            where=where, order=[{"created_at": "desc"}, {"run_id": "desc"}], take=limit + 1
        )
        page: Final = tuple(RunView.model_validate(row, from_attributes=True) for row in rows[:limit])
        return RunPage(runs=page, count=len(page), next_cursor=_encoded_cursor(page[-1]) if len(rows) > limit else None)
