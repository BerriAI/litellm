import json
from collections.abc import Mapping
from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta, timezone
from typing import (
    TYPE_CHECKING,
    Final,
    Protocol,
    cast,  # noqa: TID251  # PrismaWrapper exposes generated delegates dynamically
)
from uuid import NAMESPACE_URL, uuid5

from prisma import Json
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
from litellm.repositories.prisma_protocols import TableActions

if TYPE_CHECKING:
    from prisma import models as prisma_models

    from litellm.proxy.utils import PrismaClient


_JSON: Final = TypeAdapter[JsonValue](JsonValue)


def _json(encoded: str) -> Json:
    return Json(_JSON.validate_json(encoded))


class WorkflowTransaction(Protocol):
    @property
    def litellm_workflowdefinition(self) -> TableActions["prisma_models.LiteLLM_WorkflowDefinition"]: ...

    @property
    def litellm_workflowsnapshot(self) -> TableActions["prisma_models.LiteLLM_WorkflowSnapshot"]: ...

    @property
    def litellm_workflowexecution(self) -> TableActions["prisma_models.LiteLLM_WorkflowExecution"]: ...

    @property
    def litellm_workflowstep(self) -> TableActions["prisma_models.LiteLLM_WorkflowStep"]: ...

    @property
    def litellm_workflowrun(self) -> TableActions["prisma_models.LiteLLM_WorkflowRun"]: ...

    @property
    def litellm_workflowevent(self) -> TableActions["prisma_models.LiteLLM_WorkflowEvent"]: ...


class WorkflowDatabase(WorkflowTransaction, Protocol):
    def tx(self) -> AbstractAsyncContextManager[WorkflowTransaction]: ...


def _lease_where(run_id: str, owner: str, *, allow_cancel: bool = False) -> Mapping[str, object]:
    return {
        "run_id": run_id,
        "lease_owner": owner,
        "lease_until": {"gt": datetime.now(timezone.utc)},
        "state": "running",
        **({} if allow_cancel else {"cancel_requested": False}),
    }


async def _lock_execution(
    database: WorkflowTransaction, run_id: str, owner: str, *, allow_cancel: bool = False
) -> bool:
    return bool(
        await database.litellm_workflowexecution.update_many(
            where=_lease_where(run_id, owner, allow_cancel=allow_cancel), data={"lease_owner": owner}
        )
    )


def _step_data(step: WorkflowStep) -> Mapping[str, object]:
    return {
        "state": step.state.value,
        "output": _json(json.dumps(step.output, allow_nan=False)),
        "finished_at": step.finished_at,
        "error": step.error,
        "prompt_tokens": step.prompt_tokens,
        "completion_tokens": step.completion_tokens,
        "response_cost": step.response_cost,
        "request_id": step.request_id,
    }


class WorkflowRepository:
    def __init__(self, prisma_client: "PrismaClient") -> None:
        self._database: Final = cast(  # cast-ok: Prisma exposes generated model delegates and transactions dynamically
            WorkflowDatabase, prisma_client.writer_db
        )

    async def create(self, team_id: str, name: str, bundle_json: str, author: str) -> WorkflowDraft | WorkflowConflict:
        workflow_id: Final = str(uuid.uuid4())
        inserted: Final = await self._database.litellm_workflowdefinition.create_many(
            data=[
                {
                    "workflow_id": workflow_id,
                    "team_id": team_id,
                    "name": name,
                    "etag": str(uuid.uuid4()),
                    "bundle_json": bundle_json,
                    "contributors": [author],
                }
            ],
            skip_duplicates=True,
        )
        if not inserted:
            return WorkflowConflict(message="Workflow already exists")
        row: Final = await self._database.litellm_workflowdefinition.find_unique(where={"workflow_id": workflow_id})
        return WorkflowDraft.model_validate(row, from_attributes=True)

    async def get(self, team_id: str, name: str) -> WorkflowDraft | None:
        row: Final = await self._database.litellm_workflowdefinition.find_unique(
            where={"team_id_name": {"team_id": team_id, "name": name}}
        )
        return WorkflowDraft.model_validate(row, from_attributes=True) if row is not None else None

    async def get_id(self, workflow_id: str) -> WorkflowDraft | None:
        row: Final = await self._database.litellm_workflowdefinition.find_unique(where={"workflow_id": workflow_id})
        return WorkflowDraft.model_validate(row, from_attributes=True) if row is not None else None

    async def list(self, team_id: str, after: str | None, limit: int) -> tuple[WorkflowDraft, ...]:
        rows: Final = await self._database.litellm_workflowdefinition.find_many(
            where={"team_id": team_id, "workflow_id": {"gt": after or ""}},
            order={"workflow_id": "asc"},
            take=limit,
        )
        return tuple(WorkflowDraft.model_validate(row, from_attributes=True) for row in rows)

    async def replace(
        self, team_id: str, name: str, etag: str, bundle_json: str, author: str
    ) -> WorkflowDraft | WorkflowConflict:
        async with self._database.tx() as tx:
            previous: Final = await tx.litellm_workflowdefinition.find_first(
                where={"team_id": team_id, "name": name, "etag": etag}
            )
            if previous is None:
                return WorkflowConflict(message="Draft changed or does not exist", status_code=412)
            updated: Final = await tx.litellm_workflowdefinition.update_many(
                where={"workflow_id": previous.workflow_id, "etag": etag},
                data={
                    "etag": str(uuid.uuid4()),
                    "bundle_json": bundle_json,
                    "contributors": list(dict.fromkeys((*previous.contributors, author))),
                },
            )
            if not updated:
                return WorkflowConflict(message="Draft changed or does not exist", status_code=412)
            row: Final = await tx.litellm_workflowdefinition.find_unique(where={"workflow_id": previous.workflow_id})
            return WorkflowDraft.model_validate(row, from_attributes=True)

    async def publish(self, team_id: str, name: str, etag: str, publisher: str) -> WorkflowSnapshot | WorkflowConflict:
        async with self._database.tx() as tx:
            draft: Final = await tx.litellm_workflowdefinition.find_first(
                where={"team_id": team_id, "name": name, "etag": etag}
            )
            if draft is None:
                return WorkflowConflict(message="Draft changed or does not exist", status_code=412)
            if publisher in draft.contributors:
                return WorkflowConflict(message="Draft contributors cannot publish their own draft", status_code=403)
            snapshot_id: Final = str(uuid.uuid4())
            updated: Final = await tx.litellm_workflowdefinition.update_many(
                where={"workflow_id": draft.workflow_id, "etag": etag},
                data={"etag": str(uuid.uuid4()), "published_snapshot_id": snapshot_id},
            )
            if not updated:
                return WorkflowConflict(message="Draft changed or does not exist", status_code=412)
            row: Final = await tx.litellm_workflowsnapshot.create(
                data={
                    "snapshot_id": snapshot_id,
                    "workflow_id": draft.workflow_id,
                    "team_id": team_id,
                    "name": name,
                    "bundle_json": draft.bundle_json,
                    "published_by": publisher,
                }
            )
            return WorkflowSnapshot.model_validate(row, from_attributes=True)

    async def snapshot(self, snapshot_id: str) -> WorkflowSnapshot | None:
        row: Final = await self._database.litellm_workflowsnapshot.find_unique(where={"snapshot_id": snapshot_id})
        return WorkflowSnapshot.model_validate(row, from_attributes=True) if row is not None else None

    async def published(self, team_id: str, name: str) -> WorkflowSnapshot | None:
        draft: Final = await self.get(team_id, name)
        return await self.snapshot(draft.published_snapshot_id) if draft and draft.published_snapshot_id else None

    async def pin_draft(self, workflow_id: str, etag: str, requested_by: str) -> WorkflowSnapshot | WorkflowConflict:
        async with self._database.tx() as tx:
            locked: Final = await tx.litellm_workflowdefinition.update_many(
                where={"workflow_id": workflow_id, "etag": etag}, data={"etag": etag}
            )
            if not locked:
                return WorkflowConflict(message="Draft changed or does not exist", status_code=412)
            draft: Final = await tx.litellm_workflowdefinition.find_unique(where={"workflow_id": workflow_id})
            if draft is None:
                return WorkflowConflict(message="Draft changed or does not exist", status_code=412)
            snapshot_id: Final = str(uuid5(NAMESPACE_URL, f"draft_test/{workflow_id}/{etag}"))
            existing: Final = await tx.litellm_workflowsnapshot.find_unique(where={"snapshot_id": snapshot_id})
            if existing is not None:
                return WorkflowSnapshot.model_validate(existing, from_attributes=True)
            row: Final = await tx.litellm_workflowsnapshot.create(
                data={
                    "snapshot_id": snapshot_id,
                    "workflow_id": workflow_id,
                    "team_id": draft.team_id,
                    "name": draft.name,
                    "bundle_json": draft.bundle_json,
                    "published_by": requested_by,
                    "origin": "draft_test",
                }
            )
            return WorkflowSnapshot.model_validate(row, from_attributes=True)

    async def by_dispatch_key(self, dispatch_key: str) -> WorkflowExecution | None:
        row: Final = await self._database.litellm_workflowexecution.find_unique(where={"dispatch_key": dispatch_key})
        return WorkflowExecution.model_validate(row, from_attributes=True) if row is not None else None

    async def execution(self, run_id: str) -> WorkflowExecution | None:
        row: Final = await self._database.litellm_workflowexecution.find_unique(where={"run_id": run_id})
        return WorkflowExecution.model_validate(row, from_attributes=True) if row is not None else None

    async def dispatch(self, execution: WorkflowExecution) -> WorkflowExecution:
        async with self._database.tx() as tx:
            inserted: Final = await tx.litellm_workflowexecution.create_many(
                data=[
                    {
                        "run_id": execution.run_id,
                        "snapshot_id": execution.snapshot_id,
                        "team_id": execution.team_id,
                        "name": execution.name,
                        "principal": execution.principal,
                        "dispatch_key": execution.dispatch_key,
                        "request_digest": execution.request_digest,
                        "request_json": execution.request_json,
                        "deadline": execution.deadline,
                    }
                ],
                skip_duplicates=True,
            )
            if inserted:
                await tx.litellm_workflowrun.create(
                    data={
                        "run_id": execution.run_id,
                        "session_id": execution.run_id,
                        "workflow_type": "liteflow",
                        "created_by": execution.principal,
                        "input": _json(execution.request_json),
                        "engine_owned": True,
                        "metadata": _json(
                            json.dumps(
                                {
                                    "snapshot_id": execution.snapshot_id,
                                    "team_id": execution.team_id,
                                    "workflow_name": execution.name,
                                }
                            )
                        ),
                    }
                )
            row: Final = await tx.litellm_workflowexecution.find_unique(where={"dispatch_key": execution.dispatch_key})
            return WorkflowExecution.model_validate(row, from_attributes=True)

    async def lease_next(self, owner: str) -> WorkflowExecution | None:
        now: Final = datetime.now(timezone.utc)
        eligible: Final = {
            "state": {"in": ["pending", "running"]},
            "OR": [{"lease_until": None}, {"lease_until": {"lt": now}}],
        }
        async with self._database.tx() as tx:
            candidate: Final = await tx.litellm_workflowexecution.find_first(where=eligible, order={"deadline": "asc"})
            if candidate is None:
                return None
            updated: Final = await tx.litellm_workflowexecution.update_many(
                where={**eligible, "run_id": candidate.run_id},
                data={"lease_owner": owner, "lease_until": now + timedelta(seconds=30), "state": "running"},
            )
            if not updated:
                return None
            row: Final = await tx.litellm_workflowexecution.find_unique(where={"run_id": candidate.run_id})
            return WorkflowExecution.model_validate(row, from_attributes=True)

    async def renew(self, run_id: str, owner: str) -> bool:
        return bool(
            await self._database.litellm_workflowexecution.update_many(
                where=_lease_where(run_id, owner, allow_cancel=True),
                data={"lease_until": datetime.now(timezone.utc) + timedelta(seconds=30)},
            )
        )

    async def steps(self, run_id: str) -> tuple[WorkflowStep, ...]:
        rows: Final = await self._database.litellm_workflowstep.find_many(
            where={"run_id": run_id}, order=[{"started_at": "asc"}, {"step_id": "asc"}]
        )
        return tuple(WorkflowStep.model_validate(row, from_attributes=True) for row in rows)

    async def start_step(
        self, run_id: str, node_id: str, input: JsonValue, deadline: datetime, owner: str
    ) -> WorkflowStep | WorkflowConflict:
        async with self._database.tx() as tx:
            if not await _lock_execution(tx, run_id, owner):
                return WorkflowConflict(message="Execution lease lost")
            existing: Final = await tx.litellm_workflowstep.find_unique(
                where={"run_id_node_id": {"run_id": run_id, "node_id": node_id}}
            )
            if existing is not None:
                return WorkflowStep.model_validate(existing, from_attributes=True)
            row: Final = await tx.litellm_workflowstep.create(
                data={
                    "step_id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "node_id": node_id,
                    "input": _json(json.dumps(input, allow_nan=False)),
                    "deadline": deadline,
                }
            )
            return WorkflowStep.model_validate(row, from_attributes=True)

    async def attempt_step(self, step_id: str, owner: str) -> WorkflowStep | WorkflowConflict:
        async with self._database.tx() as tx:
            previous: Final = await tx.litellm_workflowstep.find_unique(where={"step_id": step_id})
            if previous is None or not await _lock_execution(tx, previous.run_id, owner):
                return WorkflowConflict(message="Execution lease lost")
            row: Final = await tx.litellm_workflowstep.update(
                where={"step_id": step_id}, data={"attempts": {"increment": 1}}
            )
            return WorkflowStep.model_validate(row, from_attributes=True)

    async def finish_step(self, step: WorkflowStep, owner: str) -> bool:
        async with self._database.tx() as tx:
            if not await _lock_execution(tx, step.run_id, owner):
                return False
            return bool(
                await tx.litellm_workflowstep.update_many(
                    where={"step_id": step.step_id, "run_id": step.run_id}, data=_step_data(step)
                )
            )

    async def checkpoint(self, run_id: str, checkpoint: WorkflowCheckpoint, step: WorkflowStep, owner: str) -> bool:
        async with self._database.tx() as tx:
            if step.run_id != run_id or not await _lock_execution(tx, run_id, owner):
                return False
            current: Final = await tx.litellm_workflowexecution.find_unique(where={"run_id": run_id})
            execution: Final = WorkflowExecution.model_validate(current, from_attributes=True)
            if any(item.node_path == checkpoint.node_path for item in execution.checkpoints):
                return True
            updated: Final = await tx.litellm_workflowstep.update_many(
                where={"step_id": step.step_id, "run_id": run_id}, data=_step_data(step)
            )
            if not updated:
                return False
            checkpoints: Final = (*execution.checkpoints, checkpoint)
            await tx.litellm_workflowexecution.update(
                where={"run_id": run_id},
                data={
                    "checkpoints": _json(
                        json.dumps([item.model_dump(mode="json") for item in checkpoints], allow_nan=False)
                    ),
                },
            )
            await tx.litellm_workflowevent.create(
                data={
                    "event_id": str(uuid.uuid4()),
                    "run_id": run_id,
                    "event_type": "node.completed",
                    "step_name": checkpoint.node_path,
                    "sequence_number": len(checkpoints) - 1,
                    "data": _json(checkpoint.model_dump_json()),
                }
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
            if not await _lock_execution(tx, run_id, owner, allow_cancel=True):
                return False
            previous: Final = await tx.litellm_workflowexecution.find_unique(where={"run_id": run_id})
            if previous is None:
                return False
            final_state: Final = WorkflowRunState.CANCELLED if previous.cancel_requested else state
            final_response: Final = None if previous.cancel_requested else response_json
            await tx.litellm_workflowexecution.update(
                where={"run_id": run_id},
                data={
                    "state": final_state.value,
                    "response_json": final_response,
                    "error": error,
                },
            )
            await tx.litellm_workflowrun.update(
                where={"run_id": run_id},
                data={
                    "status": final_state.value,
                    "output": _json(json.dumps({"response": json.loads(final_response) if final_response else None})),
                },
            )
            return True

    async def cancel(self, run_id: str) -> None:
        await self._database.litellm_workflowexecution.update_many(
            where={"run_id": run_id, "state": {"in": ["pending", "running"]}}, data={"cancel_requested": True}
        )
