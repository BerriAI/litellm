"""
WORKFLOW RUN MANAGEMENT

Generic durable state tracking for agents and automated workflows.

POST   /v1/workflows/runs                       - Create a workflow run
GET    /v1/workflows/runs                       - List runs (filter by type, status)
GET    /v1/workflows/runs/{run_id}              - Get run with latest event
PATCH  /v1/workflows/runs/{run_id}              - Update status, metadata, output
POST   /v1/workflows/runs/{run_id}/events       - Append event (updates run status)
GET    /v1/workflows/runs/{run_id}/events       - Full event log
POST   /v1/workflows/runs/{run_id}/messages     - Append conversation message
GET    /v1/workflows/runs/{run_id}/messages     - Fetch conversation history
"""

import json
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from prisma.errors import UniqueViolationError
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

_MAX_SEQUENCE_RETRIES = 5


def _json(value: Any) -> str:
    """Serialize a Python value for prisma-client-py Json fields (must be a string)."""
    return json.dumps(value)


# Status transitions driven by event_type
_EVENT_STATUS_MAP: Dict[str, str] = {
    "step.started": "running",
    "step.failed": "failed",
    "hook.waiting": "paused",
    "hook.received": "running",
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class WorkflowRunCreateRequest(BaseModel):
    workflow_type: str
    input: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


WorkflowRunStatus = Literal["pending", "running", "paused", "completed", "failed"]


class WorkflowRunUpdateRequest(BaseModel):
    status: Optional[WorkflowRunStatus] = None
    output: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class WorkflowEventCreateRequest(BaseModel):
    event_type: str
    step_name: str
    data: Optional[Dict[str, Any]] = None


class WorkflowMessageCreateRequest(BaseModel):
    role: str
    content: str
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _get_next_sequence_number(prisma_client: Any, run_id: str, table: str) -> int:
    """Return MAX(sequence_number) + 1 for the given run, for either events or messages."""
    if table == "events":
        rows = await prisma_client.db.litellm_workflowevent.find_many(
            where={"run_id": run_id},
            order={"sequence_number": "desc"},
            take=1,
        )
    else:
        rows = await prisma_client.db.litellm_workflowmessage.find_many(
            where={"run_id": run_id},
            order={"sequence_number": "desc"},
            take=1,
        )
    return (rows[0].sequence_number + 1) if rows else 0


async def _require_run(prisma_client: Any, run_id: str) -> Any:
    """Return the run or raise 404 — prevents FK-violation 500s on sub-resource endpoints."""
    run = await prisma_client.db.litellm_workflowrun.find_unique(
        where={"run_id": run_id}
    )
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/v1/workflows/runs",
    tags=["workflow management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_workflow_run(
    data: WorkflowRunCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Create a new workflow run. Returns run_id and session_id."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        create_data: Dict[str, Any] = {"workflow_type": data.workflow_type}
        if data.input is not None:
            create_data["input"] = _json(data.input)
        if data.metadata is not None:
            create_data["metadata"] = _json(data.metadata)
        run = await prisma_client.db.litellm_workflowrun.create(data=create_data)
        return run
    except Exception as e:
        verbose_proxy_logger.exception("Error creating workflow run: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/workflows/runs",
    tags=["workflow management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_workflow_runs(
    workflow_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=250),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List workflow runs. Filter by workflow_type and/or status."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    where: Dict[str, Any] = {}
    if workflow_type:
        where["workflow_type"] = workflow_type
    if status:
        # Support comma-separated status values: ?status=running,paused
        statuses = [s.strip() for s in status.split(",")]
        where["status"] = {"in": statuses} if len(statuses) > 1 else statuses[0]

    try:
        runs = await prisma_client.db.litellm_workflowrun.find_many(
            where=where,
            order={"created_at": "desc"},
            take=limit,
        )
        return {"runs": runs, "count": len(runs)}
    except Exception as e:
        verbose_proxy_logger.exception("Error listing workflow runs: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/v1/workflows/runs/{run_id}",
    tags=["workflow management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_workflow_run(
    run_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Get a workflow run with its most recent event."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        run = await prisma_client.db.litellm_workflowrun.find_unique(
            where={"run_id": run_id},
            include={"events": {"order_by": {"sequence_number": "desc"}, "take": 1}},
        )
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return run
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error getting workflow run: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.patch(
    "/v1/workflows/runs/{run_id}",
    tags=["workflow management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def update_workflow_run(
    run_id: str,
    data: WorkflowRunUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Update status, metadata, or output on a workflow run."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    update: Dict[str, Any] = {}
    if data.status is not None:
        update["status"] = data.status
    if data.output is not None:
        update["output"] = _json(data.output)
    if data.metadata is not None:
        update["metadata"] = _json(data.metadata)

    if not update:
        raise HTTPException(status_code=400, detail="No fields to update")

    try:
        run = await prisma_client.db.litellm_workflowrun.update(
            where={"run_id": run_id},
            data=update,
        )
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        return run
    except HTTPException:
        raise
    except Exception as e:
        verbose_proxy_logger.exception("Error updating workflow run: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/v1/workflows/runs/{run_id}/events",
    tags=["workflow management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def append_workflow_event(
    run_id: str,
    data: WorkflowEventCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Append an event to the run's event log. Also updates run.status if event_type maps to a status.

    Sequence numbers are assigned with optimistic concurrency: on a unique-constraint
    collision (concurrent append to same run), we retry up to _MAX_SEQUENCE_RETRIES times
    with a freshly computed MAX+1. The event+status update runs inside a single DB
    transaction so the materialized status never desynchronizes from the event log.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    # Validate run exists before attempting inserts — avoids FK-violation 500.
    await _require_run(prisma_client, run_id)

    new_status = _EVENT_STATUS_MAP.get(data.event_type)

    for attempt in range(_MAX_SEQUENCE_RETRIES):
        try:
            seq = await _get_next_sequence_number(prisma_client, run_id, "events")
            event_data: Dict[str, Any] = {
                "run_id": run_id,
                "event_type": data.event_type,
                "step_name": data.step_name,
                "sequence_number": seq,
            }
            if data.data is not None:
                event_data["data"] = _json(data.data)

            # Atomically insert event and update run status in one transaction.
            async with prisma_client.db.tx() as tx:
                event = await tx.litellm_workflowevent.create(data=event_data)
                if new_status:
                    await tx.litellm_workflowrun.update(
                        where={"run_id": run_id},
                        data={"status": new_status},
                    )

            return event

        except UniqueViolationError:
            if attempt == _MAX_SEQUENCE_RETRIES - 1:
                verbose_proxy_logger.exception(
                    "Sequence number collision after %d retries for run %s",
                    _MAX_SEQUENCE_RETRIES,
                    run_id,
                )
                raise HTTPException(
                    status_code=409,
                    detail="Concurrent write conflict — please retry",
                )
            # Re-read MAX+1 on next iteration.
            continue

        except Exception as e:
            verbose_proxy_logger.exception("Error appending workflow event: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    # Unreachable but satisfies type checkers.
    raise HTTPException(
        status_code=500, detail="Failed to append event"
    )  # pragma: no cover


@router.get(
    "/v1/workflows/runs/{run_id}/events",
    tags=["workflow management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_workflow_events(
    run_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Fetch full event log for a run, ordered by sequence_number."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        events = await prisma_client.db.litellm_workflowevent.find_many(
            where={"run_id": run_id},
            order={"sequence_number": "asc"},
        )
        return {"events": events, "count": len(events)}
    except Exception as e:
        verbose_proxy_logger.exception("Error listing workflow events: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/v1/workflows/runs/{run_id}/messages",
    tags=["workflow management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def append_workflow_message(
    run_id: str,
    data: WorkflowMessageCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Append a conversation message. Stores full content (not truncated).

    Uses optimistic concurrency for sequence numbers — retries on unique-constraint
    collision so concurrent appends to the same run don't produce 500 errors.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    # Validate run exists before attempting insert — avoids FK-violation 500.
    await _require_run(prisma_client, run_id)

    for attempt in range(_MAX_SEQUENCE_RETRIES):
        try:
            seq = await _get_next_sequence_number(prisma_client, run_id, "messages")
            msg_data: Dict[str, Any] = {
                "run_id": run_id,
                "role": data.role,
                "content": data.content,
                "sequence_number": seq,
            }
            if data.session_id is not None:
                msg_data["session_id"] = data.session_id
            msg = await prisma_client.db.litellm_workflowmessage.create(data=msg_data)
            return msg

        except UniqueViolationError:
            if attempt == _MAX_SEQUENCE_RETRIES - 1:
                verbose_proxy_logger.exception(
                    "Sequence number collision after %d retries for run %s",
                    _MAX_SEQUENCE_RETRIES,
                    run_id,
                )
                raise HTTPException(
                    status_code=409,
                    detail="Concurrent write conflict — please retry",
                )
            continue

        except Exception as e:
            verbose_proxy_logger.exception("Error appending workflow message: %s", e)
            raise HTTPException(status_code=500, detail=str(e))

    raise HTTPException(
        status_code=500, detail="Failed to append message"
    )  # pragma: no cover


@router.get(
    "/v1/workflows/runs/{run_id}/messages",
    tags=["workflow management"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_workflow_messages(
    run_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Fetch full conversation history for a run, ordered by sequence_number."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        messages = await prisma_client.db.litellm_workflowmessage.find_many(
            where={"run_id": run_id},
            order={"sequence_number": "asc"},
        )
        return {"messages": messages, "count": len(messages)}
    except Exception as e:
        verbose_proxy_logger.exception("Error listing workflow messages: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
