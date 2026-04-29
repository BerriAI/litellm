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
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


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


class WorkflowRunUpdateRequest(BaseModel):
    status: Optional[str] = None
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


async def _create_with_sequence(
    prisma_client: Any,
    table: str,
    run_id: str,
    row_data: Dict[str, Any],
    *,
    max_retries: int = 5,
) -> Any:
    """Insert a row, retrying on unique-constraint collisions from concurrent sequence reads.

    The SELECT-MAX → INSERT pattern in _get_next_sequence_number is not atomic.
    Under concurrent writes two callers can read the same MAX and race to INSERT
    with the same sequence_number.  The @@unique([run_id, sequence_number])
    constraint on both tables prevents silent corruption; this helper catches that
    error and re-reads the MAX before retrying, capped at max_retries attempts.
    """
    for attempt in range(max_retries):
        seq = await _get_next_sequence_number(prisma_client, run_id, table)
        try:
            if table == "events":
                return await prisma_client.db.litellm_workflowevent.create(
                    data={**row_data, "sequence_number": seq}
                )
            else:
                return await prisma_client.db.litellm_workflowmessage.create(
                    data={**row_data, "sequence_number": seq}
                )
        except Exception as e:
            is_last = attempt == max_retries - 1
            if "unique" in str(e).lower() and not is_last:
                continue
            raise


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
        return {"runs": runs, "total": len(runs)}
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
    """Append an event to the run's event log. Also updates run.status if event_type maps to a status."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        row_data: Dict[str, Any] = {
            "run_id": run_id,
            "event_type": data.event_type,
            "step_name": data.step_name,
        }
        if data.data is not None:
            row_data["data"] = _json(data.data)
        event = await _create_with_sequence(prisma_client, "events", run_id, row_data)

        new_status = _EVENT_STATUS_MAP.get(data.event_type)
        if new_status:
            await prisma_client.db.litellm_workflowrun.update(
                where={"run_id": run_id},
                data={"status": new_status},
            )

        return event
    except Exception as e:
        verbose_proxy_logger.exception("Error appending workflow event: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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
        return {"events": events, "total": len(events)}
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
    """Append a conversation message. Stores full content (not truncated)."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )

    try:
        row_data: Dict[str, Any] = {
            "run_id": run_id,
            "role": data.role,
            "content": data.content,
        }
        if data.session_id is not None:
            row_data["session_id"] = data.session_id
        msg = await _create_with_sequence(prisma_client, "messages", run_id, row_data)
        return msg
    except Exception as e:
        verbose_proxy_logger.exception("Error appending workflow message: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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
        return {"messages": messages, "total": len(messages)}
    except Exception as e:
        verbose_proxy_logger.exception("Error listing workflow messages: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
