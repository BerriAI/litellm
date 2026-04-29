"""
FastAPI endpoints for scheduled tasks.

Six endpoints under /v1/tasks. Auth via user_api_key_auth — owner_token,
user_id, team_id, agent_id are all stamped from the calling key, never
accepted from the client.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
import litellm.proxy.scheduled_tasks.store as store
from litellm.proxy.scheduled_tasks.schedule import (
    compute_next_run,
    validate_schedule,
)
from litellm.proxy.scheduled_tasks.types import (
    CreateScheduledTaskRequest,
    DueTaskResponse,
    DueTasksResponse,
    ListScheduledTasksResponse,
    ReportTaskResultRequest,
    ScheduledTaskResponse,
    UpdateScheduledTaskRequest,
)

router = APIRouter()


def _require_token(user_api_key_dict: UserAPIKeyAuth) -> str:
    """
    Resolve the FK target for owner_token. We deliberately prefer `.token`
    (the hashed value stored in LiteLLM_VerificationToken) over `.api_key`
    (the raw key as sent by the client), so the FK constraint matches.
    """
    token = user_api_key_dict.token or user_api_key_dict.api_key
    if not token:
        raise HTTPException(status_code=401, detail="missing api key")
    return token


def _row_to_response(row) -> ScheduledTaskResponse:
    """Convert a Prisma row (model object) into the response schema."""
    return ScheduledTaskResponse(
        task_id=row.task_id,
        owner_token=row.owner_token,
        user_id=row.user_id,
        team_id=row.team_id,
        agent_id=row.agent_id,
        title=row.title,
        action=row.action,
        action_args=row.action_args,
        check_prompt=row.check_prompt,
        format_prompt=row.format_prompt,
        metadata=getattr(row, "metadata", None),
        schedule_kind=row.schedule_kind,
        schedule_spec=row.schedule_spec,
        schedule_tz=row.schedule_tz,
        next_run_at=row.next_run_at,
        expires_at=row.expires_at,
        fire_once=row.fire_once,
        status=row.status,
        last_fired_at=row.last_fired_at,
        consecutive_errors=getattr(row, "consecutive_errors", 0) or 0,
        last_error=getattr(row, "last_error", None),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _get_prisma_client():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="prisma client not initialised")
    return prisma_client


@router.post(
    "/v1/tasks",
    tags=["scheduled tasks"],
    response_model=ScheduledTaskResponse,
)
async def create_scheduled_task(
    data: CreateScheduledTaskRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    if data.action == "check" and not data.check_prompt:
        raise HTTPException(
            status_code=400,
            detail="check_prompt is required when action='check'",
        )
    try:
        validate_schedule(
            kind=data.schedule_kind,
            spec=data.schedule_spec,
            tz=data.schedule_tz,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    owner_token = _require_token(user_api_key_dict)
    prisma_client = _get_prisma_client()

    active = await store.count_active_for_owner(prisma_client, owner_token)
    if active >= store.MAX_ACTIVE_TASKS_PER_KEY:
        raise HTTPException(
            status_code=429,
            detail=(
                f"too many active scheduled tasks "
                f"(max {store.MAX_ACTIVE_TASKS_PER_KEY})"
            ),
        )

    now = datetime.now(timezone.utc)
    if data.expires_at <= now:
        raise HTTPException(status_code=400, detail="expires_at must be in the future")

    next_run_at = compute_next_run(
        kind=data.schedule_kind,
        spec=data.schedule_spec,
        tz=data.schedule_tz,
        from_time=now,
    )

    row = await store.create_task(
        prisma_client,
        owner_token=owner_token,
        user_id=user_api_key_dict.user_id,
        team_id=user_api_key_dict.team_id,
        agent_id=user_api_key_dict.agent_id,
        title=data.title,
        action=data.action,
        action_args=data.action_args,
        check_prompt=data.check_prompt,
        format_prompt=data.format_prompt,
        metadata=data.metadata,
        schedule_kind=data.schedule_kind,
        schedule_spec=data.schedule_spec,
        schedule_tz=data.schedule_tz,
        next_run_at=next_run_at,
        expires_at=data.expires_at,
        fire_once=data.fire_once,
    )
    return _row_to_response(row)


@router.get(
    "/v1/tasks",
    tags=["scheduled tasks"],
    response_model=ListScheduledTasksResponse,
)
async def list_scheduled_tasks(
    include_terminal: bool = Query(False),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    owner_token = _require_token(user_api_key_dict)
    prisma_client = _get_prisma_client()
    rows = await store.list_tasks_for_owner(
        prisma_client,
        owner_token=owner_token,
        include_terminal=include_terminal,
    )
    return ListScheduledTasksResponse(tasks=[_row_to_response(r) for r in rows])


@router.get(
    "/v1/tasks/due",
    tags=["scheduled tasks"],
    response_model=DueTasksResponse,
)
async def get_due_tasks(
    actions: Optional[str] = Query(
        None,
        description="Comma-separated list of action names this worker handles.",
    ),
    limit: int = Query(20, ge=1, le=100),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Atomic claim of due tasks. Returns immediately, possibly with empty list.
    Schedule advance happens server-side: recurring rows get a fresh
    next_run_at, fire_once rows flip to 'fired'.
    """
    _require_token(user_api_key_dict)
    prisma_client = _get_prisma_client()

    parsed_actions: Optional[List[str]] = None
    if actions:
        parsed_actions = [a.strip() for a in actions.split(",") if a.strip()]
        if not parsed_actions:
            parsed_actions = None

    rows = await store.claim_due(
        prisma_client,
        agent_id=user_api_key_dict.agent_id,
        actions=parsed_actions,
        limit=limit,
    )
    return DueTasksResponse(
        tasks=[
            DueTaskResponse(
                task_id=r["task_id"],
                title=r["title"],
                action=r["action"],
                action_args=r["action_args"],
                check_prompt=r["check_prompt"],
                format_prompt=r["format_prompt"],
                metadata=r.get("metadata"),
                scheduled_for=r["scheduled_for"],
            )
            for r in rows
        ]
    )


@router.get(
    "/v1/tasks/{task_id}",
    tags=["scheduled tasks"],
    response_model=ScheduledTaskResponse,
)
async def get_scheduled_task(
    task_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    owner_token = _require_token(user_api_key_dict)
    prisma_client = _get_prisma_client()
    row = await store.get_task_for_owner(
        prisma_client,
        task_id=task_id,
        owner_token=owner_token,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    return _row_to_response(row)


@router.patch(
    "/v1/tasks/{task_id}",
    tags=["scheduled tasks"],
    response_model=ScheduledTaskResponse,
)
async def update_scheduled_task(
    task_id: str,
    data: UpdateScheduledTaskRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    owner_token = _require_token(user_api_key_dict)
    prisma_client = _get_prisma_client()

    fields = data.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="no fields to update")

    existing = await store.get_task_for_owner(
        prisma_client,
        task_id=task_id,
        owner_token=owner_token,
    )
    if existing is None:
        raise HTTPException(status_code=404, detail="task not found")
    if existing.status != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"cannot update task in status '{existing.status}'",
        )

    merged_kind = fields.get("schedule_kind", existing.schedule_kind)
    merged_spec = fields.get("schedule_spec", existing.schedule_spec)
    merged_tz = fields.get("schedule_tz", existing.schedule_tz)
    if any(k in fields for k in ("schedule_kind", "schedule_spec", "schedule_tz")):
        try:
            validate_schedule(kind=merged_kind, spec=merged_spec, tz=merged_tz)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    merged_action = fields.get("action", existing.action)
    merged_check = fields.get("check_prompt", existing.check_prompt)
    if merged_action == "check" and not merged_check:
        raise HTTPException(
            status_code=400,
            detail="check_prompt is required when action='check'",
        )

    try:
        row = await store.update_task_for_owner(
            prisma_client,
            task_id=task_id,
            owner_token=owner_token,
            fields=fields,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    return _row_to_response(row)


@router.delete(
    "/v1/tasks/{task_id}",
    tags=["scheduled tasks"],
    response_model=ScheduledTaskResponse,
)
async def cancel_scheduled_task(
    task_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    owner_token = _require_token(user_api_key_dict)
    prisma_client = _get_prisma_client()
    row = await store.cancel_task_for_owner(
        prisma_client,
        task_id=task_id,
        owner_token=owner_token,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    return _row_to_response(row)


@router.post(
    "/v1/tasks/{task_id}/report",
    tags=["scheduled tasks"],
    response_model=ScheduledTaskResponse,
)
async def report_task_result(
    task_id: str,
    data: ReportTaskResultRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Agent reports outcome of one dispatch attempt.

    success → resets the consecutive-error counter.
    error   → increments it. On the Nth consecutive error
              (store.MAX_CONSECUTIVE_ERRORS), status flips to 'failed' and
              /due stops returning the task. Caller's next list() shows it
              as failed with last_error set.
    """
    owner_token = _require_token(user_api_key_dict)
    prisma_client = _get_prisma_client()
    row = await store.report_task_result(
        prisma_client,
        task_id=task_id,
        owner_token=owner_token,
        result=data.result,
        reason=data.reason,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="task not found")
    return _row_to_response(row)
