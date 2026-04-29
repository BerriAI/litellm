"""
Storage layer for LiteLLM_ScheduledTaskTable.

CRUD via Prisma model methods. Single approved raw-SQL exception in
claim_due() — Prisma cannot express SELECT FOR UPDATE SKIP LOCKED, which
is required for multi-pod tick safety.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from litellm.proxy.scheduled_tasks.schedule import compute_next_run

# Whitelist of fields PATCH can touch. Anything outside this set would let
# a buggy caller rewrite scheduling state (status, fired flags, ...).
UPDATABLE_FIELDS = frozenset(
    {
        "title",
        "check_prompt",
        "schedule_kind",
        "schedule_spec",
        "schedule_tz",
        "next_run_at",
        "expires_at",
        "fire_once",
        "action",
        "action_args",
        "format_prompt",
    }
)

MAX_ACTIVE_TASKS_PER_KEY = 10
TERMINAL_STATUSES = ("fired", "expired", "cancelled")


async def count_active_for_owner(prisma_client: Any, owner_token: str) -> int:
    return await prisma_client.db.litellm_scheduledtasktable.count(
        where={"owner_token": owner_token, "status": "pending"},
    )


async def create_task(
    prisma_client: Any,
    *,
    owner_token: str,
    user_id: Optional[str],
    team_id: Optional[str],
    agent_id: Optional[str],
    title: str,
    action: str,
    action_args: Optional[Dict[str, Any]],
    check_prompt: Optional[str],
    format_prompt: Optional[str],
    schedule_kind: str,
    schedule_spec: str,
    schedule_tz: Optional[str],
    next_run_at: datetime,
    expires_at: datetime,
    fire_once: bool,
) -> Any:
    return await prisma_client.db.litellm_scheduledtasktable.create(
        data={
            "owner_token": owner_token,
            "user_id": user_id,
            "team_id": team_id,
            "agent_id": agent_id,
            "title": title,
            "action": action,
            "action_args": action_args,
            "check_prompt": check_prompt,
            "format_prompt": format_prompt,
            "schedule_kind": schedule_kind,
            "schedule_spec": schedule_spec,
            "schedule_tz": schedule_tz,
            "next_run_at": next_run_at,
            "expires_at": expires_at,
            "fire_once": fire_once,
        }
    )


async def list_tasks_for_owner(
    prisma_client: Any,
    *,
    owner_token: str,
    include_terminal: bool,
) -> List[Any]:
    where: Dict[str, Any] = {"owner_token": owner_token}
    if not include_terminal:
        where["status"] = "pending"
    return await prisma_client.db.litellm_scheduledtasktable.find_many(
        where=where,
        order={"created_at": "desc"},
    )


async def get_task_for_owner(
    prisma_client: Any,
    *,
    task_id: str,
    owner_token: str,
) -> Optional[Any]:
    return await prisma_client.db.litellm_scheduledtasktable.find_first(
        where={"task_id": task_id, "owner_token": owner_token},
    )


async def update_task_for_owner(
    prisma_client: Any,
    *,
    task_id: str,
    owner_token: str,
    fields: Dict[str, Any],
) -> Optional[Any]:
    """
    Update one or more whitelisted fields. Only when status='pending'.
    Returns updated row, or None if not found / not owned / not pending.
    """
    bad = [k for k in fields if k not in UPDATABLE_FIELDS]
    if bad:
        raise ValueError(f"cannot update fields: {', '.join(sorted(bad))}")
    if not fields:
        raise ValueError("no fields to update")

    existing = await prisma_client.db.litellm_scheduledtasktable.find_first(
        where={
            "task_id": task_id,
            "owner_token": owner_token,
            "status": "pending",
        },
    )
    if existing is None:
        return None
    return await prisma_client.db.litellm_scheduledtasktable.update(
        where={"task_id": task_id},
        data=fields,
    )


async def cancel_task_for_owner(
    prisma_client: Any,
    *,
    task_id: str,
    owner_token: str,
) -> Optional[Any]:
    existing = await prisma_client.db.litellm_scheduledtasktable.find_first(
        where={
            "task_id": task_id,
            "owner_token": owner_token,
            "status": "pending",
        },
    )
    if existing is None:
        return None
    return await prisma_client.db.litellm_scheduledtasktable.update(
        where={"task_id": task_id},
        data={"status": "cancelled"},
    )


async def claim_due(
    prisma_client: Any,
    *,
    agent_id: Optional[str],
    actions: Optional[List[str]],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Atomically claim due tasks and advance their schedule.

    APPROVED RAW-SQL EXCEPTION (CLAUDE.md): SELECT FOR UPDATE SKIP LOCKED
    is required for multi-pod safety and is not expressible via Prisma
    model methods. Per-row writes use Prisma inside the same transaction.
    """
    now = datetime.now(timezone.utc)

    async with prisma_client.db.tx(timeout=timedelta(seconds=30)) as tx:
        rows = await tx.query_raw(
            """
            SELECT task_id, schedule_kind, schedule_spec, schedule_tz,
                   fire_once, expires_at, next_run_at,
                   action, action_args, check_prompt, format_prompt, title
              FROM "LiteLLM_ScheduledTaskTable"
             WHERE status = 'pending'
               AND next_run_at <= now()
               AND ($1::text IS NULL OR agent_id = $1)
               AND ($2::text[] IS NULL OR action = ANY($2))
             ORDER BY next_run_at
             LIMIT $3
             FOR UPDATE SKIP LOCKED
            """,
            agent_id,
            actions,
            limit,
        )

        if not rows:
            return []

        async with tx.batch_() as batcher:
            for r in rows:
                expires_at = r["expires_at"]
                if isinstance(expires_at, str):
                    expires_at = datetime.fromisoformat(
                        expires_at.replace("Z", "+00:00")
                    )
                if expires_at.tzinfo is None:
                    expires_at = expires_at.replace(tzinfo=timezone.utc)

                if expires_at <= now:
                    new_status = "expired"
                    new_next = r["next_run_at"]
                elif r["fire_once"]:
                    new_status = "fired"
                    new_next = r["next_run_at"]
                else:
                    new_status = "pending"
                    new_next = compute_next_run(
                        kind=r["schedule_kind"],
                        spec=r["schedule_spec"],
                        tz=r["schedule_tz"],
                        from_time=now,
                    )
                batcher.litellm_scheduledtasktable.update(
                    where={"task_id": r["task_id"]},
                    data={
                        "status": new_status,
                        "next_run_at": new_next,
                        "last_fired_at": now,
                    },
                )

    return [
        {
            "task_id": r["task_id"],
            "title": r["title"],
            "action": r["action"],
            "action_args": r["action_args"],
            "check_prompt": r["check_prompt"],
            "format_prompt": r["format_prompt"],
            "scheduled_for": r["next_run_at"],
        }
        for r in rows
    ]
