"""
Storage layer for LiteLLM_ScheduledTaskTable.

CRUD via Prisma model methods. Single approved raw-SQL exception in
claim_due() — Prisma cannot express SELECT FOR UPDATE SKIP LOCKED, which
is required for multi-pod tick safety.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from litellm.proxy.scheduled_tasks.schedule import compute_next_run


def _encode_json(value: Any) -> str:
    """
    Encode a value bound for a Prisma `Json?` column. prisma-client-python
    rejects raw Python dicts/lists on Json columns; json.dumps once so the
    driver hands a string to Postgres jsonb. Read path round-trips back to
    native Python. Callers must skip the field entirely when value is None.
    """
    return json.dumps(value)


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
        "metadata",
    }
)

# Json? columns that go through Prisma — must be JSON-encoded if present,
# omitted entirely if None.
JSON_FIELDS = frozenset({"action_args", "metadata"})

MAX_ACTIVE_TASKS_PER_KEY = 10
MAX_CONSECUTIVE_ERRORS = 3
TERMINAL_STATUSES = ("fired", "expired", "cancelled", "failed")


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
    metadata: Optional[Any],
    schedule_kind: str,
    schedule_spec: str,
    schedule_tz: Optional[str],
    next_run_at: datetime,
    expires_at: datetime,
    fire_once: bool,
) -> Any:
    data: Dict[str, Any] = {
        "owner_token": owner_token,
        "user_id": user_id,
        "team_id": team_id,
        "agent_id": agent_id,
        "title": title,
        "action": action,
        "check_prompt": check_prompt,
        "format_prompt": format_prompt,
        "schedule_kind": schedule_kind,
        "schedule_spec": schedule_spec,
        "schedule_tz": schedule_tz,
        "next_run_at": next_run_at,
        "expires_at": expires_at,
        "fire_once": fire_once,
    }
    if action_args is not None:
        data["action_args"] = _encode_json(action_args)
    if metadata is not None:
        data["metadata"] = _encode_json(metadata)
    return await prisma_client.db.litellm_scheduledtasktable.create(data=data)


async def sweep_expired_for_owner(prisma_client: Any, owner_token: str) -> int:
    """
    Lazy expiry: flip pending rows past expires_at to 'expired' before any
    read returns them. Without this, a task that never fires after its
    expires_at sits in 'pending' until the next claim that happens to scan
    it — which may be never if no other rows are ever due.

    Returns count of rows flipped.
    """
    return await prisma_client.db.litellm_scheduledtasktable.update_many(
        where={
            "owner_token": owner_token,
            "status": "pending",
            "expires_at": {"lte": datetime.now(timezone.utc)},
        },
        data={"status": "expired"},
    )


async def list_tasks_for_owner(
    prisma_client: Any,
    *,
    owner_token: str,
    include_terminal: bool,
) -> List[Any]:
    await sweep_expired_for_owner(prisma_client, owner_token)
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
    await sweep_expired_for_owner(prisma_client, owner_token)
    return await prisma_client.db.litellm_scheduledtasktable.find_first(
        where={"task_id": task_id, "owner_token": owner_token},
    )


async def report_task_result(
    prisma_client: Any,
    *,
    task_id: str,
    owner_token: str,
    result: str,
    reason: Optional[str],
) -> Optional[Any]:
    """
    Agent reports outcome of one dispatch attempt.

    success → reset consecutive_errors, clear last_error.
    error   → bump consecutive_errors. If >= MAX_CONSECUTIVE_ERRORS, flip
              status to 'failed' so /due stops re-emitting it.

    Scoped by (task_id, owner_token). Returns updated row, or None if not
    found / not owned.
    """
    existing = await prisma_client.db.litellm_scheduledtasktable.find_first(
        where={"task_id": task_id, "owner_token": owner_token},
    )
    if existing is None:
        return None

    if result == "success":
        affected = await prisma_client.db.litellm_scheduledtasktable.update_many(
            where={"task_id": task_id, "owner_token": owner_token},
            data={"consecutive_errors": 0, "last_error": None},
        )
        if affected == 0:
            return None
        return await prisma_client.db.litellm_scheduledtasktable.find_first(
            where={"task_id": task_id, "owner_token": owner_token},
        )

    new_count = (existing.consecutive_errors or 0) + 1
    update_data: Dict[str, Any] = {
        "consecutive_errors": new_count,
        "last_error": reason,
    }
    where_clause: Dict[str, Any] = {
        "task_id": task_id,
        "owner_token": owner_token,
    }
    if new_count >= MAX_CONSECUTIVE_ERRORS:
        update_data["status"] = "failed"
        where_clause["status"] = "pending"
    affected = await prisma_client.db.litellm_scheduledtasktable.update_many(
        where=where_clause,
        data=update_data,
    )
    if affected == 0:
        return None
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

    # Mirror the create path: Json columns must be json-encoded for
    # prisma-client-python. Drop the key entirely when caller sent null;
    # passing None to a Json? column is rejected.
    cleaned: Dict[str, Any] = {}
    for k, v in fields.items():
        if k in JSON_FIELDS:
            if v is None:
                continue  # skip — Prisma rejects None on Json?
            cleaned[k] = _encode_json(v)
        else:
            cleaned[k] = v
    if not cleaned:
        raise ValueError("no fields to update")
    fields = cleaned

    affected = await prisma_client.db.litellm_scheduledtasktable.update_many(
        where={
            "task_id": task_id,
            "owner_token": owner_token,
            "status": "pending",
        },
        data=fields,
    )
    if affected == 0:
        return None
    return await prisma_client.db.litellm_scheduledtasktable.find_first(
        where={"task_id": task_id, "owner_token": owner_token},
    )


async def cancel_task_for_owner(
    prisma_client: Any,
    *,
    task_id: str,
    owner_token: str,
) -> Optional[Any]:
    affected = await prisma_client.db.litellm_scheduledtasktable.update_many(
        where={
            "task_id": task_id,
            "owner_token": owner_token,
            "status": "pending",
        },
        data={"status": "cancelled"},
    )
    if affected == 0:
        return None
    return await prisma_client.db.litellm_scheduledtasktable.find_first(
        where={"task_id": task_id, "owner_token": owner_token},
    )


async def claim_due(
    prisma_client: Any,
    *,
    owner_token: str,
    agent_id: Optional[str],
    actions: Optional[List[str]],
    limit: int,
) -> List[Dict[str, Any]]:
    """
    Atomically claim due tasks and advance their schedule.

    APPROVED RAW-SQL EXCEPTION (CLAUDE.md): SELECT FOR UPDATE SKIP LOCKED
    is required for multi-pod safety and is not expressible via Prisma
    model methods. Per-row writes use Prisma inside the same transaction.

    SECURITY: owner_token is required. Without it, any authenticated key
    whose agent_id is NULL would have matched every task in the table via
    the `($n::text IS NULL OR agent_id = $n)` pattern, leaking
    check_prompt / action_args across tenants. Always scope claims to the
    rows the calling key owns.
    """
    if not owner_token:
        raise ValueError("owner_token is required for claim_due")
    now = datetime.now(timezone.utc)

    async with prisma_client.db.tx(timeout=timedelta(seconds=30)) as tx:
        rows = await tx.query_raw(
            """
            SELECT task_id, schedule_kind, schedule_spec, schedule_tz,
                   fire_once, expires_at, next_run_at,
                   action, action_args, check_prompt, format_prompt,
                   metadata, title
              FROM "LiteLLM_ScheduledTaskTable"
             WHERE status = 'pending'
               AND next_run_at <= now()
               AND owner_token = $1
               AND (($2::text IS NULL AND agent_id IS NULL) OR agent_id = $2)
               AND ($3::text[] IS NULL OR action = ANY($3))
             ORDER BY next_run_at
             LIMIT $4
             FOR UPDATE SKIP LOCKED
            """,
            owner_token,
            agent_id,
            actions,
            limit,
        )

        if not rows:
            return []

        expired_task_ids: set = set()
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
                    expired_task_ids.add(r["task_id"])
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
            "metadata": r["metadata"],
            "scheduled_for": r["next_run_at"],
        }
        for r in rows
        if r["task_id"] not in expired_task_ids
    ]
