"""
Session lifecycle for managed agents.

Two cleanup paths:

1. Pre-delete (handler-driven): before deleting agent or template, stop all live
   Fargate tasks for affected sessions, then mark session rows 'dead'. DB cascade
   removes session rows after agent/template delete completes.

2. Reconciler (background sweep): every 60s, list all tagged Fargate tasks in the
   configured cluster. For each running task, look up its session_id tag in DB.
   Orphan = task running but session row missing OR status in {'dead', 'failed',
   'stopped'} OR creating > 10min. Stop orphan task. Inverse direction handled
   by pre-delete path.

Reconciler runs at proxy startup (catch crashes mid-spawn) and on a 60s interval.
"""

import asyncio
import time
from typing import Any, Dict, Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.managed_agents_endpoints.fargate.tasks import (
    TAG_SESSION_ID,
    describe_tasks_with_tags,
    list_tagged_task_arns,
    stop_task_sync,
)
from litellm.proxy.managed_agents_endpoints.warm_pool import POOL_SLOT_PREFIX

ALIVE_STATUSES = ("creating", "ready")
DEAD_STATUSES = ("dead", "failed", "stopped")
CREATING_TIMEOUT_SECONDS = 600
RECONCILE_INTERVAL_SECONDS = 60


async def stop_session_task(
    *, region: str, cluster: str, task_arn: Optional[str], session_id: str
) -> None:
    """Stop a single session's Fargate task. Idempotent."""
    if not task_arn:
        return
    try:
        await asyncio.to_thread(
            stop_task_sync, region, cluster, task_arn, f"session {session_id} delete"
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            f"stop_session_task failed (session={session_id}, arn={task_arn}): {e}"
        )


async def stop_sessions_for_agent(
    *, prisma_client: Any, region: str, cluster: str, agent_id: str
) -> int:
    """Stop all live Fargate tasks for an agent's sessions. Returns count stopped.

    Caller deletes agent row after this. DB CASCADE removes session rows.
    """
    sessions = await prisma_client.db.litellm_managedagentsessiontable.find_many(
        where={"agent_id": agent_id, "status": {"in": list(ALIVE_STATUSES)}}
    )
    if not sessions:
        return 0
    await asyncio.gather(
        *[
            stop_session_task(
                region=region,
                cluster=cluster,
                task_arn=s.task_arn,
                session_id=s.session_id,
            )
            for s in sessions
        ],
        return_exceptions=True,
    )
    now = time.time()
    await prisma_client.db.litellm_managedagentsessiontable.update_many(
        where={"agent_id": agent_id, "status": {"in": list(ALIVE_STATUSES)}},
        data={"status": "stopped", "stopped_at": _datetime_from_ts(now)},
    )
    return len(sessions)


async def stop_sessions_for_template(
    *, prisma_client: Any, region: str, cluster: str, template_id: str
) -> int:
    """Stop all live Fargate tasks for sessions of all agents under a template."""
    agents = await prisma_client.db.litellm_managedagenttable.find_many(
        where={"template_id": template_id}
    )
    if not agents:
        return 0
    counts = await asyncio.gather(
        *[
            stop_sessions_for_agent(
                prisma_client=prisma_client,
                region=region,
                cluster=cluster,
                agent_id=a.agent_id,
            )
            for a in agents
        ],
        return_exceptions=True,
    )
    return sum(c for c in counts if isinstance(c, int))


async def reconcile_orphans(
    *, prisma_client: Any, region: str, cluster: str
) -> Dict[str, int]:
    """One-shot orphan sweep. Stops Fargate tasks whose session is missing/dead.

    Returns counts: {scanned, orphaned_stopped, stale_creating_stopped}.
    """
    arns = await asyncio.to_thread(list_tagged_task_arns, region, cluster)
    tasks = await asyncio.to_thread(describe_tasks_with_tags, region, cluster, arns)

    # Filter to only managed-agent tasks (have our session tag).
    managed_tasks = [t for t in tasks if TAG_SESSION_ID in t["tags"]]
    if not managed_tasks:
        return {"scanned": 0, "orphaned_stopped": 0, "stale_creating_stopped": 0}

    # Warm-pool tasks are tagged with a sentinel session_id ('pool-warm-...') and
    # have no DB row. They are managed in-process by warm_pool.py, not the DB.
    real_tasks = [
        t
        for t in managed_tasks
        if not t["tags"][TAG_SESSION_ID].startswith(POOL_SLOT_PREFIX)
    ]
    session_ids = [t["tags"][TAG_SESSION_ID] for t in real_tasks]
    rows = (
        await prisma_client.db.litellm_managedagentsessiontable.find_many(
            where={"session_id": {"in": session_ids}}
        )
        if session_ids
        else []
    )
    by_id = {r.session_id: r for r in rows}

    orphaned = 0
    stale = 0
    now_ts = time.time()
    for task in real_tasks:
        sid = task["tags"][TAG_SESSION_ID]
        arn = task["taskArn"]
        row = by_id.get(sid)

        if row is None:
            await _stop_and_log(region, cluster, arn, sid, "missing_db_row")
            orphaned += 1
            continue

        if row.status in DEAD_STATUSES:
            await _stop_and_log(region, cluster, arn, sid, f"db_status={row.status}")
            orphaned += 1
            continue

        if row.status == "creating":
            row_created_ts = row.created_at.timestamp() if row.created_at else now_ts
            if now_ts - row_created_ts > CREATING_TIMEOUT_SECONDS:
                await _stop_and_log(region, cluster, arn, sid, "creating_timeout")
                await prisma_client.db.litellm_managedagentsessiontable.update(
                    where={"session_id": sid},
                    data={
                        "status": "failed",
                        "failure_reason": "spawn timeout — task killed by reconciler",
                        "stopped_at": _datetime_from_ts(now_ts),
                    },
                )
                stale += 1

    return {
        "scanned": len(managed_tasks),
        "orphaned_stopped": orphaned,
        "stale_creating_stopped": stale,
    }


async def reconcile_loop(
    *,
    prisma_client: Any,
    region: str,
    cluster: str,
    interval_seconds: int = RECONCILE_INTERVAL_SECONDS,
) -> None:
    """Long-running reconciler. Run as fire-and-forget asyncio task."""
    while True:
        try:
            stats = await reconcile_orphans(
                prisma_client=prisma_client, region=region, cluster=cluster
            )
            if stats["orphaned_stopped"] or stats["stale_creating_stopped"]:
                verbose_proxy_logger.info(f"managed_agents reconciler: {stats}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            verbose_proxy_logger.exception(f"managed_agents reconciler error: {e}")
        await asyncio.sleep(interval_seconds)


async def _stop_and_log(
    region: str, cluster: str, task_arn: str, session_id: str, reason: str
) -> None:
    verbose_proxy_logger.info(
        f"managed_agents: stopping orphan task arn={task_arn} session={session_id} reason={reason}"
    )
    await asyncio.to_thread(
        stop_task_sync, region, cluster, task_arn, f"orphan: {reason}"
    )


def _datetime_from_ts(ts: float):
    from datetime import datetime, timezone

    return datetime.fromtimestamp(ts, tz=timezone.utc)
