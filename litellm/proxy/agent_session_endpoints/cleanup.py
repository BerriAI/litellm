"""
Background sweeper that handles three failure modes:

  1. Expired sessions — past ``expires_at``, still non-terminal:
     run the standard terminate flow.
  2. Dead daemons — non-terminal session whose ``last_heartbeat_at`` is
     older than ``DAEMON_HEARTBEAT_DEAD_AFTER_SECONDS``: mark error.
  3. Stuck runs — running runs that haven't been touched within
     ``RUN_IDLE_TIMEOUT_SECONDS``: mark error.

The sweeper is started from ``proxy_server.py``'s startup hooks. It loops
until cancelled, sleeping `CLEANUP_SWEEPER_INTERVAL_SECONDS` between
passes.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional

from litellm._logging import verbose_proxy_logger
from litellm.proxy.agent_session_endpoints.constants import (
    CLEANUP_SWEEPER_INTERVAL_SECONDS,
    DAEMON_HEARTBEAT_DEAD_AFTER_SECONDS,
    EVENT_TYPE_RUN_ERROR,
    RUN_IDLE_TIMEOUT_SECONDS,
    RUN_STATUS_ERROR,
    SESSION_STATUS_ERROR,
    SESSION_STATUS_PROVISIONING,
    SESSION_TERMINAL_STATUSES,
)


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _sweep_expired_sessions(prisma_client) -> int:
    """Terminate sessions whose ``expires_at`` has passed and are not
    already terminal. Returns count terminated."""
    from litellm.proxy.agent_session_endpoints.session_endpoints import (
        _terminate_session_internal,
    )

    rows = await prisma_client.db.litellm_agentsession.find_many(
        where={
            "expires_at": {"lt": _now()},
            "status": {
                "notIn": list(SESSION_TERMINAL_STATUSES),
            },
        },
        take=200,
    )
    for row in rows:
        try:
            await _terminate_session_internal(row.id, reason="expired")
        except Exception as exc:
            verbose_proxy_logger.exception(
                "sweeper: failed to terminate expired session=%s: %s", row.id, exc
            )
    return len(rows)


async def _sweep_dead_daemons(prisma_client) -> int:
    """Mark sessions ``error`` whose daemon hasn't heartbeat in
    ``DAEMON_HEARTBEAT_DEAD_AFTER_SECONDS``.

    Skips ``provisioning`` sessions (they haven't registered yet — their
    own timeout is governed by `expires_at`).

    Routes through ``_terminate_session_internal`` so the VM provider
    is notified — Greptile P1: bypassing this would orphan EC2 instances
    once a real provider replaces ``NoopVMProvider``.
    """
    from litellm.proxy.agent_session_endpoints.session_endpoints import (
        _terminate_session_internal,
    )

    threshold = _now() - timedelta(seconds=DAEMON_HEARTBEAT_DEAD_AFTER_SECONDS)
    rows = await prisma_client.db.litellm_agentsession.find_many(
        where={
            "last_heartbeat_at": {"lt": threshold},
            "status": {
                "notIn": list(SESSION_TERMINAL_STATUSES)
                + [SESSION_STATUS_PROVISIONING],
            },
        },
        take=200,
    )
    if not rows:
        return 0

    # Terminate via the shared helper. It cancels active runs (with
    # `run_cancelled` events), flips the session status, AND fires
    # ``provider.terminate`` — exactly what we want here.
    for row in rows:
        try:
            await _terminate_session_internal(row.id, reason="daemon_dead")
        except Exception as exc:
            verbose_proxy_logger.exception(
                "sweeper: failed to terminate dead-daemon session=%s: %s",
                row.id,
                exc,
            )

    # Daemon-dead sessions land in ``error`` (not ``terminated``); the
    # shared helper sets ``terminated`` so we explicitly downgrade to
    # ``error`` here. (Both are terminal — no further state transitions.)
    now = _now()
    ids = [r.id for r in rows]
    await prisma_client.db.litellm_agentsession.update_many(
        where={"id": {"in": ids}},
        data={
            "status": SESSION_STATUS_ERROR,
            "updated_at": now,
        },
    )

    verbose_proxy_logger.info(
        "sweeper: terminated %d sessions (dead daemon, via provider)", len(ids)
    )
    return len(ids)


async def _sweep_stuck_runs(prisma_client) -> int:
    """Mark runs ``error`` if they've been ``running`` past the idle timeout.

    Sweeper-driven only — clients may legitimately have long-running runs
    so the threshold is generous (``RUN_IDLE_TIMEOUT_SECONDS``, 30 min by
    default).

    After flipping each run to ``error``, drive the parent session
    ``busy`` -> ``ready`` via :func:`refresh_session_status_from_runs`.
    Without this hook, sessions whose only active run was reaped here
    would report ``busy`` indefinitely — Greptile P1.
    """
    from litellm.proxy.agent_session_endpoints.session_status import (
        refresh_session_status_from_runs,
    )

    threshold = _now() - timedelta(seconds=RUN_IDLE_TIMEOUT_SECONDS)
    rows = await prisma_client.db.litellm_agentrun.find_many(
        where={
            "status": "running",
            "updated_at": {"lt": threshold},
        },
        take=200,
    )
    if not rows:
        return 0

    now = _now()
    for run in rows:
        await prisma_client.db.litellm_agentrun.update(
            where={"id": run.id},
            data={
                "status": RUN_STATUS_ERROR,
                "terminated_at": now,
                "updated_at": now,
            },
        )
        last_evt = await prisma_client.db.litellm_agentrunevent.find_first(
            where={"run_id": run.id}, order={"seq": "desc"}
        )
        next_seq = (last_evt.seq + 1) if last_evt else 1
        try:
            await prisma_client.db.litellm_agentrunevent.create(
                data={
                    "run_id": run.id,
                    "seq": next_seq,
                    "event_type": EVENT_TYPE_RUN_ERROR,
                    "payload": {"reason": "run_idle_timeout"},
                }
            )
        except Exception as exc:
            verbose_proxy_logger.warning(
                "sweeper: skipped run_error event run=%s seq=%s: %s",
                run.id,
                next_seq,
                exc,
            )
        # Run just transitioned ``running`` -> ``error`` (terminal). If
        # the parent session has no other active runs, flip
        # ``busy`` -> ``ready`` so SDK consumers see the right status.
        await refresh_session_status_from_runs(prisma_client, run.session_id)
    return len(rows)


async def run_cleanup_pass(prisma_client) -> dict:
    """Single pass — exposed for tests and on-demand triggers."""
    expired = await _sweep_expired_sessions(prisma_client)
    dead = await _sweep_dead_daemons(prisma_client)
    stuck = await _sweep_stuck_runs(prisma_client)
    return {
        "expired_sessions": expired,
        "dead_daemon_sessions": dead,
        "stuck_runs": stuck,
    }


_sweeper_task: Optional[asyncio.Task] = None


async def _cleanup_loop() -> None:
    """Long-running loop. Cancellation-safe — bails out cleanly on
    ``asyncio.CancelledError``."""
    while True:
        try:
            from litellm.proxy.proxy_server import prisma_client

            if prisma_client is not None:
                summary = await run_cleanup_pass(prisma_client)
                if any(summary.values()):
                    verbose_proxy_logger.info(
                        "agent_session_cleanup_sweeper: %s", summary
                    )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            verbose_proxy_logger.exception(
                "agent_session_cleanup_sweeper iteration failed: %s", exc
            )
        try:
            await asyncio.sleep(CLEANUP_SWEEPER_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            raise


def start_cleanup_sweeper() -> None:
    """Idempotent: spawn the sweeper task once. Subsequent calls are no-ops.

    Called from ``proxy_server.startup_event`` so it ships with the proxy.
    """
    global _sweeper_task
    if _sweeper_task is not None and not _sweeper_task.done():
        return
    _sweeper_task = asyncio.create_task(_cleanup_loop())
    verbose_proxy_logger.info("agent_session_cleanup_sweeper started")


def stop_cleanup_sweeper() -> None:
    """Cancel the sweeper task. Safe to call multiple times."""
    global _sweeper_task
    if _sweeper_task is not None and not _sweeper_task.done():
        _sweeper_task.cancel()
    _sweeper_task = None
