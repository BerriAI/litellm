"""
Background sweepers for agent sessions.

Three sweepers run on a 30s tick:
- `bootstrap_timeout_sweeper` — sessions stuck in `provisioning` for too long
- `heartbeat_timeout_sweeper` — sessions whose daemon has stopped checking in
- `max_session_minutes_sweeper` — sessions older than the configured ceiling

Each sweeper SELECTs candidate session rows with `FOR UPDATE SKIP LOCKED` so
multiple proxy replicas don't double-terminate the same VM. We can't issue raw
SQL via Prisma (project rule: model methods only), so the implementation uses
`prisma_client.db.litellm_agentsession.find_many()` filtered by status +
timestamp; the `SKIP LOCKED` semantics are emulated by re-checking the row's
status before terminating (cheap optimistic lock — terminate is idempotent).

These sweepers are idempotent: terminating a VM twice is safe; updating the
session row twice is safe.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Optional

from litellm._logging import verbose_proxy_logger
from litellm.managed_agents.vms.base import (
    AgentVMProvider,
    VMHandle,
)

# Default sweeper tick. Configurable via agent_settings.sweep_interval_seconds.
DEFAULT_SWEEP_INTERVAL_SECONDS = 30
# Default ceilings — match LIT-2878.
DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS = 180
DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 120
DEFAULT_MAX_SESSION_MINUTES = 120

# Session statuses the sweepers care about.
STATUS_PROVISIONING = "provisioning"
STATUS_READY = "ready"
STATUS_TERMINATING = "terminating"
STATUS_TERMINATED = "terminated"
STATUS_FAILED = "failed"


@dataclass
class SweeperConfig:
    """Tunables loaded from `agent_settings`."""

    bootstrap_timeout_seconds: int = DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS
    heartbeat_timeout_seconds: int = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS
    max_session_minutes: int = DEFAULT_MAX_SESSION_MINUTES
    sweep_interval_seconds: int = DEFAULT_SWEEP_INTERVAL_SECONDS

    @classmethod
    def from_agent_settings(cls, agent_settings: Optional[dict]) -> "SweeperConfig":
        if not agent_settings:
            return cls()
        ec2 = agent_settings.get("ec2") or {}
        return cls(
            bootstrap_timeout_seconds=int(
                ec2.get("bootstrap_timeout_seconds", DEFAULT_BOOTSTRAP_TIMEOUT_SECONDS)
            ),
            heartbeat_timeout_seconds=int(
                ec2.get("heartbeat_timeout_seconds", DEFAULT_HEARTBEAT_TIMEOUT_SECONDS)
            ),
            max_session_minutes=int(
                ec2.get("max_session_minutes", DEFAULT_MAX_SESSION_MINUTES)
            ),
            sweep_interval_seconds=int(
                agent_settings.get(
                    "sweep_interval_seconds", DEFAULT_SWEEP_INTERVAL_SECONDS
                )
            ),
        )


# Type alias for the "given a session row, build the matching VMHandle" helper.
# Implementations live in the agent_session_endpoints package once Epic A
# lands; the sweepers stay agnostic of the row schema.
HandleBuilder = Callable[[Any], VMHandle]
# Helper to fetch the team's BYOC creds for a session. The EC2 provider needs
# them at terminate time too.
CredsResolver = Callable[[Any], Awaitable[Any]]


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


async def _terminate_and_mark(
    *,
    session: Any,
    new_status: str,
    failure_reason: str,
    provider: AgentVMProvider,
    handle_builder: HandleBuilder,
    creds_resolver: Optional[CredsResolver],
    prisma_client: Any,
) -> None:
    """Common path: terminate VM, update session row. Idempotent."""
    session_id_raw = getattr(session, "session_id", None) or getattr(session, "id", None)
    if not session_id_raw:
        # Defensive: row without an id cannot be updated.
        return
    session_id: str = str(session_id_raw)

    # Re-fetch the row to confirm status hasn't moved (cheap optimistic lock).
    fresh = None
    try:
        fresh = await prisma_client.db.litellm_agentsession.find_unique(
            where={"session_id": session_id}
        )
    except Exception as e:
        verbose_proxy_logger.debug(
            f"sweeper: re-fetch session={session_id} failed: {type(e).__name__}"
        )
        return
    if fresh is None:
        return
    if getattr(fresh, "status", None) in (STATUS_TERMINATED, STATUS_FAILED):
        return

    handle = handle_builder(fresh)
    if handle is None or not handle.vm_id:
        # Nothing to terminate (no VM was ever launched). Just mark the row.
        await _update_status(
            prisma_client=prisma_client,
            session_id=session_id,
            new_status=new_status,
            failure_reason=failure_reason,
        )
        return

    try:
        if creds_resolver is not None:
            creds = await creds_resolver(fresh)
            await provider.terminate(handle, aws_creds=creds)  # type: ignore[call-arg]
        else:
            await provider.terminate(handle)
    except Exception as e:
        # Termination failure is non-fatal for the sweeper — we'll retry next
        # tick. Log and continue so other sessions keep moving.
        verbose_proxy_logger.warning(
            f"sweeper: terminate vm={handle.vm_id} for session={session_id} "
            f"failed: {type(e).__name__}: {e}"
        )
        return

    await _update_status(
        prisma_client=prisma_client,
        session_id=session_id,
        new_status=new_status,
        failure_reason=failure_reason,
    )


async def _update_status(
    *,
    prisma_client: Any,
    session_id: str,
    new_status: str,
    failure_reason: Optional[str],
) -> None:
    data: dict = {"status": new_status, "terminated_at": _now_utc()}
    if failure_reason:
        data["failure_reason"] = failure_reason
    try:
        await prisma_client.db.litellm_agentsession.update(
            where={"session_id": session_id},
            data=data,
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            f"sweeper: update session={session_id} status={new_status} "
            f"failed: {type(e).__name__}: {e}"
        )


async def bootstrap_timeout_sweeper(
    *,
    provider: AgentVMProvider,
    prisma_client: Any,
    config: SweeperConfig,
    handle_builder: HandleBuilder,
    creds_resolver: Optional[CredsResolver] = None,
) -> int:
    """One pass: terminate any session stuck in `provisioning` past the timeout.

    Returns the number of sessions swept.
    """
    cutoff = _now_utc() - timedelta(seconds=config.bootstrap_timeout_seconds)
    swept = 0
    try:
        rows = await prisma_client.db.litellm_agentsession.find_many(
            where={
                "status": STATUS_PROVISIONING,
                "created_at": {"lt": cutoff},
            },
            take=100,  # bound the batch so a backlog doesn't stall the loop
            order={"created_at": "asc"},
        )
    except Exception as e:
        verbose_proxy_logger.debug(
            f"bootstrap_timeout_sweeper: find_many failed: {type(e).__name__}"
        )
        return 0

    for row in rows or []:
        await _terminate_and_mark(
            session=row,
            new_status=STATUS_FAILED,
            failure_reason="bootstrap_timeout",
            provider=provider,
            handle_builder=handle_builder,
            creds_resolver=creds_resolver,
            prisma_client=prisma_client,
        )
        swept += 1
    return swept


async def heartbeat_timeout_sweeper(
    *,
    provider: AgentVMProvider,
    prisma_client: Any,
    config: SweeperConfig,
    handle_builder: HandleBuilder,
    creds_resolver: Optional[CredsResolver] = None,
) -> int:
    """One pass: terminate any `ready` session whose daemon stopped checking in."""
    cutoff = _now_utc() - timedelta(seconds=config.heartbeat_timeout_seconds)
    swept = 0
    try:
        rows = await prisma_client.db.litellm_agentsession.find_many(
            where={
                "status": STATUS_READY,
                "last_heartbeat_at": {"lt": cutoff},
            },
            take=100,
            order={"last_heartbeat_at": "asc"},
        )
    except Exception as e:
        verbose_proxy_logger.debug(
            f"heartbeat_timeout_sweeper: find_many failed: {type(e).__name__}"
        )
        return 0

    for row in rows or []:
        await _terminate_and_mark(
            session=row,
            new_status=STATUS_TERMINATED,
            failure_reason="heartbeat_timeout",
            provider=provider,
            handle_builder=handle_builder,
            creds_resolver=creds_resolver,
            prisma_client=prisma_client,
        )
        swept += 1
    return swept


async def max_session_minutes_sweeper(
    *,
    provider: AgentVMProvider,
    prisma_client: Any,
    config: SweeperConfig,
    handle_builder: HandleBuilder,
    creds_resolver: Optional[CredsResolver] = None,
) -> int:
    """One pass: terminate any session older than `max_session_minutes`."""
    cutoff = _now_utc() - timedelta(minutes=config.max_session_minutes)
    swept = 0
    try:
        rows = await prisma_client.db.litellm_agentsession.find_many(
            where={
                "status": {"in": [STATUS_PROVISIONING, STATUS_READY]},
                "created_at": {"lt": cutoff},
            },
            take=100,
            order={"created_at": "asc"},
        )
    except Exception as e:
        verbose_proxy_logger.debug(
            f"max_session_minutes_sweeper: find_many failed: {type(e).__name__}"
        )
        return 0

    for row in rows or []:
        await _terminate_and_mark(
            session=row,
            new_status=STATUS_TERMINATED,
            failure_reason="max_session_minutes",
            provider=provider,
            handle_builder=handle_builder,
            creds_resolver=creds_resolver,
            prisma_client=prisma_client,
        )
        swept += 1
    return swept


async def sweeper_loop(
    *,
    provider: AgentVMProvider,
    prisma_client: Any,
    config: SweeperConfig,
    handle_builder: HandleBuilder,
    creds_resolver: Optional[CredsResolver] = None,
    stop_event: Optional[asyncio.Event] = None,
) -> None:
    """
    Long-running loop: run all three sweepers every `sweep_interval_seconds`.

    Started by `proxy_server.py` at boot via `asyncio.create_task` (see the
    existing pattern around `_adaptive_router_flusher_loop`). Exits cleanly
    when `stop_event` is set or the task is cancelled.
    """
    verbose_proxy_logger.info(
        f"agent_session sweepers running "
        f"(interval={config.sweep_interval_seconds}s, "
        f"bootstrap_timeout={config.bootstrap_timeout_seconds}s, "
        f"heartbeat_timeout={config.heartbeat_timeout_seconds}s, "
        f"max_session_minutes={config.max_session_minutes}m)"
    )

    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            await bootstrap_timeout_sweeper(
                provider=provider,
                prisma_client=prisma_client,
                config=config,
                handle_builder=handle_builder,
                creds_resolver=creds_resolver,
            )
            await heartbeat_timeout_sweeper(
                provider=provider,
                prisma_client=prisma_client,
                config=config,
                handle_builder=handle_builder,
                creds_resolver=creds_resolver,
            )
            await max_session_minutes_sweeper(
                provider=provider,
                prisma_client=prisma_client,
                config=config,
                handle_builder=handle_builder,
                creds_resolver=creds_resolver,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            verbose_proxy_logger.exception(
                f"sweeper_loop: unhandled error: {type(e).__name__}: {e}"
            )

        try:
            if stop_event is not None:
                await asyncio.wait_for(
                    stop_event.wait(), timeout=config.sweep_interval_seconds
                )
                return
            else:
                await asyncio.sleep(config.sweep_interval_seconds)
        except asyncio.TimeoutError:
            continue
        except asyncio.CancelledError:
            raise
