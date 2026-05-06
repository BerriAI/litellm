"""
Daemon-side internal endpoints under ``/v2/sessions/{sid}/internal/...``.

These four routes are how the on-VM daemon talks back to the proxy. They
authenticate with the session-scoped JWT minted on session create, NOT a
regular user virtual key:

  * POST  /v2/sessions/{sid}/internal/register
        Daemon "I'm alive" — flips session provisioning -> ready.
  * POST  /v2/sessions/{sid}/internal/heartbeat
        Periodic liveness ping; bumps last_heartbeat_at.
  * GET   /v2/sessions/{sid}/runs/next/internal/poll
        Long-poll for the next queued run; sets it to running.
  * POST  /v2/sessions/{sid}/runs/{rid}/events:append
        Append an event; flips run status if terminal.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import ORJSONResponse, Response

from litellm._logging import verbose_proxy_logger
from litellm.proxy.agent_session_endpoints.auth import daemon_token_auth
from litellm.proxy.agent_session_endpoints.constants import (
    NEXT_RUN_LONG_POLL_TIMEOUT_SECONDS,
    NEXT_RUN_POLL_INTERVAL_SECONDS,
    RUN_STATUS_QUEUED,
    RUN_STATUS_RUNNING,
    RUN_TERMINAL_EVENT_TYPES,
    SESSION_STATUS_PROVISIONING,
    SESSION_STATUS_READY,
)
from litellm.proxy.agent_session_endpoints.session_status import (
    refresh_session_status_from_runs,
)
from litellm.proxy.agent_session_endpoints.schemas import (
    DaemonHeartbeatRequest,
    DaemonRegisterRequest,
    EventAppend,
    NextRunResponse,
)

router = APIRouter()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_seq_collision(exc: BaseException) -> bool:
    """True iff ``exc`` is a (run_id, seq) unique-constraint violation.

    We can't simply ``except prisma.errors.UniqueViolationError`` because
    the unit-test stand-in raises a plain ``RuntimeError("event_seq_collision")``,
    and we don't want the test path and prod path to diverge. Match
    either the Prisma class (when available) or our marker string.
    """
    try:
        from prisma.errors import UniqueViolationError

        if isinstance(exc, UniqueViolationError):
            return True
    except Exception:
        # Prisma unavailable in some test environments; fall through to
        # string-based detection.
        pass
    return "event_seq_collision" in str(exc)


async def _get_prisma_client_or_503():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return prisma_client


@router.post(
    "/v2/sessions/{session_id}/internal/register",
    response_class=ORJSONResponse,
    tags=["agent-internal"],
)
async def daemon_register(
    body: DaemonRegisterRequest,
    session_id: str = Path(...),
    daemon: Dict[str, Any] = Depends(daemon_token_auth),
):
    """Daemon announces itself: flip session provisioning -> ready.

    Idempotent: re-registering an already-ready session is a no-op (200).
    Vending the registration is what moves the session out of provisioning,
    so the cleanup sweeper's "stuck provisioning" rule doesn't fire after.
    """
    prisma_client = await _get_prisma_client_or_503()
    now = _now()
    session_row = daemon["_session_row"]

    update_data: Dict[str, Any] = {
        "last_heartbeat_at": now,
        "updated_at": now,
    }
    if body.vm_id and not session_row.vm_id:
        update_data["vm_id"] = body.vm_id
    if session_row.status == SESSION_STATUS_PROVISIONING:
        update_data["status"] = SESSION_STATUS_READY

    updated = await prisma_client.db.litellm_agentsession.update(
        where={"id": session_id},
        data=update_data,
    )
    verbose_proxy_logger.info(
        "session.register session_id=%s status=%s vm_id=%s",
        session_id,
        updated.status,
        updated.vm_id,
    )
    return {
        "session_id": session_id,
        "status": updated.status,
        "registered_at": now.isoformat(),
    }


@router.post(
    "/v2/sessions/{session_id}/internal/heartbeat",
    response_class=ORJSONResponse,
    tags=["agent-internal"],
)
async def daemon_heartbeat(
    body: DaemonHeartbeatRequest,
    session_id: str = Path(...),
    daemon: Dict[str, Any] = Depends(daemon_token_auth),
):
    """Bump ``last_heartbeat_at``. The cleanup sweeper uses this to detect
    dead daemons (no heartbeat for ``DAEMON_HEARTBEAT_DEAD_AFTER_SECONDS``)."""
    prisma_client = await _get_prisma_client_or_503()
    now = _now()
    await prisma_client.db.litellm_agentsession.update(
        where={"id": session_id},
        data={"last_heartbeat_at": now, "updated_at": now},
    )
    return {"session_id": session_id, "last_heartbeat_at": now.isoformat()}


async def _claim_next_queued_run(prisma_client, session_id: str) -> Optional[Any]:
    """Atomically claim the oldest queued run for ``session_id``.

    Best-effort optimistic claim: read oldest queued run, then UPDATE with
    ``where status = 'queued'`` so two concurrent daemons can't both move
    the same row to ``running`` (Prisma's ``update_many`` returns a count;
    we re-read to confirm we won).

    A future version could lift this to a single ``RETURNING`` statement
    via raw SQL or a Prisma ``transaction`` — for the noop provider this
    is sufficient because the daemon side runs single-tenant anyway.
    """
    candidate = await prisma_client.db.litellm_agentrun.find_first(
        where={"session_id": session_id, "status": RUN_STATUS_QUEUED},
        order={"created_at": "asc"},
    )
    if candidate is None:
        return None

    now = _now()
    result = await prisma_client.db.litellm_agentrun.update_many(
        where={"id": candidate.id, "status": RUN_STATUS_QUEUED},
        data={
            "status": RUN_STATUS_RUNNING,
            "started_at": now,
            "updated_at": now,
        },
    )
    # Prisma returns either a count (int) or an object with `.count`.
    count = getattr(result, "count", result)
    if not count:
        return None  # someone else won — caller can retry
    return await prisma_client.db.litellm_agentrun.find_unique(
        where={"id": candidate.id}
    )


@router.get(
    "/v2/sessions/{session_id}/runs/next/internal/poll",
    response_class=ORJSONResponse,
    response_model=NextRunResponse,
    tags=["agent-internal"],
)
async def daemon_next_run(
    session_id: str = Path(...),
    daemon: Dict[str, Any] = Depends(daemon_token_auth),
):
    """Long-poll: return the next queued run, claiming it as ``running``.

    Loops up to ``NEXT_RUN_LONG_POLL_TIMEOUT_SECONDS`` polling every
    ``NEXT_RUN_POLL_INTERVAL_SECONDS``. Returns 204 if nothing showed up
    (daemon will re-poll). The claim step uses an optimistic
    ``update_many WHERE status='queued'`` so two daemons can't both grab
    the same row.
    """
    prisma_client = await _get_prisma_client_or_503()
    # ``get_running_loop`` (not ``get_event_loop``) is the correct API
    # inside a running coroutine — see Python 3.10 deprecation.
    deadline = asyncio.get_running_loop().time() + NEXT_RUN_LONG_POLL_TIMEOUT_SECONDS

    while True:
        claimed = await _claim_next_queued_run(prisma_client, session_id)
        if claimed is not None:
            # Run just transitioned queued -> running; if the session is
            # still ``ready``, flip it to ``busy``. Idempotent — if the
            # session is already ``busy`` (e.g. concurrent runs), this
            # is a no-op via ``derive_session_status_from_runs``.
            await refresh_session_status_from_runs(prisma_client, session_id)
            return NextRunResponse(
                run_id=claimed.id,
                prompt=claimed.prompt or {},
            )
        if asyncio.get_running_loop().time() >= deadline:
            return Response(status_code=204)
        await asyncio.sleep(NEXT_RUN_POLL_INTERVAL_SECONDS)


@router.post(
    "/v2/sessions/{session_id}/runs/{run_id}/events:append",
    response_class=ORJSONResponse,
    tags=["agent-internal"],
)
async def daemon_append_event(
    body: EventAppend,
    session_id: str = Path(...),
    run_id: str = Path(...),
    daemon: Dict[str, Any] = Depends(daemon_token_auth),
):
    """Append an event to a run.

    Seq is computed as ``MAX(seq) + 1`` on the server side. The
    ``(run_id, seq)`` unique constraint is the safety net — if two
    daemon emits race, the loser gets an IntegrityError, retries, and
    inserts at the next seq.

    If ``event_type`` is one of ``run_finished | run_cancelled |
    run_error``, also flip the run's ``status`` and ``terminated_at``
    inside the same transaction-ish flow.
    """
    prisma_client = await _get_prisma_client_or_503()

    # Cross-tenant defense: token is good for ``session_id`` but the run
    # itself must belong to that session.
    run = await prisma_client.db.litellm_agentrun.find_unique(where={"id": run_id})
    if run is None or run.session_id != session_id:
        raise HTTPException(status_code=404, detail="Run not found")

    # Compute next seq. Retry once on collision; the unique constraint
    # makes this safe.
    last_evt = await prisma_client.db.litellm_agentrunevent.find_first(
        where={"run_id": run_id},
        order={"seq": "desc"},
    )
    next_seq = (last_evt.seq + 1) if last_evt else 1

    try:
        evt = await prisma_client.db.litellm_agentrunevent.create(
            data={
                "run_id": run_id,
                "seq": next_seq,
                "event_type": body.event_type,
                "payload": body.payload,
            }
        )
    except Exception as exc:
        # Treat only ``(run_id, seq)`` unique-constraint collisions as
        # retryable here; let any other error bubble up unchanged so
        # callers don't see a misleading 409 ``event_seq_collision``
        # for a transient DB outage. We detect collisions either via
        # the dedicated Prisma error class or — for unit tests using
        # an in-memory fake — a string match on the marker.
        if not _is_seq_collision(exc):
            raise
        last_evt = await prisma_client.db.litellm_agentrunevent.find_first(
            where={"run_id": run_id},
            order={"seq": "desc"},
        )
        next_seq = (last_evt.seq + 1) if last_evt else 1
        try:
            evt = await prisma_client.db.litellm_agentrunevent.create(
                data={
                    "run_id": run_id,
                    "seq": next_seq,
                    "event_type": body.event_type,
                    "payload": body.payload,
                }
            )
        except Exception as exc2:
            if not _is_seq_collision(exc2):
                raise
            verbose_proxy_logger.exception(
                "events:append double-collision run_id=%s: %s", run_id, exc2
            )
            raise HTTPException(status_code=409, detail="event_seq_collision") from exc

    # Terminal-event handling: roll the run forward.
    new_status = RUN_TERMINAL_EVENT_TYPES.get(body.event_type)
    if new_status is not None:
        now = _now()
        await prisma_client.db.litellm_agentrun.update(
            where={"id": run_id},
            data={
                "status": new_status,
                "terminated_at": now,
                "updated_at": now,
                "result": (
                    body.payload.get("result")
                    if isinstance(body.payload, dict)
                    else None
                ),
            },
        )
        # Drive the session ``busy`` -> ``ready`` flip. Without this,
        # the parent session stays permanently ``busy`` after the first
        # run terminates — Greptile P1.
        await refresh_session_status_from_runs(prisma_client, session_id)

    return {
        "run_id": run_id,
        "seq": evt.seq,
        "event_type": evt.event_type,
    }
