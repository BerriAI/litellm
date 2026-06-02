"""
Run endpoints — POST/GET runs under a session, plus the SSE event stream
(``/stream``) and explicit cancel.

Every endpoint is owner-scoped (caller must own the parent session). The
SSE stream is resumable — clients pass ``starting_seq=N`` to skip events
they've already seen, and the stream emits keep-alive comments so proxies
don't reap the connection.
"""

import asyncio
import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Header, Query
from fastapi.responses import ORJSONResponse, StreamingResponse

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.agent_session_endpoints.constants import (
    EVENT_TYPE_RUN_CANCELLED,
    RUN_ACTIVE_STATUSES,
    RUN_STATUS_CANCELLED,
    RUN_STATUS_QUEUED,
    RUN_TERMINAL_STATUSES,
    SESSION_TERMINAL_STATUSES,
    SSE_KEEPALIVE_INTERVAL_SECONDS,
    SSE_POLL_INTERVAL_SECONDS,
    SSE_TERMINAL_QUIESCE_SECONDS,
)
from litellm.proxy.agent_session_endpoints.ids import new_run_id
from litellm.proxy.agent_session_endpoints.ownership import (
    assert_caller_can_mutate,
    assert_caller_owns_session,
)
from litellm.proxy.agent_session_endpoints.pagination import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    build_page_response,
    cursor_where_clause,
    normalize_limit,
)
from litellm.proxy.agent_session_endpoints.schemas import RunCreate, RunResponse
from litellm.proxy.agent_session_endpoints.serialization import run_row_to_response
from litellm.proxy.agent_session_endpoints.session_status import (
    refresh_session_status_from_runs,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


def _now() -> datetime:
    return datetime.now(timezone.utc)


async def _get_prisma_client_or_503():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="Database unavailable")
    return prisma_client


async def _load_session_or_404(prisma_client, session_id: str):
    session = await prisma_client.db.litellm_agentsession.find_unique(
        where={"id": session_id}
    )
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


async def _find_idempotent_run(prisma_client, session_id: str, idempotency_key: str):
    """Return the existing run for ``(session_id, idempotency_key)`` if any."""
    return await prisma_client.db.litellm_agentrun.find_first(
        where={
            "session_id": session_id,
            "idempotency_key": idempotency_key,
        }
    )


async def _has_active_run(prisma_client, session_id: str) -> bool:
    """True iff session has any run in queued/running."""
    existing = await prisma_client.db.litellm_agentrun.find_first(
        where={
            "session_id": session_id,
            "status": {"in": list(RUN_ACTIVE_STATUSES)},
        }
    )
    return existing is not None


async def _next_event_seq(prisma_client, run_id: str) -> int:
    last = await prisma_client.db.litellm_agentrunevent.find_first(
        where={"run_id": run_id},
        order={"seq": "desc"},
    )
    if last is None:
        return 1
    return last.seq + 1


@router.post(
    "/v2/sessions/{session_id}/runs",
    response_class=ORJSONResponse,
    response_model=RunResponse,
    tags=["runs"],
    dependencies=[Depends(user_api_key_auth)],
)
async def create_run(
    session_id: str,
    body: RunCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
):
    """Start a new run within a session.

    Concurrency rules:
      * 409 ``run_busy`` if another run is in queued/running.
      * 409 ``session_not_accepting_runs`` if session is not ready/busy.
      * Idempotency-Key returns the existing run on retry (same row).
    """
    assert_caller_can_mutate(user_api_key_dict)
    prisma_client = await _get_prisma_client_or_503()
    session = await _load_session_or_404(prisma_client, session_id)
    assert_caller_owns_session(user_api_key_dict, session)

    if session.status in SESSION_TERMINAL_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=f"Session is {session.status}; cannot create runs",
        )

    # Idempotent retry shortcuts the busy check — if the same key already
    # produced a run, return that run regardless of current state.
    if idempotency_key:
        existing = await _find_idempotent_run(
            prisma_client, session_id, idempotency_key
        )
        if existing is not None:
            return run_row_to_response(existing)

    if await _has_active_run(prisma_client, session_id):
        raise HTTPException(
            status_code=409,
            detail="run_busy: another run is queued/running for this session",
        )

    # Insert. The unique ``(session_id, idempotency_key)`` constraint is
    # the actual safety net for racing duplicates — if two requests with
    # the same key collide on insert, one gets the IntegrityError, retries
    # the find_first, and returns the winner's row.
    payload: Dict[str, Any] = {
        "id": new_run_id(),
        "session_id": session_id,
        "status": RUN_STATUS_QUEUED,
        "prompt": body.prompt,
        "idempotency_key": idempotency_key,
        "updated_at": _now(),
    }
    try:
        row = await prisma_client.db.litellm_agentrun.create(data=payload)
    except Exception as exc:
        # Idempotent-collision recovery.
        if idempotency_key:
            existing = await _find_idempotent_run(
                prisma_client, session_id, idempotency_key
            )
            if existing is not None:
                return run_row_to_response(existing)
        # Active-run race: another caller won the busy check between our
        # check and our insert. Return the canonical 409.
        active_other = await prisma_client.db.litellm_agentrun.find_first(
            where={
                "session_id": session_id,
                "status": {"in": list(RUN_ACTIVE_STATUSES)},
            }
        )
        if active_other is not None:
            raise HTTPException(
                status_code=409,
                detail="run_busy: another run is queued/running for this session",
            ) from exc
        raise

    verbose_proxy_logger.info("run.create id=%s session_id=%s", row.id, session_id)
    # Run was just created in ``queued``. Drive the session
    # ``ready`` -> ``busy`` flip so SDK consumers polling
    # GET /v2/sessions/{id} see the right status.
    await refresh_session_status_from_runs(prisma_client, session_id)
    return run_row_to_response(row)


@router.get(
    "/v2/sessions/{session_id}/runs/{run_id}",
    response_class=ORJSONResponse,
    response_model=RunResponse,
    tags=["runs"],
    dependencies=[Depends(user_api_key_auth)],
)
async def get_run(
    session_id: str,
    run_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    prisma_client = await _get_prisma_client_or_503()
    session = await _load_session_or_404(prisma_client, session_id)
    assert_caller_owns_session(user_api_key_dict, session)

    run = await prisma_client.db.litellm_agentrun.find_unique(where={"id": run_id})
    if run is None or run.session_id != session_id:
        raise HTTPException(status_code=404, detail="Run not found")
    return run_row_to_response(run)


@router.get(
    "/v2/sessions/{session_id}/runs",
    response_class=ORJSONResponse,
    tags=["runs"],
    dependencies=[Depends(user_api_key_auth)],
)
async def list_runs(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    cursor: Optional[str] = Query(default=None),
    limit: int = Query(default=DEFAULT_PAGE_LIMIT, ge=1, le=MAX_PAGE_LIMIT),
):
    """List runs for a session.

    Cursor-paginated. Pass ``?cursor=<next_cursor>`` from the previous
    response to fetch the next page.
    """
    prisma_client = await _get_prisma_client_or_503()
    session = await _load_session_or_404(prisma_client, session_id)
    assert_caller_owns_session(user_api_key_dict, session)

    page_limit = normalize_limit(limit)
    rows = await prisma_client.db.litellm_agentrun.find_many(
        where=cursor_where_clause({"session_id": session_id}, cursor),
        order=[{"created_at": "desc"}, {"id": "desc"}],
        take=page_limit + 1,
    )
    return build_page_response(
        rows,
        page_limit,
        lambda r: run_row_to_response(r).model_dump(),
    )


@router.post(
    "/v2/sessions/{session_id}/runs/{run_id}/cancel",
    response_class=ORJSONResponse,
    response_model=RunResponse,
    tags=["runs"],
    dependencies=[Depends(user_api_key_auth)],
)
async def cancel_run(
    session_id: str,
    run_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Mark a run cancelled and emit ``run_cancelled`` event.

    Idempotent: cancelling an already-terminal run returns the row
    unchanged (200, no extra event emitted).
    """
    assert_caller_can_mutate(user_api_key_dict)
    prisma_client = await _get_prisma_client_or_503()
    session = await _load_session_or_404(prisma_client, session_id)
    assert_caller_owns_session(user_api_key_dict, session)

    run = await prisma_client.db.litellm_agentrun.find_unique(where={"id": run_id})
    if run is None or run.session_id != session_id:
        raise HTTPException(status_code=404, detail="Run not found")

    if run.status in RUN_TERMINAL_STATUSES:
        return run_row_to_response(run)

    now = _now()
    updated = await prisma_client.db.litellm_agentrun.update(
        where={"id": run_id},
        data={
            "status": RUN_STATUS_CANCELLED,
            "terminated_at": now,
            "updated_at": now,
        },
    )
    next_seq = await _next_event_seq(prisma_client, run_id)
    try:
        await prisma_client.db.litellm_agentrunevent.create(
            data={
                "run_id": run_id,
                "seq": next_seq,
                "event_type": EVENT_TYPE_RUN_CANCELLED,
                "payload": {"reason": "user_cancel"},
            }
        )
    except Exception as exc:
        verbose_proxy_logger.warning(
            "run.cancel: skipped run_cancelled emit run=%s seq=%s: %s",
            run_id,
            next_seq,
            exc,
        )
    # Run just transitioned to ``cancelled`` (terminal). If it was the
    # only active run, flip the session ``busy`` -> ``ready``.
    await refresh_session_status_from_runs(prisma_client, session_id)
    return run_row_to_response(updated)


# ---------------------------------------------------------------------------
# SSE events stream — resumable via ``?starting_seq=N``
# ---------------------------------------------------------------------------


def _sse_event_data(seq: int, event_type: str, payload: Dict[str, Any]) -> str:
    """Format a single SSE event frame.

    The ``id:`` field carries the seq so clients can resume by using the
    HTTP ``Last-Event-ID`` header (or ``starting_seq`` query param).

    Wire shape: ``{seq, type, data}``. The DB column is ``event_type``/
    ``payload`` (untouched), but the JSON the SDK sees uses Cursor's
    naming so cross-language clients can share the same schema.
    """
    body = json.dumps(
        {"seq": seq, "type": event_type, "data": payload},
        default=str,
    )
    return f"id: {seq}\ndata: {body}\n\n"


async def _events_after(prisma_client, run_id: str, last_seen_seq: int) -> List[Any]:
    return await prisma_client.db.litellm_agentrunevent.find_many(
        where={"run_id": run_id, "seq": {"gt": last_seen_seq}},
        order={"seq": "asc"},
    )


async def _stream_run_events(
    run_id: str,
    starting_seq: int,
):
    """Async generator yielding SSE-formatted bytes.

    Loop:
      1. Fetch all events with seq > last_seen.
      2. Yield each (and bump last_seen).
      3. If the run is terminal AND no new events arrived this tick AND
         the quiesce window has elapsed, close cleanly.
      4. Sleep `SSE_POLL_INTERVAL_SECONDS`. Emit `:keepalive` every
         `SSE_KEEPALIVE_INTERVAL_SECONDS`.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return

    last_seen = max(starting_seq - 1, 0)
    # ``get_running_loop`` (not ``get_event_loop``) is the correct API
    # inside a running coroutine — see PEP 654 / Python 3.10 deprecation.
    last_keepalive = asyncio.get_running_loop().time()
    terminal_first_seen_at: Optional[float] = None

    while True:
        events = await _events_after(prisma_client, run_id, last_seen)
        for evt in events:
            yield _sse_event_data(evt.seq, evt.event_type, _safe_payload(evt.payload))
            last_seen = evt.seq

        run = await prisma_client.db.litellm_agentrun.find_unique(where={"id": run_id})
        if run is None:
            # Run vanished (cascade delete). Close.
            return

        now = asyncio.get_running_loop().time()
        if run.status in RUN_TERMINAL_STATUSES:
            if terminal_first_seen_at is None:
                terminal_first_seen_at = now
            elif (
                now - terminal_first_seen_at >= SSE_TERMINAL_QUIESCE_SECONDS
                and not events
            ):
                # Run is finished AND we've drained events for at least
                # the quiesce window AND nothing new came this tick.
                return

        if now - last_keepalive >= SSE_KEEPALIVE_INTERVAL_SECONDS:
            yield ": keepalive\n\n"
            last_keepalive = now

        await asyncio.sleep(SSE_POLL_INTERVAL_SECONDS)


def _safe_payload(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {"value": value}


@router.get(
    "/v2/sessions/{session_id}/runs/{run_id}/stream",
    tags=["runs"],
    dependencies=[Depends(user_api_key_auth)],
)
async def stream_run_events(
    session_id: str,
    run_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
    starting_seq: int = Query(default=0, ge=0),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
):
    """SSE stream of every event emitted on a run.

    Resumable: pass ``?starting_seq=N`` (or the standard SSE
    ``Last-Event-ID`` header) to skip already-seen events.
    """
    prisma_client = await _get_prisma_client_or_503()
    session = await _load_session_or_404(prisma_client, session_id)
    assert_caller_owns_session(user_api_key_dict, session)

    run = await prisma_client.db.litellm_agentrun.find_unique(where={"id": run_id})
    if run is None or run.session_id != session_id:
        raise HTTPException(status_code=404, detail="Run not found")

    # `Last-Event-ID` header beats explicit query param, matching the
    # SSE-reconnect convention browsers use.
    effective_starting_seq = starting_seq
    if last_event_id:
        try:
            effective_starting_seq = max(int(last_event_id) + 1, starting_seq)
        except ValueError:
            pass  # ignore malformed header, fall back to query param

    return StreamingResponse(
        _stream_run_events(run_id, effective_starting_seq),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
