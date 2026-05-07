"""SSE events endpoint for managed agents v2 — `GET /v2/sessions/:id/events`.

Streams normalized events for a session. The adapter yields
`(event_type, data_dict)` tuples — this handler formats them as SSE
frames per contract §6.6:

    event: <event_type>
    data: <json>
    \\n\\n

The first yielded event is always `connected` (synthesized by the
adapter). Subsequent events are translated from the underlying provider
bus per the §7 event-mapping table.

Pre-forward checks (per contract §7):
1. `Authorization` valid → handled by `user_api_key_auth` dependency.
2. Session row exists for `created_by = caller` → else 404.
3. `session.status == "ready"` → else 503 (`provisioning`) or 404
   (`terminated`/`error`).
4. Resolve `sandbox_url` + `sandbox_metadata.opencode_session_id`.

Once the SSE stream has started we cannot change the HTTP status code,
so a `SandboxUnreachableError` raised mid-stream is converted to an
`error` SSE event and the stream closes cleanly.
"""

import json
from typing import Any, AsyncIterator, Dict, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from litellm.managed_agents.adapters.base import (
    SandboxAdapter,
    SandboxUnreachableError,
)
from litellm.managed_agents.adapters.registry import get_adapter
from litellm.managed_agents.db import get_session
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

# Mirrors `agents.py` — fallback owner when no user_id is attached to the
# verification token (master-key call, etc.). Without this, master-key
# callers cannot read back sessions they created (the row stores
# "default_user" but the lookup would pass None).
_DEFAULT_CREATED_BY = "default_user"

# SSE response headers per contract §6.6.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


# ---------------------------------------------------------------------------
# Pre-forward helper
# ---------------------------------------------------------------------------
# TODO: extract to shared helper — `litellm/proxy/managed_agents_endpoints/_helpers.py`
# once the messages endpoint lands. Both endpoints need the same logic.


async def _load_ready_session(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth,
) -> Dict[str, Any]:
    """Load a session row, scope to caller, and validate it is ``ready``.

    Returns the session row dict.

    Raises:
        HTTPException(404) if the session does not exist for this caller,
            or if the session is in a terminal state (``terminated``/``error``).
        HTTPException(503) if the session is still provisioning.
        HTTPException(500) if the prisma client is not initialized.
    """
    # Lazy import to avoid module-load circular import with proxy_server.
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=500, detail="Database not initialized")

    session_row = await get_session(
        prisma_client,
        session_id=session_id,
        created_by=user_api_key_dict.user_id or _DEFAULT_CREATED_BY,
    )
    if not session_row:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    status = session_row.get("status")
    if status == "provisioning":
        raise HTTPException(
            status_code=503,
            detail="Session not ready",
            headers={"Retry-After": "5"},
        )
    if status != "ready":
        # terminated, error, or any other non-ready terminal state →
        # treat as not-found per contract §7 failure modes.
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

    return session_row


def _resolve_adapter_handle(
    session_row: Dict[str, Any],
) -> Tuple[SandboxAdapter, str, str]:
    """Resolve adapter + sandbox_url + opencode_session_id from a session row.

    Raises HTTPException(500) if any required field is missing — this
    indicates an internally-corrupted row and should never happen for a
    session in ``ready`` status.
    """
    sandbox_type = session_row.get("sandbox_type")
    sandbox_url = session_row.get("sandbox_url")
    sandbox_metadata = session_row.get("sandbox_metadata") or {}
    if isinstance(sandbox_metadata, str):
        # Defensive: some Prisma clients return JSON columns as strings.
        try:
            sandbox_metadata = json.loads(sandbox_metadata)
        except (TypeError, ValueError):
            sandbox_metadata = {}
    opencode_session_id = (
        sandbox_metadata.get("opencode_session_id")
        if isinstance(sandbox_metadata, dict)
        else None
    )

    if not sandbox_type or not sandbox_url or not opencode_session_id:
        raise HTTPException(
            status_code=500, detail="Session row is missing sandbox state"
        )

    try:
        adapter = get_adapter(sandbox_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return adapter, sandbox_url, opencode_session_id


# ---------------------------------------------------------------------------
# SSE formatter
# ---------------------------------------------------------------------------


async def _format_sse(
    adapter: SandboxAdapter,
    sandbox_url: str,
    oc_sid: str,
    our_session_id: str,
) -> AsyncIterator[bytes]:
    """Format adapter event tuples as SSE frames.

    Once the stream has started we can't switch HTTP status codes, so a
    `SandboxUnreachableError` becomes an inline `error` event and the
    stream closes. Client disconnects are handled by Starlette: when the
    consumer cancels the iterator, the underlying adapter's HTTP stream
    is aborted via the async context manager.
    """
    try:
        async for event_type, data in adapter.stream_events(
            sandbox_url, oc_sid, our_session_id
        ):
            yield (f"event: {event_type}\ndata: {json.dumps(data)}\n\n").encode()
    except SandboxUnreachableError:
        error_payload = json.dumps({"error": "Sandbox unreachable"})
        yield f"event: error\ndata: {error_payload}\n\n".encode()


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------


@router.get("/v2/sessions/{session_id}/events")
async def stream_events(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> StreamingResponse:
    """SSE stream of normalized events for a session.

    See contract §6.6 — first event is always ``connected`` (synthesized
    by the adapter), followed by a stream of ``message.*`` events
    translated from the upstream sandbox bus. Pre-forward checks happen
    before streaming starts so 404/503 responses use the correct status
    code; once streaming begins, errors are delivered as inline ``error``
    events.
    """
    session_row = await _load_ready_session(session_id, user_api_key_dict)
    adapter, sandbox_url, oc_sid = _resolve_adapter_handle(session_row)

    return StreamingResponse(
        _format_sse(adapter, sandbox_url, oc_sid, session_id),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
