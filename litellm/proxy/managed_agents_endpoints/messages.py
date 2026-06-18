"""Messaging endpoints for managed agents v2.

Implements `POST /v2/sessions/:id/messages` (LIT-2920),
`GET /v2/sessions/:id/messages` (LIT-2920), and
`POST /v2/sessions/:id/abort` per contract §6.4 / §6.5.

Both message endpoints share the same pre-forward checks (contract §7):
  1. Auth (handled by Depends).
  2. Session row exists for caller → 404 if missing.
  3. session.status == "ready" → else 503 (provisioning) or 404
     (terminated/error).
  4. Resolve sandbox_url + opencode_session_id from row.

The user message is NOT persisted in our DB for v2 — opencode is the
source of truth. The returned `msg_*` id is a local handle; subsequent
GETs reflect opencode's state.

The SSE `/events` endpoint is owned by a separate module — do not add it
here.
"""

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm.managed_agents.adapters.base import (
    SandboxBadGatewayError,
    SandboxUnreachableError,
)
from litellm.managed_agents.adapters.registry import get_adapter
from litellm.managed_agents.db import get_agent, get_session
from litellm.managed_agents.id_utils import new_message_id
from litellm.managed_agents.types import (
    CreateMessageRequest,
    MessageList,
    MessageRow,
)
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

# Mirrors `agents.py` — fallback owner when no user_id is attached to the
# verification token (master-key call, etc.). Without this, master-key
# callers cannot read back sessions they created (the row stores
# "default_user" but the lookup would pass None).
_DEFAULT_CREATED_BY = "default_user"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


async def _load_ready_session(
    session_id: str,
    created_by: Optional[str],
) -> Dict[str, Any]:
    """Load the session row for a caller, validating it is `ready`.

    Returns the session row as a plain dict. Raises HTTPException with the
    correct status code per contract §7:
      - 500 if the prisma client is not connected.
      - 404 if the session does not exist for this caller.
      - 503 if the session is still provisioning (Retry-After: 5).
      - 404 if the session is terminated or in an error state.

    The 503 case carries a `Retry-After: 5` header per contract §7
    "Failure modes — Session still provisioning".
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    session = await get_session(
        prisma_client=prisma_client,
        session_id=session_id,
        created_by=created_by,
    )
    if not session:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )

    status = session.get("status")
    if status == "provisioning":
        raise HTTPException(
            status_code=503,
            detail="Session not ready",
            headers={"Retry-After": "5"},
        )
    if status != "ready":
        # terminated, error, or unknown → present as 404 to the caller
        # (contract §7: "Session terminated → 404").
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )

    return session


def _resolve_sandbox(session: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the `sandbox_url` and `opencode_session_id` from the row.

    Raises 502 if the row is in `ready` state but missing required
    sandbox routing fields — that indicates a bad-gateway condition where
    the session row is internally inconsistent.
    """
    sandbox_url = session.get("sandbox_url")
    metadata = session.get("sandbox_metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    opencode_session_id = metadata.get("opencode_session_id")

    if not sandbox_url or not opencode_session_id:
        raise HTTPException(
            status_code=502,
            detail={"error": "Bad gateway"},
        )

    return {
        "sandbox_url": sandbox_url,
        "opencode_session_id": opencode_session_id,
        "sandbox_type": session.get("sandbox_type", "opencode"),
    }


async def _resolve_model(
    session: Dict[str, Any],
    request_model: Optional[str],
    created_by: Optional[str],
) -> Optional[str]:
    """Pick the model to use: request override, else agent default.

    Returns None only if neither side provides a model — opencode tolerates
    a None model by falling back to its own default, so we don't enforce
    presence here.
    """
    if request_model:
        return request_model

    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return None

    agent_id = session.get("agent_id")
    if not agent_id:
        return None

    agent = await get_agent(
        prisma_client=prisma_client,
        agent_id=agent_id,
        created_by=created_by,
    )
    if not agent:
        return None

    config = agent.get("config") or {}
    if not isinstance(config, dict):
        return None
    model = config.get("model")
    return model if isinstance(model, str) and model else None


# ---------------------------------------------------------------------------
# POST /v2/sessions/:id/messages
# ---------------------------------------------------------------------------


@router.post(
    "/v2/sessions/{session_id}/messages",
    response_model=MessageRow,
    status_code=202,
    tags=["managed-agents"],
)
async def send_message(
    session_id: str,
    request: CreateMessageRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> MessageRow:
    """Forward a user message to the sandbox. Returns 202 with a
    synthesized user `MessageRow` whose `status` is `in_progress`.

    The actual assistant response streams via the SSE
    `/v2/sessions/:id/events` endpoint (owned by a separate module).
    """
    session = await _load_ready_session(
        session_id=session_id,
        created_by=user_api_key_dict.user_id or _DEFAULT_CREATED_BY,
    )
    sandbox = _resolve_sandbox(session)

    try:
        adapter = get_adapter(sandbox["sandbox_type"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    model_to_use = await _resolve_model(
        session=session,
        request_model=request.model,
        created_by=user_api_key_dict.user_id or _DEFAULT_CREATED_BY,
    )

    try:
        await adapter.send_message(
            sandbox["sandbox_url"],
            sandbox["opencode_session_id"],
            request.content,
            model_to_use,
        )
    except SandboxUnreachableError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "Sandbox unreachable"},
        ) from e
    except SandboxBadGatewayError as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "Bad gateway"},
        ) from e

    now = datetime.now(timezone.utc)
    return MessageRow(
        id=new_message_id(),
        session_id=session_id,
        role="user",
        content=request.content,
        model=model_to_use,
        status="in_progress",
        created_at=now,
        completed_at=None,
    )


# ---------------------------------------------------------------------------
# GET /v2/sessions/:id/messages
# ---------------------------------------------------------------------------


@router.get(
    "/v2/sessions/{session_id}/messages",
    response_model=MessageList,
    tags=["managed-agents"],
)
async def list_messages(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=500),
    cursor: Optional[str] = Query(default=None),
    role: Optional[str] = Query(default=None, pattern="^(user|assistant)$"),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> MessageList:
    """Return the message history for a session.

    Forwards to the sandbox adapter, normalizes the result, and applies
    the optional `role` filter in-memory. Pagination is a future
    enhancement — for MVP `next_cursor` is None and `has_more` is False
    (contract §6.5 explicitly defers opencode pagination).
    """
    _ = cursor  # MVP: opencode pagination is deferred.

    session = await _load_ready_session(
        session_id=session_id,
        created_by=user_api_key_dict.user_id or _DEFAULT_CREATED_BY,
    )
    sandbox = _resolve_sandbox(session)

    try:
        adapter = get_adapter(sandbox["sandbox_type"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        messages = await adapter.list_messages(
            sandbox["sandbox_url"],
            sandbox["opencode_session_id"],
            session_id,
            limit,
        )
    except SandboxUnreachableError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "Sandbox unreachable"},
        ) from e
    except SandboxBadGatewayError as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "Bad gateway"},
        ) from e

    if role:
        messages = [m for m in messages if m.role == role]

    return MessageList(
        data=messages,
        next_cursor=None,
        has_more=False,
    )


# ---------------------------------------------------------------------------
# POST /v2/sessions/:id/abort
# ---------------------------------------------------------------------------


@router.post(
    "/v2/sessions/{session_id}/abort",
    tags=["managed-agents"],
)
async def abort_session(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> Dict[str, Any]:
    """Abort the in-flight turn for a session.

    Forwards to the sandbox adapter's ``abort`` which POSTs to
    ``<sandbox_url>/session/<oc_sid>/abort``. Best-effort: provider 4xx is
    swallowed by the adapter (the session may already be idle). Connection
    failures bubble up as 504.
    """
    session = await _load_ready_session(
        session_id=session_id,
        created_by=user_api_key_dict.user_id or _DEFAULT_CREATED_BY,
    )
    sandbox = _resolve_sandbox(session)

    try:
        adapter = get_adapter(sandbox["sandbox_type"])
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        await adapter.abort(
            sandbox["sandbox_url"],
            sandbox["opencode_session_id"],
        )
    except SandboxUnreachableError as e:
        raise HTTPException(
            status_code=504,
            detail={"error": "Sandbox unreachable"},
        ) from e
    except SandboxBadGatewayError as e:
        raise HTTPException(
            status_code=502,
            detail={"error": "Bad gateway"},
        ) from e

    return {"id": session_id, "aborted": True}
