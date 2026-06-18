"""FastAPI router for `GET /v2/sessions/:id` (LIT-2923) and
`GET /v2/agents/:agent_id/sessions` (UI listing).

Returns the session row scoped to the caller. Per contract §6.3 the response
shape matches `POST /v2/sessions` — sandbox spec is reassembled from the
flat `sandbox_*` columns on the row, and `sandbox_url` / `sandbox_metadata`
are NEVER included in the response (they are internal-only state).
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException

from litellm.managed_agents.db import (
    get_agent,
    get_session,
    list_sessions,
    list_sessions_for_agent,
)
from litellm.managed_agents.types import Repo, SandboxSpec, SessionList, SessionRow
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()

# Mirrors `agents.py` — fallback owner when no user_id is attached to the
# verification token (master-key call, etc.).
_DEFAULT_CREATED_BY = "default_user"


def _row_to_session_response(row: Dict[str, Any]) -> SessionRow:
    """Map a DB row dict to the public SessionRow response.

    Strips internal-only fields (`sandbox_url`, `sandbox_metadata`) and
    reassembles the `sandbox` sub-object from the flat `sandbox_*` columns
    on the row.
    """
    sandbox = SandboxSpec(
        type=row["sandbox_type"],
        size=row["sandbox_size"],
        timeout_minutes=row["sandbox_timeout_minutes"],
        idle_timeout_minutes=row["sandbox_idle_timeout_minutes"],
        image=row.get("sandbox_image"),
    )

    repos_raw = row.get("repos") or []
    repos: List[Repo] = [Repo(**r) for r in repos_raw]

    return SessionRow(
        id=row["id"],
        agent_id=row["agent_id"],
        sandbox=sandbox,
        status=row["status"],
        repos=repos,
        created_by=row.get("created_by"),
        created_at=row["created_at"],
        terminated_at=row.get("terminated_at"),
    )


@router.get(
    "/v2/sessions",
    response_model=SessionList,
    tags=["managed-agents-v2"],
)
async def list_sessions_endpoint(
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    cursor: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SessionList:
    """List ALL sessions for the caller, optionally filtered by `agent_id`
    and/or `status`.

    This is the global counterpart to `GET /v2/agents/:agent_id/sessions` —
    used by the UI's top-level Sessions view so the caller doesn't have to
    pre-pick an agent. Scoped to `created_by` (same no-leak rule). Public
    response rows mirror `GET /v2/sessions/:id`; `sandbox_url` and
    `sandbox_metadata` are stripped.

    Pagination: simple offset-based, same shape as `GET /v2/agents`.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    created_by = user_api_key_dict.user_id or _DEFAULT_CREATED_BY

    if cursor is None or cursor == "":
        skip = 0
    else:
        try:
            skip = int(cursor)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid cursor: {cursor!r}")

    rows = await list_sessions(
        prisma_client,
        created_by=created_by,
        limit=limit + 1,
        skip=skip,
        agent_id=agent_id,
        status=status,
    )
    has_more = len(rows) > limit
    page_rows = rows[:limit]

    next_cursor = str(skip + limit) if has_more else None

    return SessionList(
        data=[_row_to_session_response(r) for r in page_rows],
        next_cursor=next_cursor,
        has_more=has_more,
    )


@router.get("/v2/sessions/{session_id}", response_model=SessionRow)
async def get_session_endpoint(
    session_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SessionRow:
    """Fetch a managed-agent session by id, scoped to the caller.

    Per contract §6.3:
      - 200 with SessionRow shape if found.
      - 404 if missing OR owned by another caller (do NOT 403 — that would
        leak existence of sessions to non-owners).
      - 401 handled by the auth dependency.

    `sandbox_url` and `sandbox_metadata` are stored on the row but stripped
    here — they are internal-only state used by the adapter.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    row = await get_session(
        prisma_client,
        session_id=session_id,
        created_by=user_api_key_dict.user_id or _DEFAULT_CREATED_BY,
    )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )

    return _row_to_session_response(row)


@router.get(
    "/v2/agents/{agent_id}/sessions",
    response_model=SessionList,
    tags=["managed-agents-v2"],
)
async def list_sessions_for_agent_endpoint(
    agent_id: str,
    status: Optional[str] = None,
    limit: int = 50,
    cursor: Optional[str] = None,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> SessionList:
    """List sessions for a given agent, scoped to the caller.

    First verifies the agent exists and is owned by the caller (404 otherwise
    — same no-leak rule as `GET /v2/sessions/:id`). Then returns a paginated
    list of sessions for that agent, newest first.

    Pagination: simple offset-based, same shape as `GET /v2/agents`.
    Optional `status` filter narrows by SessionStatus.

    Public response shape mirrors `GET /v2/sessions/:id` for each row —
    `sandbox_url` and `sandbox_metadata` are stripped.
    """
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500,
            detail={"error": CommonProxyErrors.db_not_connected_error.value},
        )

    created_by = user_api_key_dict.user_id or _DEFAULT_CREATED_BY

    # 1. Verify the agent exists for this caller (prevents leaking session
    #    existence and gives a clear 404 for unknown/unowned agents).
    agent_row = await get_agent(
        prisma_client,
        agent_id=agent_id,
        created_by=created_by,
    )
    if not agent_row:
        raise HTTPException(
            status_code=404,
            detail=f"Agent {agent_id} not found",
        )

    # 2. Parse cursor → offset.
    if cursor is None or cursor == "":
        skip = 0
    else:
        try:
            skip = int(cursor)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"Invalid cursor: {cursor!r}")

    # 3. Fetch one extra row to detect ``has_more``.
    rows = await list_sessions_for_agent(
        prisma_client,
        agent_id=agent_id,
        created_by=created_by,
        limit=limit + 1,
        skip=skip,
        status=status,
    )
    has_more = len(rows) > limit
    page_rows = rows[:limit]

    next_cursor = str(skip + limit) if has_more else None

    return SessionList(
        data=[_row_to_session_response(r) for r in page_rows],
        next_cursor=next_cursor,
        has_more=has_more,
    )
