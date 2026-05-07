"""FastAPI router for `GET /v2/sessions/:id` (LIT-2923).

Returns the session row scoped to the caller. Per contract §6.3 the response
shape matches `POST /v2/sessions` — sandbox spec is reassembled from the
flat `sandbox_*` columns on the row, and `sandbox_url` / `sandbox_metadata`
are NEVER included in the response (they are internal-only state).
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException

from litellm.managed_agents.db import get_session
from litellm.managed_agents.types import Repo, SandboxSpec, SessionRow
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth

router = APIRouter()


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
        created_by=user_api_key_dict.user_id,
    )

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"Session {session_id} not found",
        )

    return _row_to_session_response(row)
