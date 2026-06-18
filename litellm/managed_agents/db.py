"""Prisma helpers for managed agents.

Wave 1 ships only the helpers Wave 2 needs to start writing handlers.
All helpers are scoped by `created_by` so callers cannot see other users'
agents/sessions.

Pattern: `prisma` is imported lazily inside helper bodies that need
`prisma.Json(...)` — the `prisma` package is generated at runtime by the
Prisma CLI, may not be importable at module load in all environments
(matches the convention used in `litellm/proxy/management_endpoints/
key_management_endpoints.py` and `litellm/proxy/db/exception_handler.py`).
The `prisma_client` instance itself comes from
`litellm.proxy.proxy_server` and is passed in by the caller.

NEVER use raw SQL — use Prisma model methods only.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from litellm.proxy.utils import PrismaClient


# ---------------------------------------------------------------------------
# Agent helpers
# ---------------------------------------------------------------------------


async def insert_agent(
    prisma_client: PrismaClient,
    *,
    agent_id: str,
    name: str,
    config: Dict[str, Any],
    created_by: Optional[str],
) -> Dict[str, Any]:
    """Insert a new managed-agent row.

    `config` is the raw JSON-serializable dict — caller is responsible for
    encrypting any secrets in-place (e.g. the `litellm_api_key`).
    """
    import prisma  # generated at runtime; see module docstring

    now = datetime.now(timezone.utc)
    row = await prisma_client.db.litellm_managedagent.create(
        data={
            "id": agent_id,
            "name": name,
            "config": prisma.Json(config),  # type: ignore[attr-defined]
            "created_by": created_by,
            "created_at": now,
            "updated_at": now,
        }
    )
    return _row_to_dict(row)


async def get_agent(
    prisma_client: PrismaClient,
    *,
    agent_id: str,
    created_by: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Read an agent scoped to `created_by`. Returns None if not found."""
    row = await prisma_client.db.litellm_managedagent.find_first(
        where={"id": agent_id, "created_by": created_by}
    )
    return _row_to_dict(row) if row else None


async def get_agent_by_name(
    prisma_client: PrismaClient,
    *,
    name: str,
    created_by: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Lookup an agent by `(name, created_by)` for duplicate detection."""
    row = await prisma_client.db.litellm_managedagent.find_first(
        where={"name": name, "created_by": created_by}
    )
    return _row_to_dict(row) if row else None


async def list_agents(
    prisma_client: PrismaClient,
    *,
    created_by: Optional[str],
    limit: int,
    skip: int,
) -> List[Dict[str, Any]]:
    """List agents scoped to `created_by`, newest first.

    Simple offset-based pagination — `skip` is the offset, `limit` is the
    page size. Returns a list of plain dicts; the caller is responsible for
    masking secrets and reshaping into the response model.
    """
    rows = await prisma_client.db.litellm_managedagent.find_many(
        where={"created_by": created_by},
        take=limit,
        skip=skip,
        order={"created_at": "desc"},
    )
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


async def get_session(
    prisma_client: PrismaClient,
    *,
    session_id: str,
    created_by: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Read a session scoped to `created_by`. Returns None if not found."""
    row = await prisma_client.db.litellm_managedagentsession.find_first(
        where={"id": session_id, "created_by": created_by}
    )
    return _row_to_dict(row) if row else None


async def list_sessions_for_agent(
    prisma_client: PrismaClient,
    *,
    agent_id: str,
    created_by: Optional[str],
    limit: int,
    skip: int,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List sessions for a given `agent_id`, scoped to `created_by`, newest first.

    Optional `status` filter narrows by `SessionStatus`. Simple offset-based
    pagination — `skip` is the offset, `limit` is the page size. Returns a
    list of plain dicts; the caller maps each to the public response shape
    (stripping `sandbox_url` / `sandbox_metadata`).
    """
    where: Dict[str, Any] = {"agent_id": agent_id, "created_by": created_by}
    if status is not None:
        where["status"] = status

    rows = await prisma_client.db.litellm_managedagentsession.find_many(
        where=where,
        take=limit,
        skip=skip,
        order={"created_at": "desc"},
    )
    return [_row_to_dict(r) for r in rows]


async def list_sessions(
    prisma_client: PrismaClient,
    *,
    created_by: Optional[str],
    limit: int,
    skip: int,
    agent_id: Optional[str] = None,
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List ALL sessions for a caller, optionally filtered by agent_id and/or status.

    Same shape as `list_sessions_for_agent` minus the required `agent_id`.
    Used by `GET /v2/sessions` to power a global session listing for the UI.
    Newest-first ordering, simple offset-based pagination. Returns a list of
    plain dicts; the caller maps each to the public response shape (stripping
    `sandbox_url` / `sandbox_metadata`).
    """
    where: Dict[str, Any] = {"created_by": created_by}
    if agent_id is not None:
        where["agent_id"] = agent_id
    if status is not None:
        where["status"] = status

    rows = await prisma_client.db.litellm_managedagentsession.find_many(
        where=where,
        take=limit,
        skip=skip,
        order={"created_at": "desc"},
    )
    return [_row_to_dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Message helpers (placeholder — opencode is the source of truth for v2)
# ---------------------------------------------------------------------------


async def list_messages(
    prisma_client: PrismaClient,
    *,
    session_id: str,
    created_by: Optional[str],
    limit: int = 50,
    cursor: Optional[str] = None,
    role: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Placeholder list-messages helper.

    v2 does NOT store messages locally — opencode is the source of truth.
    The adapter (`litellm/managed_agents/adapters/opencode.py`) will fetch
    messages from opencode and normalize them. This helper exists so future
    adapters that DO persist messages locally have a stable entrypoint.

    For now, returns an empty list. Wave 2 callers should use the adapter
    directly, not this function, for the opencode happy path.
    """
    _ = (prisma_client, session_id, created_by, limit, cursor, role)
    return []


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _row_to_dict(row: Any) -> Dict[str, Any]:
    """Convert a Prisma row object to a plain dict.

    Prisma rows are pydantic models — use `model_dump()` to get a JSON-safe
    dict. Falls back to `dict(row)` for legacy clients.
    """
    if row is None:
        return {}
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "dict"):
        return row.dict()
    return dict(row)
