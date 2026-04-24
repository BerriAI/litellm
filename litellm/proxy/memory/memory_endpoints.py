"""
MEMORY MANAGEMENT

CRUD endpoints for user/team-scoped memory entries.

POST   /v1/memory               - Create a memory entry
GET    /v1/memory               - List memory entries visible to the caller
GET    /v1/memory/{key}         - Get a single memory entry by key
PUT    /v1/memory/{key}         - Upsert (create or update) a memory entry by key
DELETE /v1/memory/{key}         - Delete a memory entry by key

Scoping:
- Rows carry both `user_id` and `team_id` (each optional).
- Visibility: PROXY_ADMIN sees all rows. Non-admin callers see rows whose
  `user_id` matches their own OR whose `team_id` matches their own.
- On create, `user_id`/`team_id` default to the caller's identity unless
  the caller is a PROXY_ADMIN who explicitly supplies a different scope.
"""

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import (
    CommonProxyErrors,
    LitellmUserRoles,
    UserAPIKeyAuth,
)
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.memory_management import (
    LiteLLM_MemoryRow,
    MemoryCreateRequest,
    MemoryDeleteResponse,
    MemoryListResponse,
    MemoryUpdateRequest,
)

router = APIRouter()


def _is_admin(user_api_key_dict: UserAPIKeyAuth) -> bool:
    return user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN


def _visibility_filter(user_api_key_dict: UserAPIKeyAuth) -> Optional[dict]:
    """
    Prisma `where` fragment restricting rows to those the caller can see.
    Returns None for admins (no restriction).
    """
    if _is_admin(user_api_key_dict):
        return None
    ors: List[dict] = []
    if user_api_key_dict.user_id:
        ors.append({"user_id": user_api_key_dict.user_id})
    if user_api_key_dict.team_id:
        ors.append({"team_id": user_api_key_dict.team_id})
    if not ors:
        # Caller has neither user_id nor team_id — match nothing.
        return {"memory_id": "__no_match__"}
    return {"OR": ors}


def _row_to_model(row: Any) -> LiteLLM_MemoryRow:
    return LiteLLM_MemoryRow(
        memory_id=row.memory_id,
        key=row.key,
        value=row.value,
        metadata=getattr(row, "metadata", None),
        user_id=row.user_id,
        team_id=row.team_id,
        created_at=row.created_at,
        created_by=row.created_by,
        updated_at=row.updated_at,
        updated_by=row.updated_by,
    )


def _require_prisma():
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(
            status_code=500, detail=CommonProxyErrors.db_not_connected_error.value
        )
    return prisma_client


def _internal_error(
    log_message: str, exc: Exception, default_detail: str
) -> HTTPException:
    """
    Build a 500 HTTPException with a generic, caller-safe `detail` while
    logging the actual exception server-side. Avoids leaking internal Prisma /
    DB details (table names, columns, connection metadata) to API callers.
    """
    verbose_proxy_logger.exception(log_message, exc)
    return HTTPException(status_code=500, detail=default_detail)


def _assert_write_access(row: Any, user_api_key_dict: UserAPIKeyAuth) -> None:
    """
    Enforce ownership for mutations (PUT/DELETE).

    The visibility filter uses an OR (`user_id == caller OR team_id == caller`)
    so team members can READ each other's team-scoped rows. That's intentional
    for list/get. For writes, broader visibility != broader authority — without
    this check, any team member could overwrite or delete a teammate's
    personal row whenever both `user_id` and `team_id` are stamped on it.

    Rule (non-admin): the row must be authored by the caller's user_id, or be
    a "pure team row" (no user_id stamped) within the caller's team.
    Admins may modify any row.
    """
    if _is_admin(user_api_key_dict):
        return
    row_user_id = getattr(row, "user_id", None)
    row_team_id = getattr(row, "team_id", None)

    # Personal ownership: caller authored this row.
    if row_user_id and row_user_id == user_api_key_dict.user_id:
        return
    # Pure team row (no user_id stamped) inside the caller's team.
    if (
        row_user_id is None
        and row_team_id is not None
        and row_team_id == user_api_key_dict.team_id
    ):
        return
    raise HTTPException(
        status_code=403,
        detail="You do not have permission to modify this memory entry.",
    )


def _is_unique_violation(exc: Exception) -> bool:
    """
    Detect a Prisma unique-constraint violation.

    Prefer the typed error code `P2002` from `PrismaClientKnownRequestError`;
    fall back to string matching so we stay robust across Prisma versions
    where the typed class may be unavailable or differently named.
    """
    code = getattr(exc, "code", None)
    if code == "P2002":
        return True
    msg = str(exc)
    return (
        "P2002" in msg or "Unique" in msg or "unique" in msg or "UniqueViolation" in msg
    )


def _resolve_scope(
    user_api_key_dict: UserAPIKeyAuth,
    requested_user_id: Optional[str],
    requested_team_id: Optional[str],
) -> tuple[Optional[str], Optional[str]]:
    """
    Resolve the (user_id, team_id) to stamp on a new row.

    - PROXY_ADMIN: may override either dimension via the request body.
    - Everyone else: the requested values must match their own (or be omitted).

    Also rejects identity-less creation: a row with both user_id and team_id
    NULL is invisible to every non-admin caller (the visibility filter would
    never match it), so we refuse to create orphan rows unless the caller is
    a PROXY_ADMIN who is explicitly stamping a global/shared row.
    """
    if _is_admin(user_api_key_dict):
        user_id = (
            requested_user_id
            if requested_user_id is not None
            else user_api_key_dict.user_id
        )
        team_id = (
            requested_team_id
            if requested_team_id is not None
            else user_api_key_dict.team_id
        )
        return user_id, team_id

    if requested_user_id is not None and requested_user_id != user_api_key_dict.user_id:
        raise HTTPException(
            status_code=403,
            detail="Only proxy admins may set user_id to a different user.",
        )
    if requested_team_id is not None and requested_team_id != user_api_key_dict.team_id:
        raise HTTPException(
            status_code=403,
            detail="Only proxy admins may set team_id to a different team.",
        )
    user_id = user_api_key_dict.user_id
    team_id = user_api_key_dict.team_id
    if not user_id and not team_id:
        # Orphan row: no user_id and no team_id means no non-admin can ever
        # see it again via the visibility filter. Reject up front.
        raise HTTPException(
            status_code=400,
            detail=(
                "Cannot create a memory entry without a user_id or team_id. "
                "Authenticate with a key that has a user_id or team_id, or call "
                "as a proxy admin."
            ),
        )
    return user_id, team_id


@router.post(
    "/v1/memory",
    tags=["memory management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_MemoryRow,
)
async def create_memory(
    body: MemoryCreateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Create a new memory entry for the caller (or, for admins, any scope)."""
    prisma_client = _require_prisma()
    user_id, team_id = _resolve_scope(user_api_key_dict, body.user_id, body.team_id)

    # Prisma's Python client rejects `metadata=None` on a `Json?` field —
    # the field must be omitted entirely to store SQL NULL. Build the data
    # dict conditionally so we only include metadata when the caller sent it.
    create_data: dict = {
        "key": body.key,
        "value": body.value,
        "user_id": user_id,
        "team_id": team_id,
        "created_by": user_api_key_dict.user_id,
        "updated_by": user_api_key_dict.user_id,
    }
    if body.metadata is not None:
        create_data["metadata"] = body.metadata

    try:
        row = await prisma_client.db.litellm_memorytable.create(data=create_data)
    except Exception as e:
        # Key is globally unique. Any duplicate → 409.
        if _is_unique_violation(e):
            raise HTTPException(
                status_code=409,
                detail=f"Memory with key '{body.key}' already exists.",
            )
        raise _internal_error(
            "Error creating memory: %s",
            e,
            "Internal error creating memory entry.",
        )

    return _row_to_model(row)


@router.get(
    "/v1/memory",
    tags=["memory management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=MemoryListResponse,
)
async def list_memory(
    key: Optional[str] = Query(None, description="Filter by exact key match."),
    key_prefix: Optional[str] = Query(
        None,
        description=(
            "Filter by key prefix (Redis-style namespace scan). "
            "Mutually exclusive with `key`; if both are provided, `key_prefix` wins."
        ),
    ),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """List memory entries visible to the caller."""
    prisma_client = _require_prisma()

    # Build the key filter first (prefix wins if both `key` and `key_prefix`
    # are passed). Then AND it with the visibility filter via an explicit
    # top-level "AND" — safer than `dict.update` since future visibility
    # filters could grow an "OR" key that would clobber this one if merged
    # by key.
    key_filter: dict = {}
    if key_prefix is not None:
        key_filter["key"] = {"startsWith": key_prefix}
    elif key is not None:
        key_filter["key"] = key

    vis = _visibility_filter(user_api_key_dict)
    where: dict
    if vis is None:
        where = key_filter
    elif not key_filter:
        where = vis
    else:
        where = {"AND": [key_filter, vis]}

    try:
        total = await prisma_client.db.litellm_memorytable.count(where=where)
        rows = await prisma_client.db.litellm_memorytable.find_many(
            where=where,
            order={"updated_at": "desc"},
            skip=(page - 1) * page_size,
            take=page_size,
        )
    except Exception as e:
        raise _internal_error(
            "Error listing memory: %s", e, "Internal error listing memory entries."
        )

    return MemoryListResponse(memories=[_row_to_model(r) for r in rows], total=total)


async def _find_memory_for_caller(
    prisma_client: Any, key: str, user_api_key_dict: UserAPIKeyAuth
) -> Any:
    """Look up a memory row by key, scoped to the caller's visibility."""
    key_filter = {"key": key}
    vis = _visibility_filter(user_api_key_dict)
    where = key_filter if vis is None else {"AND": [key_filter, vis]}
    rows = await prisma_client.db.litellm_memorytable.find_many(
        where=where, take=1, order={"updated_at": "desc"}
    )
    if not rows:
        raise HTTPException(
            status_code=404, detail=f"Memory with key '{key}' not found"
        )
    return rows[0]


@router.get(
    "/v1/memory/{key:path}",
    tags=["memory management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_MemoryRow,
)
async def get_memory(
    key: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Get a single memory entry by key, scoped to the caller."""
    prisma_client = _require_prisma()
    row = await _find_memory_for_caller(prisma_client, key, user_api_key_dict)
    return _row_to_model(row)


@router.put(
    "/v1/memory/{key:path}",
    tags=["memory management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=LiteLLM_MemoryRow,
)
async def upsert_memory(
    key: str,
    body: MemoryUpdateRequest,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """
    Upsert a memory entry by key within the caller's scope.

    If no row exists for (key, caller.user_id, caller.team_id), create one.
    If one exists, update the value/metadata fields that were provided.
    """
    prisma_client = _require_prisma()

    # Distinguish "metadata omitted from request" from "metadata: null".
    # Omitted → don't touch the existing field. Explicit null → clear to SQL NULL.
    # `model_fields_set` (Pydantic v2) only contains field names the caller
    # actually sent in the payload.
    fields_sent = body.model_fields_set
    metadata_explicit = "metadata" in fields_sent

    data: dict = {}
    if body.value is not None:
        data["value"] = body.value
    if metadata_explicit:
        # body.metadata may be None here — Prisma update accepts None on a
        # nullable Json? field and sets the column to SQL NULL.
        data["metadata"] = body.metadata
    if not data:
        raise HTTPException(
            status_code=400,
            detail="Request body must include at least one of: value, metadata.",
        )
    data["updated_by"] = user_api_key_dict.user_id

    async def _find_existing() -> Any:
        """Return the caller-visible row for `key`, or None."""
        try:
            return await _find_memory_for_caller(prisma_client, key, user_api_key_dict)
        except HTTPException as e:
            if e.status_code == 404:
                return None
            raise

    try:
        existing = await _find_existing()
        if existing is not None:
            # Visibility != write authority. Make sure the caller actually
            # owns this row (their user_id matches, or it's a pure team row in
            # their team) — otherwise a teammate could overwrite a personal
            # entry through the OR-based visibility filter.
            _assert_write_access(existing, user_api_key_dict)
            row = await prisma_client.db.litellm_memorytable.update(
                where={"memory_id": existing.memory_id},
                data=data,
            )
        else:
            if body.value is None:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot create a new memory via PUT without a 'value'.",
                )
            # PUT-create must honor admin scope override, matching POST semantics.
            user_id, team_id = _resolve_scope(
                user_api_key_dict, body.user_id, body.team_id
            )
            # Omit `metadata` when None — Prisma rejects None on Json? fields.
            create_data: dict = {
                "key": key,
                "value": body.value,
                "user_id": user_id,
                "team_id": team_id,
                "created_by": user_api_key_dict.user_id,
                "updated_by": user_api_key_dict.user_id,
            }
            if body.metadata is not None:
                create_data["metadata"] = body.metadata
            try:
                row = await prisma_client.db.litellm_memorytable.create(
                    data=create_data
                )
            except Exception as e:
                # Race: a concurrent PUT/POST created the row after our check.
                # Re-read and fall back to an update so the PUT stays idempotent
                # instead of surfacing a 500 on a unique-violation.
                if not _is_unique_violation(e):
                    raise
                existing_after_race = await _find_existing()
                if existing_after_race is None:
                    # Row exists globally but isn't visible to this caller
                    # (owned by someone else). Treat as conflict.
                    raise HTTPException(
                        status_code=409,
                        detail=f"Memory with key '{key}' already exists.",
                    )
                # Same write-authorization check as the non-race path.
                _assert_write_access(existing_after_race, user_api_key_dict)
                row = await prisma_client.db.litellm_memorytable.update(
                    where={"memory_id": existing_after_race.memory_id},
                    data=data,
                )
    except HTTPException:
        raise
    except Exception as e:
        raise _internal_error(
            "Error upserting memory: %s", e, "Internal error updating memory entry."
        )

    return _row_to_model(row)


@router.delete(
    "/v1/memory/{key:path}",
    tags=["memory management"],
    dependencies=[Depends(user_api_key_auth)],
    response_model=MemoryDeleteResponse,
)
async def delete_memory(
    key: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    """Delete a memory entry by key, scoped to the caller."""
    prisma_client = _require_prisma()
    row = await _find_memory_for_caller(prisma_client, key, user_api_key_dict)
    # Visibility != write authority — see the upsert handler for the rationale.
    _assert_write_access(row, user_api_key_dict)
    try:
        await prisma_client.db.litellm_memorytable.delete(
            where={"memory_id": row.memory_id}
        )
    except Exception as e:
        raise _internal_error(
            "Error deleting memory: %s", e, "Internal error deleting memory entry."
        )

    return MemoryDeleteResponse(key=key, deleted=True)
