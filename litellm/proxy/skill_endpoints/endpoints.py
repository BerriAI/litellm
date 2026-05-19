"""
xct-native skills CRUD — ``/v1/xct-skills`` (S2-03).

Coexists with the Anthropic skill pass-through (``/v1/skills``) in the same
``LiteLLM_SkillsTable``; rows are distinguished by ``source``:
- ``source='anthropic'``  → Anthropic SDK pass-through (legacy).
- ``source='custom'``     → xct-native, the rows this module reads/writes.

See ADR-0001 (S2-01) for the design rationale.

S2-03 scope: pure CRUD. ZIP upload pipeline arrives in S2-07; publish/version
semantics in S2-06; chat completion injection in S2-05; capability discovery
wire-up in S2-04.
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm._logging import verbose_proxy_logger
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.types.xct_skills import (
    XCTSkill,
    XCTSkillCreate,
    XCTSkillListResponse,
    XCTSkillPatch,
)

router = APIRouter()

# Reserve source='custom' for xct-native rows (ADR-0001 stays with this enum
# value rather than introducing 'xct' to avoid a backfill migration).
_XCT_SOURCE = "custom"


def _is_admin(uak: UserAPIKeyAuth) -> bool:
    return uak.user_role in (
        LitellmUserRoles.PROXY_ADMIN,
        LitellmUserRoles.PROXY_ADMIN.value,
    )


def _row_to_skill(row) -> XCTSkill:
    """Map a Prisma row (or dict) to the public XCTSkill response."""
    if hasattr(row, "model_dump"):
        data = row.model_dump()
    elif isinstance(row, dict):
        data = row
    else:
        data = vars(row)
    return XCTSkill(
        skill_id=data["skill_id"],
        display_title=data.get("display_title"),
        description=data.get("description"),
        instructions=data.get("instructions"),
        system_prompt_template=data.get("system_prompt_template"),
        tool_schema=data.get("tool_schema"),
        source=data.get("source", _XCT_SOURCE),
        version=data.get("version"),
        is_public=bool(data.get("is_public", False)),
        team_id=data.get("team_id"),
        user_id=data.get("user_id"),
        created_by=data.get("created_by"),
        created_at=data.get("created_at"),
        updated_at=data.get("updated_at"),
        xct_metadata=data.get("xct_metadata") or {},
    )


async def _require_writable(skill_id: str, uak: UserAPIKeyAuth):
    """Raise 404 / 403 / return row if caller may mutate this skill."""
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="DB not initialized")
    row = await prisma_client.db.litellm_skillstable.find_unique(
        where={"skill_id": skill_id}
    )
    if row is None or getattr(row, "source", None) != _XCT_SOURCE:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    if _is_admin(uak):
        return row
    if uak.user_id and getattr(row, "user_id", None) == uak.user_id:
        return row
    raise HTTPException(
        status_code=403,
        detail="Only the skill owner or a proxy admin may modify this skill.",
    )


@router.post(
    "/v1/xct-skills",
    tags=["[beta] XCT Skills"],
    response_model=XCTSkill,
)
async def create_skill(
    payload: XCTSkillCreate,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> XCTSkill:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="DB not initialized")

    create_data: dict = {
        "display_title": payload.display_title,
        "description": payload.description,
        "instructions": payload.instructions,
        "system_prompt_template": payload.system_prompt_template,
        "tool_schema": payload.tool_schema,
        "source": _XCT_SOURCE,
        "version": payload.version or "1",
        "is_public": payload.is_public,
        "team_id": payload.team_id or user_api_key_dict.team_id,
        "user_id": user_api_key_dict.user_id,
        "xct_metadata": payload.xct_metadata or {},
        "created_by": user_api_key_dict.user_id,
    }
    row = await prisma_client.db.litellm_skillstable.create(data=create_data)
    return _row_to_skill(row)


@router.get(
    "/v1/xct-skills",
    tags=["[beta] XCT Skills"],
    response_model=XCTSkillListResponse,
)
async def list_skills(
    q: Optional[str] = Query(None, description="Match display_title/description."),
    team_id: Optional[str] = None,
    cursor: Optional[str] = Query(
        None, description="skill_id of the previous page tail."
    ),
    limit: int = Query(50, ge=1, le=200),
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> XCTSkillListResponse:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        return XCTSkillListResponse(data=[], has_more=False)

    # Build a single where clause; push every filter to the DB.
    where: dict = {"source": _XCT_SOURCE}
    if team_id is not None:
        where["team_id"] = team_id
    if q:
        # Postgres ILIKE wrapped by Prisma's `contains` mode.
        where["OR"] = [
            {"display_title": {"contains": q, "mode": "insensitive"}},
            {"description": {"contains": q, "mode": "insensitive"}},
        ]

    # Non-admin scoping: caller must own, share team, or skill is public.
    if not _is_admin(user_api_key_dict):
        caller_clauses = [{"is_public": True}]
        if user_api_key_dict.user_id:
            caller_clauses.append({"user_id": user_api_key_dict.user_id})
        if user_api_key_dict.team_id:
            caller_clauses.append({"team_id": user_api_key_dict.team_id})
        # Combine with the source filter via AND.
        where = {"AND": [where, {"OR": caller_clauses}]}

    find_args: dict = {
        "where": where,
        "take": limit + 1,
        "order": {"skill_id": "asc"},
    }
    if cursor is not None:
        find_args["cursor"] = {"skill_id": cursor}
        find_args["skip"] = 1

    rows = await prisma_client.db.litellm_skillstable.find_many(**find_args)
    has_more = len(rows) > limit
    rows = rows[:limit]
    next_cursor = rows[-1].skill_id if has_more and rows else None
    return XCTSkillListResponse(
        data=[_row_to_skill(r) for r in rows],
        has_more=has_more,
        next_cursor=next_cursor,
    )


@router.get(
    "/v1/xct-skills/{skill_id}",
    tags=["[beta] XCT Skills"],
    response_model=XCTSkill,
)
async def get_skill(
    skill_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> XCTSkill:
    from litellm.proxy.proxy_server import prisma_client

    if prisma_client is None:
        raise HTTPException(status_code=503, detail="DB not initialized")
    row = await prisma_client.db.litellm_skillstable.find_unique(
        where={"skill_id": skill_id}
    )
    if row is None or getattr(row, "source", None) != _XCT_SOURCE:
        raise HTTPException(status_code=404, detail=f"Skill '{skill_id}' not found")
    # Read scoping: public OR owned OR same team OR admin.
    if not _is_admin(user_api_key_dict):
        if not (
            getattr(row, "is_public", False)
            or getattr(row, "user_id", None) == user_api_key_dict.user_id
            or (
                user_api_key_dict.team_id
                and getattr(row, "team_id", None) == user_api_key_dict.team_id
            )
        ):
            raise HTTPException(
                status_code=403,
                detail=f"Skill '{skill_id}' is not accessible.",
            )
    return _row_to_skill(row)


@router.patch(
    "/v1/xct-skills/{skill_id}",
    tags=["[beta] XCT Skills"],
    response_model=XCTSkill,
)
async def patch_skill(
    skill_id: str,
    patch: XCTSkillPatch,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> XCTSkill:
    from litellm.proxy.proxy_server import prisma_client

    await _require_writable(skill_id, user_api_key_dict)
    update_data = {k: v for k, v in patch.model_dump().items() if v is not None}
    if not update_data:
        # Re-fetch + return; no-op patches shouldn't error.
        return await get_skill(skill_id, user_api_key_dict)
    update_data["updated_by"] = user_api_key_dict.user_id
    row = await prisma_client.db.litellm_skillstable.update(
        where={"skill_id": skill_id},
        data=update_data,
    )
    return _row_to_skill(row)


@router.delete(
    "/v1/xct-skills/{skill_id}",
    tags=["[beta] XCT Skills"],
)
async def delete_skill(
    skill_id: str,
    user_api_key_dict: UserAPIKeyAuth = Depends(user_api_key_auth),
) -> dict:
    from litellm.proxy.proxy_server import prisma_client

    await _require_writable(skill_id, user_api_key_dict)
    try:
        await prisma_client.db.litellm_skillstable.delete(where={"skill_id": skill_id})
    except Exception as e:  # pragma: no cover — wrap unexpected DB errors
        verbose_proxy_logger.exception("delete_skill failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
    return {"skill_id": skill_id, "deleted": True}
