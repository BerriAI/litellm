from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, TypeAdapter

from litellm.constants import UI_SESSION_TOKEN_TEAM_ID
from litellm.proxy.utils import PrismaClient


class _UserId(BaseModel):
    model_config = ConfigDict(frozen=True)
    user_id: str


_USER_IDS: Final = TypeAdapter(tuple[_UserId, ...])


async def sorted_user_ids(
    prisma_client: PrismaClient,
    *,
    sort_by: Literal["status", "key_count"],
    descending: bool,
    skip: int,
    take: int,
    role: str | None,
    user_ids: str | None,
    user_email: str | None,
    team: str | None,
    sso_user_ids: str | None,
    organization_ids: str | None,
    search: str | None,
) -> Sequence[str]:
    ids: Final = tuple(value.strip() for value in (user_ids or "").split(",") if value.strip())
    sso_ids: Final = tuple(value.strip() for value in (sso_user_ids or "").split(",") if value.strip())
    org_ids: Final = tuple(value.strip() for value in (organization_ids or "").split(",") if value.strip())
    expression: Final = (
        "CASE WHEN u.metadata->'scim_active' = 'false'::jsonb THEN 1 ELSE 0 END"
        if sort_by == "status"
        else '(SELECT COUNT(*) FROM "LiteLLM_VerificationToken" k '
        "WHERE k.user_id = u.user_id AND k.team_id IS DISTINCT FROM $11::text)"
    )
    direction: Final = "DESC" if descending else "ASC"
    sql: Final = f"""
        SELECT u.user_id FROM "LiteLLM_UserTable" u
        WHERE ($1::text IS NULL OR u.user_role = $1)
          AND ($2::text[] IS NULL
               OR CASE WHEN cardinality($2::text[]) = 1
                  THEN u.user_id ILIKE '%' || ($2::text[])[1] || '%'
                  ELSE u.user_id = ANY($2::text[]) END)
          AND ($3::text IS NULL OR u.user_email ILIKE '%' || $3 || '%')
          AND ($4::text IS NULL OR $4 = ANY(u.teams))
          AND ($5::boolean OR u.sso_user_id = ANY($6::text[]))
          AND (cardinality($7::text[]) = 0 OR EXISTS (
               SELECT 1 FROM "LiteLLM_OrganizationMembership" m
               WHERE m.user_id = u.user_id AND m.organization_id = ANY($7::text[])))
          AND ($8::text IS NULL OR u.user_id ILIKE '%' || $8 || '%'
               OR u.user_email ILIKE '%' || $8 || '%')
        ORDER BY {expression} {direction}, u.user_id ASC
        LIMIT $9 OFFSET $10
    """
    rows: Final = _USER_IDS.validate_python(
        await prisma_client.db.query_raw(
            sql,
            role or None,
            ids if user_ids else None,
            user_email,
            team,
            sso_user_ids is None,
            sso_ids,
            org_ids,
            search or None,
            take,
            skip,
            *((UI_SESSION_TOKEN_TEAM_ID,) if sort_by == "key_count" else ()),
        )
    )
    return tuple(row.user_id for row in rows)
