"""Who may act on a team: the one place management routes read team roster and organization membership.

``resolve_team_access`` ranks the caller over a team (proxy admin, org admin, team admin, or none); the
``require_*`` helpers turn "none" into the 403 the management routes have always raised.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Literal, NoReturn, TypeAlias

from fastapi import HTTPException, status

from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, UserAPIKeyAuth
from litellm.repositories.verification_token_repository import VerificationTokenRepository

if TYPE_CHECKING:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient

TeamAccessRole: TypeAlias = Literal["proxy_admin", "org_admin", "team_admin"]


def is_team_admin(user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable) -> bool:
    for member in team_obj.members_with_roles:
        if (member.user_id is not None and member.user_id == user_api_key_dict.user_id) and member.role == "admin":
            return True

    return False


async def is_org_admin_for_team(user_api_key_dict: UserAPIKeyAuth, team_obj: LiteLLM_TeamTable) -> bool:
    """True when the team belongs to an organization the caller holds ``org_admin`` in."""
    if not team_obj.organization_id or not user_api_key_dict.user_id:
        return False

    from litellm.proxy.auth.auth_checks import get_user_object
    from litellm.proxy.proxy_server import (
        prisma_client,
        proxy_logging_obj,
        user_api_key_cache,
    )

    caller_user: Final = await get_user_object(
        user_id=user_api_key_dict.user_id,
        prisma_client=prisma_client,
        user_api_key_cache=user_api_key_cache,
        user_id_upsert=False,
        proxy_logging_obj=proxy_logging_obj,
    )
    if caller_user is None:
        return False

    for m in caller_user.organization_memberships or []:
        if m.organization_id == team_obj.organization_id and m.user_role == LitellmUserRoles.ORG_ADMIN.value:
            return True

    return False


async def resolve_team_access(
    team_obj: LiteLLM_TeamTable,
    user_api_key_dict: UserAPIKeyAuth,
) -> TeamAccessRole | None:
    """Strongest role the caller holds over ``team_obj``, or None when they hold none.

    Org admin outranks team admin so a caller holding both keeps unrestricted edits.
    """
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        return "proxy_admin"

    if await is_org_admin_for_team(user_api_key_dict=user_api_key_dict, team_obj=team_obj):
        return "org_admin"

    if is_team_admin(user_api_key_dict=user_api_key_dict, team_obj=team_obj):
        return "team_admin"

    return None


def team_access_denied() -> NoReturn:
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="You do not have access to this team",
    )


async def require_team_access(
    team_obj: LiteLLM_TeamTable,
    user_api_key_dict: UserAPIKeyAuth,
) -> None:
    """Raise 403 unless the caller is a proxy admin, an org admin for the team's org, or a team admin."""
    if await resolve_team_access(team_obj=team_obj, user_api_key_dict=user_api_key_dict) is None:
        team_access_denied()


async def require_key_access(
    user_api_key_dict: UserAPIKeyAuth,
    hashed_token: str | None,
    prisma_client: PrismaClient | None,
    user_api_key_cache: UserApiKeyCache,
    route: str,
) -> None:
    """Raise unless the caller is a proxy admin, a team admin of the key's team, or an org admin for its org.

    404 when the key does not exist, 403 otherwise; a proxy admin passes without the key being looked up. Team
    admin is checked before org admin so a team admin whose user row is gone still passes, as it always has.
    """
    if user_api_key_dict.user_role == LitellmUserRoles.PROXY_ADMIN:
        return

    from litellm.proxy.auth.auth_checks import get_team_object

    target_key_row: Final = await VerificationTokenRepository(prisma_client).table.find_unique(
        where={"token": hashed_token}
    )
    if target_key_row is None:
        raise HTTPException(
            status_code=404,
            detail={"error": f"Key not found: {hashed_token}"},
        )

    if target_key_row.team_id:
        team_obj: Final = await get_team_object(
            team_id=target_key_row.team_id,
            prisma_client=prisma_client,
            user_api_key_cache=user_api_key_cache,
            check_db_only=True,
        )
        if team_obj is not None and (
            is_team_admin(user_api_key_dict=user_api_key_dict, team_obj=team_obj)
            or await is_org_admin_for_team(user_api_key_dict=user_api_key_dict, team_obj=team_obj)
        ):
            return

    raise HTTPException(
        status_code=403,
        detail={
            "error": f"Only proxy admins, team admins, or org admins can call {route}. "
            f"user_role={user_api_key_dict.user_role}, user_id={user_api_key_dict.user_id}"
        },
    )
