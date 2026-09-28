from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.proxy._types import LiteLLM_UserTable, LitellmUserRoles

if TYPE_CHECKING:
    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
    from litellm.proxy.utils import PrismaClient, ProxyLogging


def holds_org_admin(user: LiteLLM_UserTable | None, organization_id: str) -> bool:
    return user is not None and any(
        membership.organization_id == organization_id and membership.user_role == LitellmUserRoles.ORG_ADMIN.value
        for membership in user.organization_memberships or []
    )


@dataclass(frozen=True, slots=True)
class PrismaOrgRoles:
    prisma_client: PrismaClient | None
    user_api_key_cache: UserApiKeyCache
    proxy_logging_obj: ProxyLogging

    async def is_org_admin(self, user_id: str, organization_id: str) -> bool:
        from litellm.proxy.auth.auth_checks import get_user_object

        user: Final = await get_user_object(
            user_id=user_id,
            prisma_client=self.prisma_client,
            user_api_key_cache=self.user_api_key_cache,
            user_id_upsert=False,
            proxy_logging_obj=self.proxy_logging_obj,
        )
        return holds_org_admin(user, organization_id)
