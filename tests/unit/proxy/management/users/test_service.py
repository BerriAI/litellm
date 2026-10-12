from __future__ import annotations

from datetime import datetime, timezone
from typing import Final

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.proxy._types import LiteLLM_OrganizationMembershipTable, LiteLLM_UserTable, LitellmUserRoles
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management.users.service import PrismaOrgRoles, holds_org_admin
from litellm.proxy.utils import ProxyLogging

NOW: Final = datetime.now(timezone.utc)


def user_in(*memberships: tuple[str, str]) -> LiteLLM_UserTable:
    return LiteLLM_UserTable(
        user_id="u1",
        organization_memberships=[
            LiteLLM_OrganizationMembershipTable(
                user_id="u1", organization_id=organization_id, user_role=role, created_at=NOW, updated_at=NOW
            )
            for organization_id, role in memberships
        ],
    )


@pytest.mark.parametrize(
    ("user", "expected"),
    [
        (user_in(("org-1", LitellmUserRoles.ORG_ADMIN.value)), True),
        (user_in(("org-2", LitellmUserRoles.ORG_ADMIN.value)), False),
        (user_in(("org-1", LitellmUserRoles.INTERNAL_USER.value)), False),
        (user_in(("org-2", LitellmUserRoles.ORG_ADMIN.value), ("org-1", LitellmUserRoles.ORG_ADMIN.value)), True),
        (user_in(), False),
        (LiteLLM_UserTable(user_id="u1", organization_memberships=None), False),
        (None, False),
    ],
)
def test_holds_org_admin_needs_the_org_admin_role_in_that_org(user: LiteLLM_UserTable | None, expected: bool) -> None:
    assert holds_org_admin(user, "org-1") is expected


@pytest.mark.parametrize(
    ("organization_id", "expected"),
    [("org-1", True), ("org-2", False)],
)
async def test_prisma_org_roles_answers_from_the_cached_user_row(organization_id: str, expected: bool) -> None:
    cache: Final = UserApiKeyCache()
    await cache.async_set_cache(key="u1", value=user_in(("org-1", LitellmUserRoles.ORG_ADMIN.value)))
    roles: Final = PrismaOrgRoles(None, cache, ProxyLogging(user_api_key_cache=DualCache()))
    assert await roles.is_org_admin("u1", organization_id) is expected
