"""Organization membership rows: the one place that knows which users hold which role in an organization."""

from typing import TYPE_CHECKING, Final

from litellm.proxy.auth.roles import Role
from litellm.repositories.table_repositories import PrismaTableRepository

if TYPE_CHECKING:
    from prisma import models as prisma_models  # noqa: F401  # resolved only from the quoted base-class subscript below


class OrganizationMembershipRepository(PrismaTableRepository["prisma_models.LiteLLM_OrganizationMembership"]):
    table_name = "litellm_organizationmembership"

    async def is_org_admin(self, user_id: str, organization_id: str) -> bool:
        membership: Final = await self.table.find_unique(
            where={"user_id_organization_id": {"user_id": user_id, "organization_id": organization_id}}
        )
        return membership is not None and membership.user_role == Role.ORG_ADMIN.value
