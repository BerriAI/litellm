from __future__ import annotations

from litellm.repositories.organization_membership_repository import OrganizationMembershipRepository


def get_org_roles() -> OrganizationMembershipRepository:
    from litellm.proxy.proxy_server import prisma_client

    return OrganizationMembershipRepository(prisma_client)
