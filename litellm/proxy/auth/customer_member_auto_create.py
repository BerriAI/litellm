"""Registration of a customer member the request named but the proxy has never seen.

Customers send a shared service key with ``x-litellm-customer-org-alias`` and ``x-litellm-customer-member-id``
on every call, and their users appear for the first time on their first request. The member is created under
the organization's ``default`` team, so it inherits the organization's default budget. In an organization without
a default budget the member is registered with no budget: its spend is tracked and nothing caps it until an admin
assigns one.
"""

from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy, ErrorCode, Failure
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    CustomerMember,
    MemberCreate,
    Organization,
    OrganizationCreate,
    Team,
    TeamCreate,
)

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

AUTO_CREATE_ACTOR: Final = "x-litellm-customer-member-id"
AUTO_CREATE_ORG_ACTOR: Final = "x-litellm-customer-org-alias"
DEFAULT_TEAM_ALIAS: Final = "default"


def default_team_id(customer_org_id: str) -> str:
    return f"{customer_org_id}:{DEFAULT_TEAM_ALIAS}"


async def _default_team(hierarchy: CustomerHierarchy, customer_org_id: str) -> Team | Failure:
    team_id: Final = default_team_id(customer_org_id)
    existing: Final = await hierarchy.get_team(team_id)
    if not isinstance(existing, Failure):
        return existing
    created: Final = await hierarchy.create_team(
        TeamCreate(customer_team_id=team_id, alias=DEFAULT_TEAM_ALIAS, parent_org_id=customer_org_id),
        AUTO_CREATE_ACTOR,
    )
    if isinstance(created, Failure) and created.code is ErrorCode.ALREADY_EXISTS:
        return await hierarchy.get_team(team_id)
    return created


async def ensure_customer_member(
    customer_org_id: str, customer_member_id: str, prisma_client: "PrismaClient"
) -> CustomerMember | Failure:
    """The member row for ``customer_member_id`` in ``customer_org_id``'s default team, created when absent.
    Two workers registering the same member at once both end up with the one row."""
    from litellm.proxy.proxy_server import premium_user

    hierarchy: Final = CustomerHierarchy(prisma_client, premium_user=premium_user)
    team: Final = await _default_team(hierarchy, customer_org_id)
    if isinstance(team, Failure):
        return team
    created: Final = await hierarchy.create_member(
        MemberCreate(customer_member_id=customer_member_id, parent_team_ids=(team.customer_team_id,)),
        AUTO_CREATE_ACTOR,
        budget_required=False,
    )
    if isinstance(created, Failure) and created.code is ErrorCode.ALREADY_EXISTS:
        return await hierarchy.get_member(customer_member_id)
    if not isinstance(created, Failure):
        verbose_proxy_logger.info(
            "Registered customer member %s in team %s of organization %s",
            customer_member_id,
            team.customer_team_id,
            customer_org_id,
        )
    return created


async def ensure_customer_organization(
    customer_org_alias: str, prisma_client: "PrismaClient"
) -> Organization | Failure:
    """The organization row carrying ``customer_org_alias``, created with the alias as its id when absent, so a
    customer's first request is charged to it. Two workers registering the same alias at once both end up with
    the one row."""
    from litellm.proxy.proxy_server import premium_user

    hierarchy: Final = CustomerHierarchy(prisma_client, premium_user=premium_user)
    created: Final = await hierarchy.create_organization(
        OrganizationCreate(customer_org_id=customer_org_alias, alias=customer_org_alias), AUTO_CREATE_ORG_ACTOR
    )
    if isinstance(created, Failure) and created.code in (ErrorCode.ALREADY_EXISTS, ErrorCode.ALIAS_ALREADY_EXISTS):
        return await hierarchy.get_organization(customer_org_alias)
    if not isinstance(created, Failure):
        verbose_proxy_logger.info("Registered customer organization %s", customer_org_alias)
    return created
