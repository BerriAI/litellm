"""Registration of a customer member the request named but the proxy has never seen.

Customers send a shared service key with ``x-litellm-customer-org-alias`` and ``x-litellm-customer-member-id``
on every call, and their users appear for the first time on their first request. The member is created under
the organization's ``default`` team, so it inherits the organization's default budget. An organization without
a default budget cannot host an unregistered member, because a member must have some budget to be charged to.
"""

from typing import TYPE_CHECKING, Final

from litellm._logging import verbose_proxy_logger
from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy, ErrorCode, Failure
from litellm.types.proxy.management_endpoints.customer_hierarchy import CustomerMember, MemberCreate, Team, TeamCreate

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

AUTO_CREATE_ACTOR: Final = "x-litellm-customer-member-id"
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
    hierarchy: Final = CustomerHierarchy(prisma_client)
    team: Final = await _default_team(hierarchy, customer_org_id)
    if isinstance(team, Failure):
        return team
    created: Final = await hierarchy.create_member(
        MemberCreate(customer_member_id=customer_member_id, parent_team_id=team.customer_team_id), AUTO_CREATE_ACTOR
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
