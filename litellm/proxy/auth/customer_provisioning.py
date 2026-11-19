"""Whether a request may register a customer organization or member the proxy has never seen.

Each app carries the answer for its own requests, and one gateway setting answers for a request that belongs to no
app. A request that may not register runs for the customers the proxy already knows and is refused for any other.
"""

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Final, TypeAlias

from fastapi import status

from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.app_auth import authenticated_resource_id
from litellm.proxy.auth.customer_hierarchy_auth import (
    CUSTOMER_MEMBER_ID_HEADER,
    CUSTOMER_ORG_ALIAS_HEADER,
    CustomerBudgetExceededError,
    MemberRegistrar,
    OrganizationRegistrar,
)
from litellm.proxy.auth.customer_member_auto_create import ensure_customer_member, ensure_customer_organization
from litellm.proxy.management_helpers.apps import AppStore
from litellm.proxy.management_helpers.customer_hierarchy import ErrorCode, Failure
from litellm.types.proxy.auth.customer_hierarchy import CustomerHierarchyAuth, CustomerRequestIdentity
from litellm.types.proxy.management_endpoints.customer_hierarchy import CustomerMember, Organization

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient

RegistrationRefusal: TypeAlias = Callable[[], Awaitable[str | None]]
HierarchyResolver: TypeAlias = Callable[
    [MemberRegistrar, OrganizationRegistrar], Awaitable[CustomerHierarchyAuth | None]
]


async def registration_refusal(prisma_client: "PrismaClient | None", authenticated: UserAPIKeyAuth) -> str | None:
    """Why this request may not register a new customer, or None when it may."""
    if prisma_client is None:
        return None
    from litellm.proxy.proxy_server import user_api_key_cache

    store: Final = AppStore(prisma_client.db, cache=user_api_key_cache)
    definition: Final = await store.definition()
    resource_id: Final = (
        await authenticated_resource_id(store, definition.resource_type, authenticated)
        if definition.resource_type is not None
        else None
    )
    app: Final = (
        await store.by_resource(definition.resource_type, resource_id)
        if definition.resource_type is not None and resource_id is not None
        else None
    )
    if app is None:
        return (
            None
            if definition.auto_create_customers_without_app
            else "Requests that belong to no app may not create customers."
        )
    return None if app.auto_create_customers else f"App {app.alias} may not create customers."


def gated_registrars(refusal: RegistrationRefusal) -> tuple[MemberRegistrar, OrganizationRegistrar]:
    async def member(
        customer_org_id: str, customer_member_id: str, prisma_client: "PrismaClient"
    ) -> CustomerMember | Failure:
        if await refusal() is not None:
            return Failure(ErrorCode.NOT_FOUND)
        return await ensure_customer_member(customer_org_id, customer_member_id, prisma_client)

    async def organization(customer_org_alias: str, prisma_client: "PrismaClient") -> Organization | Failure:
        if await refusal() is not None:
            return Failure(ErrorCode.NOT_FOUND)
        return await ensure_customer_organization(customer_org_alias, prisma_client)

    return member, organization


def unknown_customer_header(identity: CustomerRequestIdentity, hierarchy: CustomerHierarchyAuth | None) -> str | None:
    """The header naming a customer the resolved hierarchy does not carry, or None when every named one is known."""
    if identity.customer_org_alias is not None and (
        hierarchy is None or hierarchy.customer_org_alias != identity.customer_org_alias
    ):
        return CUSTOMER_ORG_ALIAS_HEADER
    if identity.customer_member_id is not None and (hierarchy is None or hierarchy.customer_member_id is None):
        return CUSTOMER_MEMBER_ID_HEADER
    return None


async def refuse_unknown_customer(
    identity: CustomerRequestIdentity, hierarchy: CustomerHierarchyAuth | None, refusal: RegistrationRefusal
) -> None:
    header: Final = unknown_customer_header(identity, hierarchy)
    if header is None:
        return
    reason: Final = await refusal()
    if reason is None:
        return
    raise ProxyException(
        message=f"Unknown customer in {header}. {reason}",
        type=ProxyErrorTypes.auth_error,
        param=header,
        code=status.HTTP_403_FORBIDDEN,
    )


async def resolve_registered_customer(
    identity: CustomerRequestIdentity, refusal: RegistrationRefusal, resolve: HierarchyResolver
) -> CustomerHierarchyAuth | None:
    """The hierarchy ``resolve`` loads, refusing a customer this request names but may not register."""
    register_member, register_organization = gated_registrars(refusal)
    try:
        hierarchy: Final = await resolve(register_member, register_organization)
    except CustomerBudgetExceededError as error:
        await refuse_unknown_customer(identity, error.customer_hierarchy, refusal)
        raise
    await refuse_unknown_customer(identity, hierarchy, refusal)
    return hierarchy
