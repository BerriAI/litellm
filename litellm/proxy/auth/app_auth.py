from collections.abc import Mapping, Sequence
from functools import partial
from typing import TYPE_CHECKING, Final

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.customer_hierarchy_auth import (
    CustomerBudgetExceededError,
    check_customer_hierarchy_access,
    customer_scope,
)
from litellm.proxy.auth.customer_spend import current_customer_spend
from litellm.proxy.management_helpers.apps import AppStore, app_scope_id
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerScopeAuth,
    CustomerScopeKey,
    app_budget_limits,
)
from litellm.types.proxy.management_endpoints.apps import AppFailure, AppResourceType
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router_api import RegistryReader


def selected_app_assertion(headers: Mapping[str, object]) -> str | None:
    return next(
        (value.strip() for key, value in headers.items() if key.lower() == "x-litellm-app" and isinstance(value, str)),
        None,
    )


def app_scope(
    key: CustomerScopeKey,
    entity_id: str,
    spend: float,
    budget: Budget | None,
) -> CustomerScopeAuth:
    has_duration: Final = budget is not None and budget.budget_duration is not None and budget.max_budget is not None
    return customer_scope(key, entity_id, spend, budget).model_copy(
        update={
            "budget_limits": app_budget_limits(budget),
            "soft_budget_duration": budget.budget_duration
            if budget is not None and budget.soft_budget is not None
            else None,
            **({"max_budget": None} if has_duration else {}),
        }
    )


def raise_app_failure(error: AppFailure) -> None:
    raise ProxyException(
        message=error.detail,
        type="auth_error",
        param="x-litellm-app",
        code=error.status,
    )


async def authenticated_resource_id(
    store: AppStore, resource_type: AppResourceType, authenticated: UserAPIKeyAuth
) -> str | None:
    match resource_type:
        case AppResourceType.TEAM:
            return authenticated.team_id
        case AppResourceType.PROJECT:
            return authenticated.project_id
        case AppResourceType.VIRTUAL_KEY:
            return authenticated.key_id or await store.key_resource_id(authenticated.token)


def authenticated_resource_alias(
    resource_type: AppResourceType, resource_id: str, authenticated: UserAPIKeyAuth
) -> str | None:
    match resource_type:
        case AppResourceType.TEAM:
            return (authenticated.team_alias or resource_id) if "team_alias" in authenticated.model_fields_set else None
        case AppResourceType.PROJECT:
            return (
                (authenticated.project_alias or resource_id)
                if "project_alias" in authenticated.model_fields_set
                else None
            )
        case AppResourceType.VIRTUAL_KEY:
            return (
                (authenticated.key_alias or resource_id)
                if authenticated.key_id is not None or "key_alias" in authenticated.model_fields_set
                else None
            )


async def resolve_app_hierarchy(
    hierarchy: CustomerHierarchyAuth | None,
    headers: Mapping[str, object],
    metadata: Mapping[str, object],
    prisma_client: "PrismaClient | None",
    llm_router: "RegistryReader | None",
    request_model: str | Sequence[str] | None,
    route: str,
    authenticated: UserAPIKeyAuth,
) -> CustomerHierarchyAuth | None:
    import litellm
    from litellm.proxy.proxy_server import get_current_spend, user_api_key_cache

    assertion: Final = selected_app_assertion(headers)
    if prisma_client is None:
        return hierarchy
    store: Final = AppStore(prisma_client.db, cache=user_api_key_cache)
    definition: Final = await store.definition()
    if definition.resource_type is None:
        return hierarchy
    resource_id: Final = await authenticated_resource_id(store, definition.resource_type, authenticated)
    if resource_id is None:
        if assertion is not None:
            raise_app_failure(
                AppFailure(
                    403,
                    "The authenticated identity has no app resource matching this header.",
                )
            )
        return hierarchy
    app: Final = await store.by_resource(
        definition.resource_type,
        resource_id,
        authenticated_resource_alias(definition.resource_type, resource_id, authenticated),
    )
    if app is None:
        if assertion is not None and assertion != resource_id:
            resource: Final = await store.resource(definition.resource_type, resource_id)
            if isinstance(resource, AppFailure) or assertion != resource.alias:
                raise_app_failure(AppFailure(403, "The app header does not match the authenticated resource."))
        return hierarchy
    if assertion is not None and assertion not in (
        app.app_id,
        app.resource_id,
        app.alias,
    ):
        raise_app_failure(AppFailure(403, "The app header does not match the authenticated resource."))
    org: Final = (
        await store.organization(app, hierarchy.customer_org_id)
        if hierarchy is not None and hierarchy.customer_org_id
        else None
    )
    member: Final = (
        await store.member(app, hierarchy.customer_member_id, org)
        if hierarchy is not None and hierarchy.customer_member_id and org is not None
        else None
    )
    app_scopes: Final = (
        app_scope("app", app.app_id, app.spend, app.budget),
        *(
            (
                app_scope(
                    "app_customer_org",
                    app_scope_id(app.app_id, org.customer_org_id),
                    org.spend,
                    org.effective_budget,
                ),
            )
            if org is not None
            else ()
        ),
        *(
            (
                app_scope(
                    "app_customer_member",
                    app_scope_id(app.app_id, member.customer_member_id),
                    member.spend,
                    member.effective_budget,
                ),
            )
            if member is not None
            else ()
        ),
    )
    app_hierarchy: Final = await current_customer_spend(
        CustomerHierarchyAuth(scopes=app_scopes),
        partial(get_current_spend, recheck_recorded_spend=True),
    )
    base: Final = hierarchy or CustomerHierarchyAuth(scopes=())
    combined: Final = base.model_copy(
        update={
            "app_id": app.app_id,
            "app_alias": app.alias,
            "app_customer_margin": app.customer_margin,
            "scopes": (*base.scopes, *app_hierarchy.scopes),
        }
    )
    try:
        await check_customer_hierarchy_access(app_hierarchy, llm_router, request_model, route, metadata)
    except litellm.BudgetExceededError as error:
        raise CustomerBudgetExceededError(combined, error) from error
    return combined
