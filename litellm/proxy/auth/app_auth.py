from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

from litellm.proxy._types import ProxyException
from litellm.proxy.auth.customer_hierarchy_auth import CustomerBudgetExceededError, check_customer_hierarchy_access
from litellm.proxy.auth.customer_spend import current_customer_spend
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_helpers.apps import AppStore, app_scope_id
from litellm.types.proxy.auth.customer_hierarchy import (
    AppDurationLimit,
    CustomerHierarchyAuth,
    CustomerScopeAuth,
    CustomerScopeKey,
)
from litellm.types.proxy.management_endpoints.apps import AppFailure
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.router_api import RegistryReader


def app_alias_from_request(headers: Mapping[str, object], metadata: Mapping[str, object]) -> str | AppFailure | None:
    bound: Final = metadata.get("app_alias")
    selected: Final = next(
        (
            value.strip()
            for key, value in headers.items()
            if key.lower() == "x-litellm-app" and isinstance(value, str) and value.strip()
        ),
        None,
    )
    if isinstance(bound, str) and bound:
        if selected is not None and bound != selected:
            return AppFailure(403, "This virtual key is bound to a different app.")
        return bound
    return selected


def app_scope(
    key: CustomerScopeKey,
    entity_id: str,
    spend: float,
    budget: Budget | None,
) -> CustomerScopeAuth:
    duration: Final = (
        (AppDurationLimit(budget_duration=budget.budget_duration, max_budget=budget.max_budget),)
        if budget is not None and budget.budget_duration is not None and budget.max_budget is not None
        else ()
    )
    return CustomerScopeAuth.model_validate(
        {
            "key": key,
            "entity_id": entity_id,
            "spend": spend,
            **(budget.model_dump(exclude={"soft_budget", "budget_duration"}) if budget is not None else {}),
            "budget_limits": (budget.budget_limits or ()) if budget is not None else (),
            **(
                {"max_budget": None, "budget_limits": (*(budget.budget_limits or ()), *duration)}
                if budget is not None and duration
                else {}
            ),
        }
    )


def raise_app_failure(error: AppFailure) -> None:
    raise ProxyException(message=error.detail, type="auth_error", param="x-litellm-app", code=error.status)


async def resolve_app_hierarchy(
    hierarchy: CustomerHierarchyAuth | None,
    headers: Mapping[str, object],
    metadata: Mapping[str, object],
    prisma_client: "PrismaClient | None",
    cache: UserApiKeyCache,
    llm_router: "RegistryReader | None",
    request_model: str | Sequence[str] | None,
    route: str,
) -> CustomerHierarchyAuth | None:
    import litellm
    from litellm.proxy.proxy_server import get_current_spend

    alias: Final = app_alias_from_request(headers, metadata)
    if isinstance(alias, AppFailure):
        raise_app_failure(alias)
        return hierarchy
    if alias is None:
        return hierarchy
    if prisma_client is None:
        raise_app_failure(AppFailure(503, "Database unavailable."))
        return hierarchy
    store: Final = AppStore(prisma_client.db, cache=cache)
    app: Final = await store.by_alias(alias)
    if isinstance(app, AppFailure):
        raise_app_failure(app)
        return hierarchy
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
                    "app_customer_org", app_scope_id(app.app_id, org.customer_org_id), org.spend, org.effective_budget
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
    base: Final = hierarchy or CustomerHierarchyAuth(scopes=())
    combined: Final = base.model_copy(
        update={
            "app_id": app.app_id,
            "app_alias": app.alias,
            "scopes": (*base.scopes, *app_scopes),
        }
    )
    current: Final = await current_customer_spend(combined, get_current_spend)
    try:
        await check_customer_hierarchy_access(current, llm_router, request_model, route, metadata)
    except litellm.BudgetExceededError as error:
        raise CustomerBudgetExceededError(current, error) from error
    return current
