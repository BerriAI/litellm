import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from functools import partial
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel

from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_batch import Batch
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.customer_hierarchy_auth import (
    CustomerBudgetExceededError,
    check_customer_hierarchy_access,
    customer_scope,
)
from litellm.proxy.auth.customer_spend import current_customer_spend
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler
from litellm.proxy.management_helpers.apps import APP_AUTH_POLICY_KEY, AppStore, app_scope_id
from litellm.types.proxy.auth.customer_hierarchy import (
    CustomerHierarchyAuth,
    CustomerScopeAuth,
    CustomerScopeKey,
    app_budget_limits,
)
from litellm.types.proxy.management_endpoints.apps import App, AppFailure, AppMember, AppOrganization, AppResourceType
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient
    from litellm.redis.stores.auth_cache import AuthCacheStore
    from litellm.router_api import RegistryReader

_APP_AUTH_SNAPSHOT_TTL: Final = 60.0


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


class _AppAuthRecords(BaseModel, frozen=True):
    app: App
    organization: AppOrganization | None = None
    member: AppMember | None = None


async def _load_app_records(
    store: AppStore,
    resource_type: AppResourceType,
    resource_id: str,
    assertion: str | None,
    hierarchy: CustomerHierarchyAuth | None,
    *,
    resource_alias: str | None = None,
) -> _AppAuthRecords | None:
    app: Final = await store.by_resource(resource_type, resource_id, resource_alias)
    if app is None:
        if assertion is not None and assertion != resource_id:
            resource: Final = await store.resource(resource_type, resource_id)
            if isinstance(resource, AppFailure) or assertion != resource.alias:
                raise_app_failure(AppFailure(403, "The app header does not match the authenticated resource."))
        return None
    _check_app_assertion(app, assertion)
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
    return _AppAuthRecords(app=app, organization=org, member=member)


def _check_app_assertion(app: App, assertion: str | None) -> None:
    if assertion is not None and assertion not in (app.app_id, app.resource_id, app.alias):
        raise_app_failure(AppFailure(403, "The app header does not match the authenticated resource."))


def _app_records_key(
    generation: str, resource_type: AppResourceType, resource_id: str, org_id: str, member_id: str | None
) -> str:
    identity: Final = json.dumps((generation, resource_type.value, resource_id, org_id, member_id))
    return "customer_app_records:" + hashlib.sha256(identity.encode()).hexdigest()


def _snapshot_ttl(records: _AppAuthRecords, read_at: datetime, now: datetime) -> float:
    return min(
        (
            _APP_AUTH_SNAPSHOT_TTL,
            *(
                (record.budget_override_expires_at - now).total_seconds()
                for record in (records.organization, records.member)
                if record is not None
                and record.budget_override_expires_at is not None
                and record.budget_override_expires_at > read_at
            ),
        )
    )


async def _read_app_records(
    store: AppStore,
    resource_type: AppResourceType,
    resource_id: str,
    assertion: str | None,
    hierarchy: CustomerHierarchyAuth | None,
    cache: "AuthCacheStore | None",
    authenticated: UserAPIKeyAuth,
) -> _AppAuthRecords | None:
    resource_alias: Final = authenticated_resource_alias(resource_type, resource_id, authenticated)
    if cache is None or hierarchy is None or hierarchy.customer_org_id is None:
        return await _load_app_records(
            store, resource_type, resource_id, assertion, hierarchy, resource_alias=resource_alias
        )

    # An outage snapshot must not renew the lifetime of previously cached policy.
    database_store: Final = AppStore(store.db, now=store.now, cache_reads=False)
    generation: Final = await _app_policy_generation(cache)
    if generation is None:
        return await _load_app_records(database_store, resource_type, resource_id, assertion, hierarchy)
    key: Final = _app_records_key(
        generation, resource_type, resource_id, hierarchy.customer_org_id, hierarchy.customer_member_id
    )
    try:
        records: Final = await _load_app_records(database_store, resource_type, resource_id, assertion, hierarchy)
    except Exception as error:
        fault: Final = PrismaDBExceptionHandler.find_database_service_unavailable_error_in_chain(error)
        if fault is None:
            raise
        if (
            PrismaDBExceptionHandler.is_permanent_database_fault(fault)
            or not PrismaDBExceptionHandler.should_allow_request_on_db_unavailable()
        ):
            raise PrismaDBExceptionHandler.service_unavailable_proxy_exception(error) from error
        cached: Final = await _recover_app_records(cache, key, generation)
        if cached is None:
            raise PrismaDBExceptionHandler.service_unavailable_proxy_exception(error) from error
        _check_app_assertion(cached.app, assertion)
        authenticated.customer_auth_db_unavailable = True  # rebind-ok: request-local auth state
        return cached
    if records is not None:
        now: Final = datetime.now(timezone.utc)
        org_key: Final = _app_records_key(generation, resource_type, resource_id, hierarchy.customer_org_id, None)
        snapshots: Final = (
            (key, records),
            *(
                ((org_key, _AppAuthRecords(app=records.app, organization=records.organization)),)
                if org_key != key
                else ()
            ),
        )
        try:
            await cache.write_back(
                tuple(
                    (snapshot_key, snapshot.model_dump(mode="json"), ttl)
                    for snapshot_key, snapshot in snapshots
                    if (ttl := _snapshot_ttl(snapshot, store.now, now)) >= 1
                ),
                batch=Batch.NONE,
            )
        except Exception as error:  # noqa: BLE001  # a Redis write failure must not reject a verified DB result
            verbose_proxy_logger.warning("Unable to cache customer app records: %s", type(error).__name__)
    return records


async def _app_policy_generation(cache: "AuthCacheStore") -> str | None:
    generation: Final = await cache.lease(APP_AUTH_POLICY_KEY)
    if generation is not None:
        return generation
    try:
        await cache.delete((APP_AUTH_POLICY_KEY,))
    except Exception as error:  # noqa: BLE001  # without a confirmed generation, use the database only
        verbose_proxy_logger.warning("Unable to initialize customer app policy generation: %s", type(error).__name__)
        return None
    return await cache.lease(APP_AUTH_POLICY_KEY)


async def _recover_app_records(cache: "AuthCacheStore", key: str, generation: str) -> _AppAuthRecords | None:
    try:
        payload: Final = (await cache.read_missing((key,), batch=Batch.NONE)).get(key)
        if payload is None or await cache.lease(APP_AUTH_POLICY_KEY) != generation:
            return None
        return _AppAuthRecords.model_validate(payload)
    except Exception as error:  # noqa: BLE001  # Redis or invalid cached policy must be treated as a miss
        verbose_proxy_logger.warning("Unable to read customer app outage cache: %s", type(error).__name__)
        return None


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
    records: Final = await _read_app_records(
        store,
        definition.resource_type,
        resource_id,
        assertion,
        hierarchy,
        user_api_key_cache.customer_auth_store
        if PrismaDBExceptionHandler.should_allow_request_on_db_unavailable()
        else None,
        authenticated,
    )
    if records is None:
        return hierarchy
    app, org, member = records.app, records.organization, records.member
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
    base: Final = hierarchy or CustomerHierarchyAuth(scopes=())
    checked_scopes: Final = (*base.scopes, *app_scopes) if authenticated.customer_auth_db_unavailable else app_scopes
    app_hierarchy: Final = await current_customer_spend(
        CustomerHierarchyAuth(scopes=checked_scopes),
        partial(get_current_spend, require_verified_spend=True)
        if authenticated.customer_auth_db_unavailable
        else partial(get_current_spend, recheck_recorded_spend=True),
    )
    combined: Final = base.model_copy(
        update={
            "app_id": app.app_id,
            "app_alias": app.alias,
            "app_customer_margin": app.customer_margin,
            "scopes": app_hierarchy.scopes
            if authenticated.customer_auth_db_unavailable
            else (*base.scopes, *app_hierarchy.scopes),
        }
    )
    try:
        await check_customer_hierarchy_access(app_hierarchy, llm_router, request_model, route, metadata)
    except litellm.BudgetExceededError as error:
        raise CustomerBudgetExceededError(combined, error) from error
    return combined
