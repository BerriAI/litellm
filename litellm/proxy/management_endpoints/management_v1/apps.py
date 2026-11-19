import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Final, TypeAlias, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query

from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import (
    customer_actor,
    require_customer_admin,
)
from litellm.proxy.management_helpers.apps import AppStore
from litellm.types.proxy.management_endpoints.apps import (
    App,
    AppCreate,
    AppDefinition,
    AppFailure,
    AppMember,
    AppMemberPut,
    AppOrganization,
    AppOrganizationPut,
    AppPatch,
    AppResource,
    AppResourceType,
)

T = TypeVar("T")  # rebind-ok: TypeVar declarations cannot be annotated Final


def resolve(result: T | AppFailure) -> T:
    if isinstance(result, AppFailure):
        raise HTTPException(result.status, result.detail)
    return result


async def get_app_store() -> AsyncIterator[AppStore]:
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        raise HTTPException(503, "Database unavailable.")
    async with prisma_client.db.tx() as db:
        store: Final = AppStore(db, cache=user_api_key_cache)
        yield store
    await store.invalidate_definition_cache()


Store: TypeAlias = Annotated[AppStore, Depends(get_app_store)]
Actor: TypeAlias = Annotated[str, Depends(customer_actor)]
router: Final = APIRouter(
    prefix=MANAGEMENT_V1_PREFIX + "/apps",
    tags=["apps"],
    dependencies=(Depends(require_customer_admin),),
)


@router.get("/definition", response_model=AppDefinition)
async def get_app_definition(store: Store) -> AppDefinition:
    return await store.definition()


@router.put("/definition", response_model=AppDefinition)
async def put_app_definition(data: AppDefinition, store: Store) -> AppDefinition:
    return resolve(await store.put_definition(data))


@router.get("/resources", response_model=tuple[AppResource, ...])
async def list_app_resources(
    store: Store,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
    resource_type: AppResourceType | None = None,
) -> tuple[AppResource, ...]:
    return await store.resources(offset, limit, resource_type)


@router.post(
    "",
    response_model=App,
    status_code=201,
    description="Configure budget policy for an existing resource of the gateway app type. Proxy admins only.",
)
async def create_app(data: AppCreate, store: Store, actor: Actor) -> App:
    return await store.with_budget_usage(resolve(await store.create(data, actor)))


@router.get("", response_model=tuple[App, ...])
async def list_apps(
    store: Store,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> tuple[App, ...]:
    return tuple(await asyncio.gather(*(store.with_budget_usage(app) for app in await store.list(offset, limit))))


@router.get("/{app_id}", response_model=App)
async def get_app(app_id: str, store: Store) -> App:
    return await store.with_budget_usage(resolve(await store.get(app_id)))


@router.patch("/{app_id}", response_model=App)
async def patch_app(app_id: str, data: AppPatch, store: Store, actor: Actor) -> App:
    return await store.with_budget_usage(resolve(await store.patch(app_id, data, actor)))


@router.put("/{app_id}/customer_organizations/{customer_org_id}", response_model=AppOrganization)
async def put_app_organization(
    app_id: str,
    customer_org_id: str,
    data: AppOrganizationPut,
    store: Store,
    actor: Actor,
) -> AppOrganization:
    return await store.with_budget_usage(resolve(await store.put_organization(app_id, customer_org_id, data, actor)))


@router.get("/{app_id}/customer_organizations/{customer_org_id}", response_model=AppOrganization)
async def get_app_organization(app_id: str, customer_org_id: str, store: Store) -> AppOrganization:
    app: Final = resolve(await store.get(app_id))
    organization: Final = await store.find_organization(app, customer_org_id)
    if organization is None:
        return resolve(AppFailure(404, "App customer organization not found."))
    return await store.with_budget_usage(organization)


@router.put("/{app_id}/customer_members/{customer_member_id}", response_model=AppMember)
async def put_app_member(
    app_id: str,
    customer_member_id: str,
    data: AppMemberPut,
    store: Store,
    actor: Actor,
) -> AppMember:
    return await store.with_budget_usage(resolve(await store.put_member(app_id, customer_member_id, data, actor)))


@router.get("/{app_id}/customer_members/{customer_member_id}", response_model=AppMember)
async def get_app_member(app_id: str, customer_member_id: str, store: Store) -> AppMember:
    return await store.with_budget_usage(resolve(await store.get_member(app_id, customer_member_id)))
