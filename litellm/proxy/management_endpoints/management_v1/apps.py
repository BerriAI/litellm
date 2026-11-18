import asyncio
from collections.abc import AsyncIterator
from typing import Annotated, Final, TypeAlias, TypeVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from pydantic import TypeAdapter

from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import customer_actor, require_customer_admin
from litellm.proxy.management_helpers.apps import AppStore, app_alias_cache_key
from litellm.types.proxy.management_endpoints.apps import (
    App,
    AppCreate,
    AppFailure,
    AppMember,
    AppMemberPut,
    AppOrganization,
    AppOrganizationPut,
    AppPatch,
    AppTier,
    TierDelete,
    TierPatch,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import BudgetTemplate

T = TypeVar("T")  # rebind-ok: TypeVar declarations cannot be annotated Final


def resolve(result: T | AppFailure) -> T:
    if isinstance(result, AppFailure):
        raise HTTPException(result.status, result.detail)
    return result


async def get_app_store(request: Request) -> AsyncIterator[AppStore]:
    from litellm.proxy.proxy_server import prisma_client, user_api_key_cache

    if prisma_client is None:
        resolve(AppFailure(503, "Database unavailable."))
        return
    app_id: Final = TypeAdapter[str | None](str | None).validate_python(request.path_params.get("app_id"))
    before: Final = await AppStore(prisma_client.db).get(app_id) if app_id is not None else None
    async with prisma_client.db.tx() as db:
        yield AppStore(db)
    after: Final = await AppStore(prisma_client.db).get(app_id) if app_id is not None else None
    aliases: Final = frozenset(value.alias for value in (before, after) if isinstance(value, App))
    await asyncio.gather(*(user_api_key_cache.async_delete_cache(key=app_alias_cache_key(alias)) for alias in aliases))


Store: TypeAlias = Annotated[AppStore, Depends(get_app_store)]
Actor: TypeAlias = Annotated[str, Depends(customer_actor)]
router: Final = APIRouter(
    prefix=MANAGEMENT_V1_PREFIX + "/apps",
    tags=["apps"],
    dependencies=(Depends(require_customer_admin),),
)


@router.post("", response_model=App, status_code=201, description="Create an app with its budgets. Proxy admins only.")
async def create_app(data: AppCreate, store: Store, actor: Actor) -> App:
    return resolve(await store.create(data, actor))


@router.get("", response_model=tuple[App, ...])
async def list_apps(
    store: Store,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> tuple[App, ...]:
    return await store.list(offset, limit)


@router.get("/{app_id}", response_model=App)
async def get_app(app_id: str, store: Store) -> App:
    return resolve(await store.get(app_id))


@router.patch("/{app_id}", response_model=App)
async def patch_app(app_id: str, data: AppPatch, store: Store, actor: Actor) -> App:
    return resolve(await store.patch(app_id, data, actor))


@router.delete("/{app_id}", status_code=204)
async def delete_app(app_id: str, store: Store) -> Response:
    resolve(await store.delete(app_id))
    return Response(status_code=204)


@router.post(
    "/{app_id}/tiers",
    response_model=AppTier,
    status_code=201,
    description="Create a budget tier owned by this app. Proxy admins only.",
)
async def create_tier(app_id: str, data: BudgetTemplate, store: Store, actor: Actor) -> AppTier:
    return resolve(await store.create_tier(app_id, data, actor))


@router.patch("/{app_id}/tiers", response_model=AppTier)
async def patch_tier(app_id: str, data: TierPatch, store: Store, actor: Actor) -> AppTier:
    return resolve(await store.patch_tier(app_id, data, actor))


@router.delete("/{app_id}/tiers", status_code=204)
async def delete_tier(app_id: str, data: TierDelete, store: Store, actor: Actor) -> Response:
    resolve(await store.delete_tier(app_id, data.budget_id, actor))
    return Response(status_code=204)


@router.put("/{app_id}/customer_organizations/{customer_org_id}", response_model=AppOrganization)
async def put_app_organization(
    app_id: str,
    customer_org_id: str,
    data: AppOrganizationPut,
    store: Store,
    actor: Actor,
) -> AppOrganization:
    return resolve(await store.put_organization(app_id, customer_org_id, data, actor))


@router.get("/{app_id}/customer_organizations/{customer_org_id}", response_model=AppOrganization)
async def get_app_organization(app_id: str, customer_org_id: str, store: Store) -> AppOrganization:
    app: Final = resolve(await store.get(app_id))
    row: Final = await store.organizations.find_unique(
        where={"app_id_customer_org_id": {"app_id": app_id, "customer_org_id": customer_org_id}},
    )
    if row is None:
        return resolve(AppFailure(404, "App customer organization not found."))
    return await store.organization_response(app, row)


@router.get("/{app_id}/customer_organizations", response_model=tuple[AppOrganization, ...])
async def list_app_organizations(
    app_id: str,
    store: Store,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> tuple[AppOrganization, ...]:
    app: Final = resolve(await store.get(app_id))
    rows: Final = await store.organizations.find_many(
        where={"app_id": app_id},
        skip=offset,
        take=limit,
        order={"customer_org_id": "asc"},
    )
    return tuple(await asyncio.gather(*(store.organization_response(app, row) for row in rows)))


@router.put("/{app_id}/customer_members/{customer_member_id}", response_model=AppMember)
async def put_app_member(
    app_id: str,
    customer_member_id: str,
    data: AppMemberPut,
    store: Store,
    actor: Actor,
) -> AppMember:
    return resolve(await store.put_member(app_id, customer_member_id, data, actor))


@router.get("/{app_id}/customer_members/{customer_member_id}", response_model=AppMember)
async def get_app_member(app_id: str, customer_member_id: str, store: Store) -> AppMember:
    return resolve(await store.get_member(app_id, customer_member_id))
