import asyncio
from typing import Annotated, Final

from fastapi import APIRouter, Depends, Query

from litellm.proxy.management_endpoints.management_v1.apps import Actor, Store, resolve
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_endpoints.management_v1.customers import require_customer_admin
from litellm.proxy.management_helpers.app_groups import AppGroupStore
from litellm.types.proxy.management_endpoints.app_groups import (
    AppGroup,
    AppGroupBudgetPut,
    AppGroupCreate,
    AppGroupOrganization,
    AppGroupPatch,
)

router: Final = APIRouter(
    prefix=MANAGEMENT_V1_PREFIX + "/app_groups",
    tags=["app groups"],
    dependencies=(Depends(require_customer_admin),),
)


@router.post("", response_model=AppGroup, status_code=201)
async def create_app_group(data: AppGroupCreate, store: Store, actor: Actor) -> AppGroup:
    """Create an app group from selected apps and an optional custom budget or app group budget preset.

    Omit budget and budget_id to use the app group catalog default. Explicit null disables the shared cap.
    Proxy admins only.
    """
    groups: Final = AppGroupStore(store)
    return await store.with_budget_usage(resolve(await groups.create(data, actor)))


@router.get("", response_model=tuple[AppGroup, ...])
async def list_app_groups(
    store: Store,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> tuple[AppGroup, ...]:
    groups: Final = AppGroupStore(store)
    rows: Final = await groups.list(offset, limit)
    return tuple(await asyncio.gather(*(store.with_budget_usage(row) for row in rows)))


@router.get("/{app_group_id}", response_model=AppGroup)
async def get_app_group(app_group_id: str, store: Store) -> AppGroup:
    groups: Final = AppGroupStore(store)
    return await store.with_budget_usage(resolve(await groups.get(app_group_id)))


@router.patch("/{app_group_id}", response_model=AppGroup)
async def patch_app_group(app_group_id: str, data: AppGroupPatch, store: Store, actor: Actor) -> AppGroup:
    groups: Final = AppGroupStore(store)
    return await store.with_budget_usage(resolve(await groups.patch(app_group_id, data, actor)))


@router.get("/{app_group_id}/customer_organizations", response_model=tuple[AppGroupOrganization, ...])
async def list_app_group_organizations(
    app_group_id: str,
    store: Store,
    offset: Annotated[int, Query(ge=0)] = 0,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
) -> tuple[AppGroupOrganization, ...]:
    groups: Final = AppGroupStore(store)
    resolve(await groups.get(app_group_id))
    rows: Final = await groups.list_organizations(app_group_id, offset, limit)
    return tuple(await asyncio.gather(*(store.with_budget_usage(row) for row in rows)))


@router.put("/{app_group_id}/customer_organizations/{customer_org_id}", response_model=AppGroupOrganization)
async def put_app_group_organization(
    app_group_id: str, customer_org_id: str, data: AppGroupBudgetPut, store: Store, actor: Actor
) -> AppGroupOrganization:
    groups: Final = AppGroupStore(store)
    return await store.with_budget_usage(
        resolve(await groups.put_organization(app_group_id, customer_org_id, data, actor))
    )
