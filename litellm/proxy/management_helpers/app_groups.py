import asyncio
from collections.abc import Mapping
from typing import Final

from litellm.proxy.management_helpers.apps import AppStore, app_budget_selection
from litellm.proxy.management_helpers.customer_hierarchy import BudgetGroupScope, PrismaRecord
from litellm.redis.stores.auth_cache import AuthObject
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.types.proxy.management_endpoints.app_groups import (
    AppGroup,
    AppGroupBudgetPut,
    AppGroupCreate,
    AppGroupOrganization,
    AppGroupPatch,
    AppGroupSettings,
)
from litellm.types.proxy.management_endpoints.apps import App, AppCreate, AppFailure, AppPatch


class AppGroupRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_appgrouptable"


class AppGroupOrganizationRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_appgroupcustomerorgtable"


class AppGroupStore:
    def __init__(self, apps: AppStore) -> None:
        self.apps: Final = apps
        self.groups: Final = AppGroupRepository(apps).table
        self.organizations: Final = AppGroupOrganizationRepository(apps).table

    async def response(self, row: PrismaRecord) -> AppGroup:
        group: Final = AppGroup.model_validate(row.model_dump())
        return group.model_copy(update={"budget": await self.apps.budget(group.budget_id)})

    async def get(self, app_group_id: str) -> AppGroup | AppFailure:
        async def read() -> PrismaRecord | None:
            return await self.groups.find_unique(where={"app_group_id": app_group_id})

        row: Final = await self.apps.cached_row(AuthObject.customer_app("app_group", app_group_id), read)
        return (
            await self.response(AppGroup.model_validate(row.values))
            if row.values is not None
            else AppFailure(404, "App group not found.")
        )

    async def list(self, offset: int, limit: int) -> tuple[AppGroup, ...]:
        rows: Final = await self.groups.find_many(skip=offset, take=limit, order={"app_group_id": "asc"})
        return tuple(await asyncio.gather(*(self.response(row) for row in rows)))

    async def validate_budget(self, data: AppGroupBudgetPut, scope: BudgetGroupScope) -> AppFailure | None:
        if not await self.apps.valid_profile(data.budget_id, scope):
            return AppFailure(422, "Budget preset must belong to the selected budget catalog.")
        return None

    async def validate_apps(self, data: AppGroupSettings) -> AppFailure | None:
        if "apps" not in data.model_fields_set:
            return None
        definition: Final = await self.apps.lock_definition()
        for selected in data.apps:
            if selected.resource_type != definition.resource_type:
                return AppFailure(422, "Selected apps must match the gateway app definition.")
            if isinstance(
                resource := await self.apps.resource(selected.resource_type, selected.resource_id), AppFailure
            ):
                return resource
        return None

    async def replace_apps(self, group_id: str, data: AppGroupSettings, actor: str) -> AppFailure | None:
        if "apps" not in data.model_fields_set:
            return None
        selected: Final = frozenset((app.resource_type, app.resource_id) for app in data.apps)
        current: Final = await self.apps.apps.find_many(where={"app_group_id": group_id})
        for app in (App.model_validate(row.model_dump()) for row in current):
            if (app.resource_type, app.resource_id) in selected:
                continue
            if isinstance(removed := await self.apps.patch(app.app_id, AppPatch(app_group_id=None), actor), AppFailure):
                return removed
        for binding in data.apps:
            if isinstance(
                configured := await self.apps.create(
                    AppCreate(
                        resource_type=binding.resource_type, resource_id=binding.resource_id, app_group_id=group_id
                    ),
                    actor,
                ),
                AppFailure,
            ):
                return configured
            if configured.app_group_id == group_id:
                continue
            if isinstance(
                assigned := await self.apps.patch(configured.app_id, AppPatch(app_group_id=group_id), actor), AppFailure
            ):
                return assigned
        return None

    async def fields(
        self, data: AppGroupPatch | AppGroupCreate, current_id: str | None, actor: str
    ) -> Mapping[str, object]:
        from prisma import Json

        selected: Final = await self.apps.linked_budget_id(
            current_id,
            app_budget_selection(data),
            actor,
            BudgetGroupScope.APP_GROUP,
            apply_default=isinstance(data, AppGroupCreate),
        )
        return {
            **data.model_dump(include={"name", "description"}, exclude_unset=True),
            **(
                {"metadata": Json(dict(data.metadata)) if data.metadata is not None else None}
                if "metadata" in data.model_fields_set
                else {}
            ),
            **(
                {"budget_id": selected}
                if isinstance(data, AppGroupCreate) or {"budget", "budget_id"} & data.model_fields_set
                else {}
            ),
            "updated_by": actor,
        }

    async def create(self, data: AppGroupCreate, actor: str) -> AppGroup | AppFailure:
        membership_failure: Final = await self.validate_apps(data)
        if membership_failure is not None:
            return membership_failure
        await self.apps.customers.lock_customer_identities()
        failure: Final = await self.validate_budget(data, BudgetGroupScope.APP_GROUP)
        if failure is not None:
            return failure
        if await self.groups.find_unique(where={"name": data.name}) is not None:
            return AppFailure(409, "An app group with this name already exists.")
        row: Final = await self.groups.create(data={**await self.fields(data, None, actor), "created_by": actor})
        created: Final = await self.response(row)
        assigned: Final = await self.replace_apps(created.app_group_id, data, actor)
        return assigned if assigned is not None else created

    async def patch(self, app_group_id: str, data: AppGroupPatch, actor: str) -> AppGroup | AppFailure:
        membership_failure: Final = await self.validate_apps(data)
        if membership_failure is not None:
            return membership_failure
        await self.apps.customers.lock_customer_identities()
        current: Final = await self.get(app_group_id)
        if isinstance(current, AppFailure):
            return current
        failure: Final = await self.validate_budget(data, BudgetGroupScope.APP_GROUP)
        if failure is not None:
            return failure
        if (
            data.name is not None
            and data.name != current.name
            and await self.groups.find_unique(where={"name": data.name}) is not None
        ):
            return AppFailure(409, "An app group with this name already exists.")
        row: Final = await self.groups.update(
            where={"app_group_id": app_group_id}, data=await self.fields(data, current.budget_id, actor)
        )
        if row is None:
            return AppFailure(404, "App group not found.")
        updated: Final = await self.response(row)
        self.apps.customers.auth_cache_keys += (AuthObject.customer_app("app_group", app_group_id).key,)
        await self.apps.customers.delete_released_budget(current.budget_id, updated.budget_id)
        assigned: Final = await self.replace_apps(app_group_id, data, actor)
        return assigned if assigned is not None else updated

    async def organization_response(self, row: PrismaRecord) -> AppGroupOrganization:
        org: Final = AppGroupOrganization.model_validate(row.model_dump())
        return org.model_copy(update={"budget": await self.apps.budget(org.budget_id)})

    async def organization(self, app_group_id: str, customer_org_id: str) -> AppGroupOrganization:
        identity: Final = {"app_group_id": app_group_id, "customer_org_id": customer_org_id}

        async def read() -> PrismaRecord:
            existing: Final = await self.organizations.find_unique(where={"app_group_id_customer_org_id": identity})
            if existing is not None:
                return existing
            return await self.organizations.upsert(
                where={"app_group_id_customer_org_id": identity},
                data={"create": identity, "update": {}},
            )

        row: Final = await self.apps.cached_row(
            AuthObject.customer_app_association("group_organization", app_group_id, customer_org_id), read
        )
        return await self.organization_response(AppGroupOrganization.model_validate(row.values))

    async def list_organizations(self, app_group_id: str, offset: int, limit: int) -> tuple[AppGroupOrganization, ...]:
        rows: Final = await self.organizations.find_many(
            where={"app_group_id": app_group_id}, skip=offset, take=limit, order={"customer_org_id": "asc"}
        )
        return tuple(await asyncio.gather(*(self.organization_response(row) for row in rows)))

    async def put_organization(
        self, app_group_id: str, customer_org_id: str, data: AppGroupBudgetPut, actor: str
    ) -> AppGroupOrganization | AppFailure:
        await self.apps.customers.lock_customer_identities()
        group: Final = await self.get(app_group_id)
        if isinstance(group, AppFailure):
            return group
        failure: Final = await self.validate_budget(data, BudgetGroupScope.APP_ORGANIZATION)
        if failure is not None:
            return failure
        if await self.apps.customers.organizations.find_unique(where={"customer_org_id": customer_org_id}) is None:
            return AppFailure(404, "Customer organization not found.")
        current: Final = await self.organization(app_group_id, customer_org_id)
        selected: Final = await self.apps.linked_budget_id(
            current.budget_id, app_budget_selection(data), actor, BudgetGroupScope.APP_ORGANIZATION
        )
        identity: Final = {"app_group_id": app_group_id, "customer_org_id": customer_org_id}
        row: Final = await self.organizations.update(
            where={"app_group_id_customer_org_id": identity}, data={"budget_id": selected}
        )
        if row is None:
            return AppFailure(404, "App group customer organization not found.")
        self.apps.customers.auth_cache_keys += (
            AuthObject.customer_app_association("group_organization", app_group_id, customer_org_id).key,
        )
        await self.apps.customers.delete_released_budget(current.budget_id, selected)
        return await self.organization_response(row)
