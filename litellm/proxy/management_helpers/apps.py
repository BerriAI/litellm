import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, cast  # noqa: TID251  # Prisma proxies implement the generated client surface

from pydantic import BaseModel, ConfigDict

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchyStore,
    PrismaRecord,
    related_budget,
)
from litellm.repositories.table_repositories import PrismaTableRepository
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
    TierPatch,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, BudgetTemplate

if TYPE_CHECKING:
    from prisma import Prisma


@dataclass(frozen=True, slots=True)
class _Connection:
    db: object


class _AppAlias(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_id: str


def app_alias_cache_key(alias: str) -> str:
    return f"app_alias:{alias}"


class AppRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_apptable"


class AppCustomerOrganizationRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_appcustomerorgtable"


class AppCustomerMemberRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_appcustomermembertable"


def app_scope_id(app_id: str, customer_id: str) -> str:
    return f"{app_id}:{customer_id}"


def app_scope_identity(entity_id: str) -> tuple[str, str]:
    app_id, _, customer_id = entity_id.partition(":")
    return app_id, customer_id


class AppStore:
    def __init__(self, db: object, now: datetime | None = None, cache: UserApiKeyCache | None = None) -> None:
        self.db: Final = cast("Prisma", db)  # cast-ok: Prisma wrappers share the generated client API
        self.cache: Final = cache
        self.now: Final = now or datetime.now(timezone.utc)
        connection: Final = _Connection(db)
        self.apps: Final = AppRepository(connection).table
        self.organizations: Final = AppCustomerOrganizationRepository(connection).table
        self.members: Final = AppCustomerMemberRepository(connection).table
        self.customers: Final = CustomerHierarchyStore(self.db, self.now, premium_user=True)

    async def budget(self, budget_id: str | None) -> Budget | None:
        if budget_id is None:
            return None
        row: Final = await self.customers.budgets.find_unique(where={"budget_id": budget_id})
        return related_budget(row.model_dump(), self.now) if row is not None else None

    async def get(self, app_id: str, *, lock: bool = False) -> App | AppFailure:
        if lock:
            await self.db.query_raw('SELECT app_id FROM "LiteLLM_AppTable" WHERE app_id = $1 FOR UPDATE', app_id)
        row: Final = await self.apps.find_unique(where={"app_id": app_id})
        if row is None:
            return AppFailure(404, "App not found.")
        return await self.response(row)

    async def response(self, row: PrismaRecord) -> App:
        app: Final = App.model_validate(row.model_dump())
        tiers: Final = await asyncio.gather(*(self.tier(budget_id) for budget_id in app.tier_budget_ids))
        return app.model_copy(
            update={
                "budget": await self.budget(app.budget_id),
                "tiers": tuple(tier for tier in tiers if tier is not None),
                "default_customer_member_budget": await self.budget(app.default_customer_member_budget_id),
            }
        )

    async def by_alias(self, alias: str) -> App | AppFailure:
        row: Final = await self.alias_row(alias)
        if row is None:
            return AppFailure(400, "Unknown app alias.")
        app: Final = await self.response(row)
        if self.cache is not None:
            await self.cache.async_set_cache(
                app_alias_cache_key(alias),
                _AppAlias(app_id=app.app_id),
                model_type=_AppAlias,
                ttl=60,
            )
        return app

    async def alias_row(self, alias: str) -> PrismaRecord | None:
        if self.cache is not None:
            cached: Final = await self.cache.async_get_cache(key=app_alias_cache_key(alias), model_type=_AppAlias)
            if cached is not None:
                row: Final = await self.apps.find_unique(where={"app_id": cached.app_id})
                if row is not None and App.model_validate(row.model_dump()).alias == alias:
                    return row
        return await self.apps.find_unique(where={"alias": alias})

    async def list(self, offset: int, limit: int) -> tuple[App, ...]:
        rows: Final = await self.apps.find_many(skip=offset, take=limit, order={"app_id": "asc"})
        return tuple(await asyncio.gather(*(self.response(row) for row in rows)))

    async def create(self, data: AppCreate, actor: str) -> App | AppFailure:
        from prisma.errors import UniqueViolationError

        if data.default_tier_budget_id is not None:
            return AppFailure(422, "Create the app's tiers before selecting a default tier.")
        values: Final = await self.fields(data, actor)
        try:
            row: Final = await self.apps.create(data={"alias": data.alias, "created_by": actor, **values})
        except UniqueViolationError:
            return AppFailure(409, "App alias already exists.")
        return await self.response(row)

    async def fields(self, data: AppPatch | AppCreate, actor: str, current: App | None = None) -> Mapping[str, object]:
        from prisma import Json

        budget_id: Final = (
            await self.customers.update_budget(current.budget_id if current is not None else None, data.budget, actor)
            if data.budget is not None
            else None
        )
        member_id: Final = (
            await self.customers.update_budget(
                current.default_customer_member_budget_id if current is not None else None,
                data.default_customer_member_budget,
                actor,
            )
            if data.default_customer_member_budget is not None
            else None
        )
        return {
            **({"alias": data.alias} if data.alias is not None else {}),
            **({"budget_id": budget_id} if "budget" in data.model_fields_set else {}),
            **(
                {"default_tier_budget_id": data.default_tier_budget_id}
                if "default_tier_budget_id" in data.model_fields_set
                else {}
            ),
            **(
                {"default_customer_member_budget_id": member_id}
                if "default_customer_member_budget" in data.model_fields_set
                else {}
            ),
            **({"upsert_policy": Json(dict(data.upsert_policy))} if data.upsert_policy is not None else {}),
            "updated_by": actor,
        }

    async def patch(self, app_id: str, data: AppPatch, actor: str) -> App | AppFailure:
        from prisma.errors import UniqueViolationError

        app: Final = await self.get(app_id, lock=True)
        if isinstance(app, AppFailure):
            return app
        if data.default_tier_budget_id is not None and data.default_tier_budget_id not in app.tier_budget_ids:
            return AppFailure(422, "The default tier must belong to this app.")
        if "alias" in data.model_fields_set and data.alias is None:
            return AppFailure(422, "App alias cannot be null.")
        try:
            row: Final = await self.apps.update(where={"app_id": app_id}, data=await self.fields(data, actor, app))
        except UniqueViolationError:
            return AppFailure(409, "App alias already exists.")
        return AppFailure(404, "App not found.") if row is None else await self.response(row)

    async def delete(self, app_id: str) -> None | AppFailure:
        app: Final = await self.get(app_id, lock=True)
        if isinstance(app, AppFailure):
            return app
        await self.apps.delete(where={"app_id": app_id})
        return None

    async def tier(self, budget_id: str) -> AppTier | None:
        row: Final = await self.customers.budgets.find_unique(where={"budget_id": budget_id})
        budget: Final = related_budget(row.model_dump()) if row is not None else None
        if row is None or budget is None:
            return None
        return AppTier.model_validate({**row.model_dump(), **budget.model_dump()})

    async def create_tier(self, app_id: str, data: BudgetTemplate, actor: str) -> AppTier | AppFailure:
        app: Final = await self.get(app_id, lock=True)
        if isinstance(app, AppFailure):
            return app
        budget_id: Final = await self.customers.create_budget(data, actor, data.name)
        await self.apps.update(
            where={"app_id": app_id},
            data={
                "tier_budget_ids": (*app.tier_budget_ids, budget_id),
                "updated_by": actor,
            },
        )
        tier: Final = await self.tier(budget_id)
        return tier if tier is not None else AppFailure(404, "Tier not found.")

    async def patch_tier(self, app_id: str, data: TierPatch, actor: str) -> AppTier | AppFailure:
        app: Final = await self.get(app_id, lock=True)
        if isinstance(app, AppFailure):
            return app
        if data.budget_id not in app.tier_budget_ids:
            return AppFailure(422, "Tier does not belong to this app.")
        budget: Final = Budget.model_validate(data.model_dump(exclude={"budget_id", "name"}, exclude_unset=True))
        await self.customers.update_budget(data.budget_id, budget, actor)
        if "name" in data.model_fields_set:
            if data.name is None:
                return AppFailure(422, "Tier name cannot be null.")
            await self.customers.budgets.update(where={"budget_id": data.budget_id}, data={"name": data.name})
        tier: Final = await self.tier(data.budget_id)
        return tier if tier is not None else AppFailure(404, "Tier not found.")

    async def delete_tier(self, app_id: str, budget_id: str, actor: str) -> None | AppFailure:
        app: Final = await self.get(app_id, lock=True)
        if isinstance(app, AppFailure):
            return app
        if budget_id not in app.tier_budget_ids:
            return AppFailure(422, "Tier does not belong to this app.")
        if app.default_tier_budget_id == budget_id or await self.organizations.count(
            where={"app_id": app_id, "tier_budget_id": budget_id}
        ):
            return AppFailure(409, "A default or referenced tier cannot be deleted.")
        await self.apps.update(
            where={"app_id": app_id},
            data={
                "tier_budget_ids": tuple(value for value in app.tier_budget_ids if value != budget_id),
                "updated_by": actor,
            },
        )
        await self.customers.budgets.delete(where={"budget_id": budget_id})
        return None

    async def organization(self, app: App, customer_org_id: str) -> AppOrganization:
        identity: Final = {"app_id": app.app_id, "customer_org_id": customer_org_id}
        existing: Final = await self.organizations.find_unique(where={"app_id_customer_org_id": identity})
        row: Final = (
            existing
            if existing is not None
            else await self.organizations.upsert(
                where={"app_id_customer_org_id": identity},
                data={"create": identity, "update": {"app_id": app.app_id}},
            )
        )
        return await self.organization_response(app, row)

    async def organization_response(self, app: App, row: PrismaRecord) -> AppOrganization:
        org: Final = AppOrganization.model_validate(row.model_dump())
        return org.model_copy(
            update={
                "effective_budget": await self.budget(
                    org.budget_override_id or org.tier_budget_id or app.default_tier_budget_id
                ),
                "effective_customer_member_budget": await self.budget(
                    org.customer_member_budget_id or app.default_customer_member_budget_id
                ),
            }
        )

    async def put_organization(
        self,
        app_id: str,
        customer_org_id: str,
        data: AppOrganizationPut,
        actor: str,
    ) -> AppOrganization | AppFailure:
        app: Final = await self.get(app_id, lock=True)
        if isinstance(app, AppFailure):
            return app
        if data.tier_budget_id is not None and data.tier_budget_id not in app.tier_budget_ids:
            return AppFailure(422, "Tier does not belong to this app.")
        if await self.customers.organizations.find_unique(where={"customer_org_id": customer_org_id}) is None:
            return AppFailure(404, "Customer organization not found.")
        override: Final = (
            await self.customers.create_budget(data.budget_override, actor)
            if data.budget_override is not None
            else None
        )
        member: Final = (
            await self.customers.create_budget(data.customer_member_budget, actor)
            if data.customer_member_budget is not None
            else None
        )
        identity: Final = {"app_id": app_id, "customer_org_id": customer_org_id}
        fields: Final = {
            "tier_budget_id": data.tier_budget_id,
            "budget_override_id": override,
            "customer_member_budget_id": member,
        }
        row: Final = await self.organizations.upsert(
            where={"app_id_customer_org_id": identity},
            data={"create": {**identity, **fields}, "update": fields},
        )
        return await self.organization_response(app, row)

    async def member(self, app: App, customer_member_id: str, org: AppOrganization) -> AppMember:
        identity: Final = {"app_id": app.app_id, "customer_member_id": customer_member_id}
        existing: Final = await self.members.find_unique(where={"app_id_customer_member_id": identity})
        row: Final = (
            existing
            if existing is not None
            else await self.members.upsert(
                where={"app_id_customer_member_id": identity},
                data={"create": identity, "update": {"app_id": app.app_id}},
            )
        )
        member: Final = AppMember.model_validate(row.model_dump())
        return member.model_copy(
            update={
                "effective_budget": (
                    await self.budget(member.budget_override_id)
                    if member.budget_override_id is not None
                    else org.effective_customer_member_budget
                ),
            }
        )

    async def put_member(
        self,
        app_id: str,
        customer_member_id: str,
        data: AppMemberPut,
        actor: str,
    ) -> AppMember | AppFailure:
        app: Final = await self.get(app_id, lock=True)
        if isinstance(app, AppFailure):
            return app
        customer: Final = await self.customers.members.find_unique(where={"customer_member_id": customer_member_id})
        if customer is None:
            return AppFailure(404, "Customer member not found.")
        parent: Final = customer.model_dump().get("parent_org_id")
        if not isinstance(parent, str):
            return AppFailure(404, "Customer organization not found.")
        override: Final = (
            await self.customers.create_budget(data.budget_override, actor)
            if data.budget_override is not None
            else None
        )
        identity: Final = {"app_id": app_id, "customer_member_id": customer_member_id}
        await self.members.upsert(
            where={"app_id_customer_member_id": identity},
            data={
                "create": {**identity, "budget_override_id": override},
                "update": {"budget_override_id": override},
            },
        )
        return await self.member(app, customer_member_id, await self.organization(app, parent))
