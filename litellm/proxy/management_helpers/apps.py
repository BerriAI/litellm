import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Final,
    TypeVar,
    cast,  # noqa: TID251  # Prisma proxies implement the generated client surface
)
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from litellm.proxy.management_helpers.customer_budget_usage import app_budget_usage
from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchyStore,
    PrismaRecord,
    organization_budget_group_ids,
    related_budget,
)
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.types.proxy.auth.customer_hierarchy import CustomerScopeKey
from litellm.types.proxy.management_endpoints.apps import (
    App,
    AppCreate,
    AppDefinition,
    AppFailure,
    AppMember,
    AppMemberBudgetSource,
    AppMemberPut,
    AppOrganization,
    AppOrganizationPut,
    AppPatch,
    AppResource,
    AppResourceType,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget

if TYPE_CHECKING:
    from prisma import Prisma


@dataclass(frozen=True, slots=True)
class _Connection:
    db: object


class _DefinitionState(AppDefinition):
    model_config = ConfigDict(frozen=True, extra="ignore")

    locked: bool = False


class _ConfigRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    param_value: object


class _ResourceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    team_id: str | None = None
    team_alias: str | None = None
    project_id: str | None = None
    project_alias: str | None = None
    key_id: str | None = None
    key_alias: str | None = None
    token: str | None = Field(default=None, repr=False)


_DEFINITION_PARAM: Final = "customer_app_definition"
_PARENT_ORG_ID: Final = TypeAdapter(str)
AppEntity: Final = TypeVar("AppEntity", App, AppOrganization, AppMember)


class AppRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_apptable"


class AppCustomerOrganizationRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_appcustomerorgtable"


class AppCustomerMemberRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_appcustomermembertable"


class _ConfigRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_config"


class _TeamRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_teamtable"


class _ProjectRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_projecttable"


class _KeyRepository(PrismaTableRepository[PrismaRecord]):
    table_name = "litellm_verificationtoken"


def app_scope_id(app_id: str, customer_id: str) -> str:
    return f"{app_id}:{customer_id}"


def app_scope_identity(entity_id: str) -> tuple[str, str]:
    app_id, _, customer_id = entity_id.partition(":")
    return app_id, customer_id


@dataclass(frozen=True, slots=True)
class _AppBudgetIdentity:
    key: CustomerScopeKey
    entity_id: str
    budget: Budget | None


def _app_budget_identity(
    entity: App | AppOrganization | AppMember,
) -> _AppBudgetIdentity:
    if isinstance(entity, App):
        return _AppBudgetIdentity("app", entity.app_id, entity.budget)
    if isinstance(entity, AppOrganization):
        return _AppBudgetIdentity(
            "app_customer_org",
            app_scope_id(entity.app_id, entity.customer_org_id),
            entity.effective_budget,
        )
    return _AppBudgetIdentity(
        "app_customer_member",
        app_scope_id(entity.app_id, entity.customer_member_id),
        entity.effective_budget,
    )


class AppStore:
    def __init__(
        self,
        db: object,
        now: datetime | None = None,
    ) -> None:
        self.db: Final = cast("Prisma", db)  # cast-ok: Prisma wrappers share the generated client API
        self.now: Final = now or datetime.now(timezone.utc)
        connection: Final = _Connection(db)
        self.apps: Final = AppRepository(connection).table
        self.organizations: Final = AppCustomerOrganizationRepository(connection).table
        self.members: Final = AppCustomerMemberRepository(connection).table
        self.config: Final = _ConfigRepository(connection).table
        self.teams: Final = _TeamRepository(connection).table
        self.projects: Final = _ProjectRepository(connection).table
        self.keys: Final = _KeyRepository(connection).table
        self.customers: Final = CustomerHierarchyStore(self.db, self.now, premium_user=True)

    @staticmethod
    def definition_state(row: PrismaRecord | None) -> _DefinitionState:
        if row is None:
            return _DefinitionState()
        value: Final = _ConfigRecord.model_validate(row.model_dump()).param_value
        return (
            _DefinitionState.model_validate_json(value)
            if isinstance(value, str)
            else _DefinitionState.model_validate(value)
        )

    async def definition(self) -> AppDefinition:
        row: Final = await self.config.find_unique(where={"param_name": _DEFINITION_PARAM})
        state: Final = self.definition_state(row)
        return AppDefinition(
            resource_type=state.resource_type,
            auto_create_customers_without_app=state.auto_create_customers_without_app,
        )

    async def lock_definition(self) -> _DefinitionState:
        from prisma import Json

        row: Final = await self.config.upsert(
            where={"param_name": _DEFINITION_PARAM},
            data={
                "create": {
                    "param_name": _DEFINITION_PARAM,
                    "param_value": Json({"resource_type": None}),
                },
                "update": {"last_run_at": self.now},
            },
        )
        return self.definition_state(row)

    async def put_definition(self, data: AppDefinition) -> AppDefinition | AppFailure:
        from prisma import Json

        current: Final = await self.lock_definition()
        without_app: Final = (
            data.auto_create_customers_without_app
            if "auto_create_customers_without_app" in data.model_fields_set
            else current.auto_create_customers_without_app
        )
        stored: Final = AppDefinition(resource_type=data.resource_type, auto_create_customers_without_app=without_app)
        if data.resource_type != current.resource_type and (
            current.locked or await self.apps.count(where={"resource_type": {"not": None}})
        ):
            return AppFailure(
                409,
                "The app resource definition cannot change after a resource has been configured.",
            )
        if stored == AppDefinition.model_validate(current.model_dump(exclude={"locked"})):
            return stored
        await self.config.update(
            where={"param_name": _DEFINITION_PARAM},
            data={"param_value": Json({**stored.model_dump(mode="json"), "locked": current.locked})},
        )
        return stored

    def resource_table(self, resource_type: AppResourceType) -> tuple[TableActions[PrismaRecord], str]:
        match resource_type:
            case AppResourceType.TEAM:
                return self.teams, "team_id"
            case AppResourceType.PROJECT:
                return self.projects, "project_id"
            case AppResourceType.VIRTUAL_KEY:
                return self.keys, "key_id"

    async def resource_response(self, resource_type: AppResourceType, row: PrismaRecord) -> AppResource | None:
        record: Final = _ResourceRecord.model_validate(row.model_dump())
        match resource_type:
            case AppResourceType.TEAM:
                return AppResource(
                    resource_type=resource_type,
                    resource_id=record.team_id or "",
                    alias=record.team_alias or record.team_id or "",
                )
            case AppResourceType.PROJECT:
                return AppResource(
                    resource_type=resource_type,
                    resource_id=record.project_id or "",
                    alias=record.project_alias or record.project_id or "",
                )
            case AppResourceType.VIRTUAL_KEY:
                key_id: Final = await self.ensure_key_id(record)
                if key_id is None:
                    return None
                return AppResource(
                    resource_type=resource_type,
                    resource_id=key_id,
                    alias=record.key_alias or key_id,
                )

    async def ensure_key_id(self, record: _ResourceRecord) -> str | None:
        if record.key_id is not None:
            return record.key_id
        candidate: Final = str(uuid4())
        await self.keys.update_many(where={"token": record.token, "key_id": None}, data={"key_id": candidate})
        persisted: Final = await self.keys.find_unique(where={"token": record.token})
        return _ResourceRecord.model_validate(persisted.model_dump()).key_id if persisted is not None else None

    async def key_resource_id(self, token_hash: str | None) -> str | None:
        if token_hash is None:
            return None
        row: Final = await self.keys.find_unique(where={"token": token_hash})
        return await self.ensure_key_id(_ResourceRecord.model_validate(row.model_dump())) if row is not None else None

    async def resource(self, resource_type: AppResourceType, resource_id: str) -> AppResource | AppFailure:
        table, identity = self.resource_table(resource_type)
        row: Final = await table.find_unique(where={identity: resource_id})
        resource: Final = await self.resource_response(resource_type, row) if row is not None else None
        return resource if resource is not None else AppFailure(404, "App resource not found.")

    async def resources(
        self, offset: int, limit: int, resource_type: AppResourceType | None = None
    ) -> tuple[AppResource, ...]:
        definition: Final = await self.definition()
        selected: Final = resource_type or definition.resource_type
        if selected is None:
            return ()
        table, identity = self.resource_table(selected)
        order: Final = "token" if selected == AppResourceType.VIRTUAL_KEY else identity
        rows: Final = await table.find_many(skip=offset, take=limit, order={order: "asc"})
        resolved: Final = await asyncio.gather(*(self.resource_response(selected, row) for row in rows))
        resources: Final = tuple(resource for resource in resolved if resource is not None)
        bindings: Final = await find_many_in(
            self.apps,
            "resource_id",
            (resource.resource_id for resource in resources),
            where={"resource_type": selected.value},
        )
        apps: Final = {
            app.resource_id: app.app_id for app in (App.model_validate(row.model_dump()) for row in bindings)
        }
        return tuple(resource.model_copy(update={"app_id": apps.get(resource.resource_id)}) for resource in resources)

    async def budget(self, budget_id: str | None) -> Budget | None:
        if budget_id is None:
            return None
        row: Final = await self.customers.budgets.find_unique(where={"budget_id": budget_id})
        return related_budget(row.model_dump(), self.now) if row is not None else None

    async def with_budget_usage(self, entity: AppEntity) -> AppEntity:
        identity: Final = _app_budget_identity(entity)
        usage: Final = await app_budget_usage(
            identity.key,
            identity.entity_id,
            identity.budget,
            entity.spend,
            self.customers.window_spend,
            self.now,
            self.customers.settings,
        )
        return entity.model_copy(update={"budget_usage": usage})

    async def valid_profile(self, budget_id: str | None) -> bool:
        if budget_id is None:
            return True
        return (
            budget_id in await organization_budget_group_ids(self.customers.config)
            and await self.budget(budget_id) is not None
        )

    async def get(self, app_id: str) -> App | AppFailure:
        row: Final = await self.apps.find_unique(where={"app_id": app_id})
        if row is None or not row.model_dump().get("resource_id"):
            return AppFailure(404, "App not found.")
        return await self.response(row)

    async def response(self, row: PrismaRecord) -> App:
        app: Final = App.model_validate(row.model_dump())
        resource: Final = await self.resource(app.resource_type, app.resource_id)
        return app.model_copy(
            update={
                "alias": (resource.alias if isinstance(resource, AppResource) else app.resource_id),
                "budget": await self.budget(app.budget_id),
                "default_customer_member_budget": await self.budget(app.default_customer_member_budget_id),
            }
        )

    async def by_resource(self, resource_type: AppResourceType, resource_id: str) -> App | None:
        row: Final = await self.apps.find_unique(
            where={
                "resource_type_resource_id": {
                    "resource_type": resource_type.value,
                    "resource_id": resource_id,
                }
            }
        )
        return await self.response(row) if row is not None else None

    async def list(self, offset: int, limit: int) -> tuple[App, ...]:
        definition: Final = await self.definition()
        if definition.resource_type is None:
            return ()
        rows: Final = await self.apps.find_many(
            where={"resource_type": definition.resource_type.value},
            skip=offset,
            take=limit,
            order={"app_id": "asc"},
        )
        return tuple(await asyncio.gather(*(self.response(row) for row in rows)))

    async def create(self, data: AppCreate, actor: str) -> App | AppFailure:
        from prisma import Json

        definition: Final = await self.lock_definition()
        if definition.resource_type is None:
            return AppFailure(
                409,
                "Choose the gateway app resource definition before configuring app budgets.",
            )
        if data.resource_type is not None and data.resource_type != definition.resource_type:
            return AppFailure(422, "The resource type must match the gateway app definition.")
        resource: Final = await self.resource(definition.resource_type, data.resource_id)
        if isinstance(resource, AppFailure):
            return resource
        if not await self.valid_profile(data.default_tier_budget_id):
            return AppFailure(422, "The default profile must be an existing customer budget group.")
        existing: Final = await self.by_resource(definition.resource_type, data.resource_id)
        if existing is not None:
            return existing
        values: Final = await self.fields(data, actor)
        identity: Final = {
            "resource_type": definition.resource_type.value,
            "resource_id": data.resource_id,
        }
        row: Final = await self.apps.upsert(
            where={"resource_type_resource_id": identity},
            data={
                "create": {
                    **identity,
                    "alias": f"{definition.resource_type.value}:{data.resource_id}",
                    "created_by": actor,
                    **values,
                },
                "update": {},
            },
        )
        await self.config.update(
            where={"param_name": _DEFINITION_PARAM},
            data={
                "param_value": Json(
                    {
                        "resource_type": definition.resource_type.value,
                        "locked": True,
                        "auto_create_customers_without_app": definition.auto_create_customers_without_app,
                    }
                )
            },
        )
        return await self.response(row)

    async def fields(self, data: AppPatch | AppCreate, actor: str, current: App | None = None) -> Mapping[str, object]:
        budget_id: Final = (
            await self.customers.update_budget(current.budget_id if current else None, data.budget, actor)
            if data.budget is not None
            else None
        )
        member_id: Final = (
            await self.customers.update_budget(
                current.default_customer_member_budget_id if current else None,
                data.default_customer_member_budget,
                actor,
            )
            if data.default_customer_member_budget is not None
            else None
        )
        return {
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
            **({"auto_create_customers": data.auto_create_customers} if data.auto_create_customers is not None else {}),
            **({"customer_margin": data.customer_margin} if "customer_margin" in data.model_fields_set else {}),
            "updated_by": actor,
        }

    async def writable(self, app_id: str) -> App | AppFailure:
        definition: Final = await self.lock_definition()
        if definition.resource_type is None:
            return AppFailure(
                409,
                "Choose the gateway app resource definition before configuring app budgets.",
            )
        app: Final = await self.get(app_id)
        if isinstance(app, AppFailure):
            return app
        if app.resource_type != definition.resource_type:
            return AppFailure(409, "The app binding does not match the gateway resource definition.")
        return app

    async def patch(self, app_id: str, data: AppPatch, actor: str) -> App | AppFailure:
        app: Final = await self.writable(app_id)
        if isinstance(app, AppFailure):
            return app
        if not await self.valid_profile(data.default_tier_budget_id):
            return AppFailure(422, "The default profile must be an existing customer budget group.")
        row: Final = await self.apps.update(where={"app_id": app_id}, data=await self.fields(data, actor, app))
        if row is None:
            return AppFailure(404, "App not found.")
        updated: Final = await self.response(row)
        await self.customers.delete_released_budget(app.budget_id, updated.budget_id)
        await self.customers.delete_released_budget(
            app.default_customer_member_budget_id,
            updated.default_customer_member_budget_id,
        )
        return updated

    async def find_organization(self, app: App, customer_org_id: str) -> AppOrganization | None:
        row: Final = await self.organizations.find_unique(
            where={"app_id_customer_org_id": {"app_id": app.app_id, "customer_org_id": customer_org_id}},
        )
        return await self.organization_response(app, row) if row is not None else None

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
        app: Final = await self.writable(app_id)
        if isinstance(app, AppFailure):
            return app
        if not await self.valid_profile(data.tier_budget_id):
            return AppFailure(
                422,
                "The profile must be an existing customer organization budget group.",
            )
        if await self.customers.organizations.find_unique(where={"customer_org_id": customer_org_id}) is None:
            return AppFailure(404, "Customer organization not found.")
        identity: Final = {"app_id": app_id, "customer_org_id": customer_org_id}
        previous: Final = await self.organizations.find_unique(where={"app_id_customer_org_id": identity})
        current: Final = AppOrganization.model_validate(previous.model_dump()) if previous is not None else None
        override: Final = (
            await self.customers.update_budget(
                current.budget_override_id if current else None,
                data.budget_override,
                actor,
            )
            if data.budget_override is not None
            else None
        )
        member: Final = (
            await self.customers.update_budget(
                current.customer_member_budget_id if current else None,
                data.customer_member_budget,
                actor,
            )
            if data.customer_member_budget is not None
            else None
        )
        fields: Final = {
            "tier_budget_id": data.tier_budget_id,
            "budget_override_id": override,
            "customer_member_budget_id": member,
        }
        row: Final = await self.organizations.upsert(
            where={"app_id_customer_org_id": identity},
            data={"create": {**identity, **fields}, "update": fields},
        )
        if current is not None:
            await self.customers.delete_released_budget(current.budget_override_id, override)
            await self.customers.delete_released_budget(current.customer_member_budget_id, member)
        return await self.organization_response(app, row)

    async def member(self, app: App, customer_member_id: str, org: AppOrganization) -> AppMember:
        identity: Final = {
            "app_id": app.app_id,
            "customer_member_id": customer_member_id,
        }
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
        return await self.member_response(app, member, org)

    async def member_response(self, app: App, member: AppMember, organization: AppOrganization | None) -> AppMember:
        inherited: Final = (
            organization.effective_customer_member_budget
            if organization is not None
            else app.default_customer_member_budget
        )
        organization_default_id: Final = organization.customer_member_budget_id if organization is not None else None
        return member.model_copy(
            update={
                "effective_budget": (
                    await self.budget(member.budget_override_id) if member.budget_override_id is not None else inherited
                ),
                "budget_source": self.member_budget_source(member, app, organization_default_id),
            }
        )

    @staticmethod
    def member_budget_source(member: AppMember, app: App, customer_default_id: str | None) -> AppMemberBudgetSource:
        if member.budget_override_id is not None:
            return AppMemberBudgetSource.MEMBER_OVERRIDE
        if customer_default_id is not None:
            return AppMemberBudgetSource.CUSTOMER_DEFAULT
        if app.default_customer_member_budget_id is not None:
            return AppMemberBudgetSource.APP_DEFAULT
        return AppMemberBudgetSource.NONE

    async def get_member(self, app_id: str, customer_member_id: str) -> AppMember | AppFailure:
        app: Final = await self.get(app_id)
        if isinstance(app, AppFailure):
            return app
        customer: Final = await self.customers.members.find_unique(where={"customer_member_id": customer_member_id})
        if customer is None:
            return AppFailure(404, "Customer member not found.")
        parent: Final = _PARENT_ORG_ID.validate_python(customer.model_dump()["parent_org_id"])
        organization: Final = await self.find_organization(app, parent)
        row: Final = await self.members.find_unique(
            where={
                "app_id_customer_member_id": {
                    "app_id": app_id,
                    "customer_member_id": customer_member_id,
                }
            },
        )
        member: Final = (
            AppMember.model_validate(row.model_dump())
            if row is not None
            else AppMember(app_id=app_id, customer_member_id=customer_member_id)
        )
        return await self.member_response(app, member, organization)

    async def put_member(
        self,
        app_id: str,
        customer_member_id: str,
        data: AppMemberPut,
        actor: str,
    ) -> AppMember | AppFailure:
        app: Final = await self.writable(app_id)
        if isinstance(app, AppFailure):
            return app
        customer: Final = await self.customers.members.find_unique(where={"customer_member_id": customer_member_id})
        if customer is None:
            return AppFailure(404, "Customer member not found.")
        parent: Final = _PARENT_ORG_ID.validate_python(customer.model_dump()["parent_org_id"])
        identity: Final = {"app_id": app_id, "customer_member_id": customer_member_id}
        previous: Final = await self.members.find_unique(where={"app_id_customer_member_id": identity})
        current: Final = AppMember.model_validate(previous.model_dump()) if previous is not None else None
        override: Final = (
            await self.customers.update_budget(
                current.budget_override_id if current else None,
                data.budget_override,
                actor,
            )
            if data.budget_override is not None
            else None
        )
        await self.members.upsert(
            where={"app_id_customer_member_id": identity},
            data={
                "create": {**identity, "budget_override_id": override},
                "update": {"budget_override_id": override},
            },
        )
        if current is not None:
            await self.customers.delete_released_budget(current.budget_override_id, override)
        return await self.member(app, customer_member_id, await self.organization(app, parent))
