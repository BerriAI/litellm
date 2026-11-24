import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    TypeVar,
    cast,  # noqa: TID251  # Prisma proxies implement the generated client surface
)
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm._internal_context import with_service_target
from litellm.constants import AUTH_OBJECTS_TARGET
from litellm.proxy.common_utils.cache_invalidation_pubsub import evict_and_broadcast
from litellm.proxy.common_utils.user_api_key_cache import get_management_object_ttl
from litellm.proxy.management_helpers.customer_budget_usage import app_budget_usage
from litellm.proxy.management_helpers.customer_hierarchy import (
    BudgetGroupScope,
    CustomerHierarchyStore,
    PrismaRecord,
    budget_group_ids,
    default_budget_group_id,
    effective_member_budget,
    organization_budget_group_ids,
    override_expiry,
    promotes_legacy_override,
    related_budget,
    utc_datetime,
)
from litellm.redis.stores.auth_cache import AuthObject
from litellm.repositories.app_customer_member_repository import AppCustomerMemberRepository
from litellm.repositories.app_customer_organization_repository import AppCustomerOrganizationRepository
from litellm.repositories.app_repository import AppRepository
from litellm.repositories.chunked_in import find_many_in
from litellm.repositories.prisma_protocols import TableActions
from litellm.repositories.table_repositories import PrismaTableRepository
from litellm.types.proxy.auth.customer_hierarchy import CustomerScopeKey
from litellm.types.proxy.management_endpoints.app_groups import AppGroup, AppGroupBudgetPut, AppGroupOrganization
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
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, MemberPatch

if TYPE_CHECKING:
    from prisma import Prisma

    from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache


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


class _CachedRow(BaseModel):
    model_config = ConfigDict(frozen=True)

    values: Mapping[str, object] | None


class _SharedProfiles(BaseModel):
    model_config = ConfigDict(frozen=True)

    ids: tuple[str, ...]


_DEFINITION_PARAM: Final = "customer_app_definition"
_DEFINITION: Final = AuthObject.customer_app_definition()
_PARENT_ORG_ID: Final = TypeAdapter(str)
AppEntity = TypeVar("AppEntity", App, AppOrganization, AppMember, AppGroup, AppGroupOrganization)
AppAssociation = TypeVar("AppAssociation", AppOrganization, AppMember)
CachedModel = TypeVar("CachedModel", bound=BaseModel)


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


class _OverrideFields(TypedDict):
    budget_override_id: ReadOnly[str | None]
    budget_override_expires_at: ReadOnly[datetime | None]


@dataclass(frozen=True, slots=True)
class _AssociationOverride:
    budget_override_id: str | None
    budget_override_expires_at: datetime | None

    def fields(self) -> _OverrideFields:
        return {
            "budget_override_id": self.budget_override_id,
            "budget_override_expires_at": self.budget_override_expires_at,
        }


@dataclass(frozen=True, slots=True)
class _AssociationBudget:
    permanent: Budget | None
    effective: Budget | None


def _dumped_fields(values: Mapping[str, object]) -> Mapping[str, object]:
    return values


def app_budget_selection(
    data: AppCreate | AppPatch | AppOrganizationPut | AppMemberPut | AppGroupBudgetPut,
    field: Literal["budget", "default_customer_member_budget", "customer_member_budget"] = "budget",
) -> MemberPatch:
    selected: Final = _dumped_fields(data.model_dump(include={field, f"{field}_id"}, exclude_unset=True))
    return MemberPatch.model_validate(
        {"budget" if name == field else "budget_id": value for name, value in selected.items()}
    )


def _association_base(current: AppAssociation | None, data: AppOrganizationPut | AppMemberPut) -> AppAssociation | None:
    patch: Final = MemberPatch.model_validate(
        data.model_dump(
            include={"budget", "budget_id", "budget_override", "budget_override_expires_at"}, exclude_unset=True
        )
    )
    if current is None or not promotes_legacy_override(current, patch):
        return current
    if isinstance(data, AppOrganizationPut) and data.tier_budget_id is not None:
        return current
    return current.model_copy(
        update={
            "budget_id": current.budget_override_id,
            "budget_override_id": None,
            "budget_override_expires_at": None,
        }
    )


def _app_budget_identity(
    entity: App | AppOrganization | AppMember | AppGroup | AppGroupOrganization,
) -> _AppBudgetIdentity:
    if isinstance(entity, App):
        return _AppBudgetIdentity("app", entity.app_id, entity.budget)
    if isinstance(entity, AppGroup):
        return _AppBudgetIdentity("app_group", entity.app_group_id, entity.budget)
    if isinstance(entity, AppGroupOrganization):
        return _AppBudgetIdentity(
            "app_group_customer_org", app_scope_id(entity.app_group_id, entity.customer_org_id), entity.budget
        )
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
        cache: "UserApiKeyCache | None" = None,
        *,
        cache_reads: bool = True,
    ) -> None:
        self.db: Final = cast("Prisma", db)  # cast-ok: Prisma wrappers share the generated client API
        self.now: Final = now or datetime.now(timezone.utc)
        self.cache: Final = cache
        self.cache_reads: Final = cache_reads
        self._definition_cache_dirty = False
        self._dirty_cache_keys: tuple[str, ...] = ()
        connection: Final = _Connection(db)
        self.apps: Final = AppRepository(connection).table
        self.organizations: Final = AppCustomerOrganizationRepository(connection).table
        self.members: Final = AppCustomerMemberRepository(connection).table
        self.config: Final = _ConfigRepository(connection).table
        self.teams: Final = _TeamRepository(connection).table
        self.projects: Final = _ProjectRepository(connection).table
        self.keys: Final = _KeyRepository(connection).table
        self.customers: Final = CustomerHierarchyStore(self.db, self.now, premium_user=True)

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def cached(
        self, key: AuthObject, model: type[CachedModel], load: Callable[[], Awaitable[CachedModel]]
    ) -> CachedModel:
        if self.cache is None or not self.cache_reads:
            return await load()
        cached: Final = await self.cache.async_get_cache(key=key, model_type=model)
        if cached is not None:
            return cached
        lease: Final = await self.cache.lease(key)
        value: Final = await load()
        if not isinstance(value, _CachedRow) or value.values is not None:
            _ = await self.cache.async_set_leased(lease, value, get_management_object_ttl(self.cache))
        return value

    async def cached_row(self, key: AuthObject, load: Callable[[], Awaitable[PrismaRecord | None]]) -> _CachedRow:
        async def read() -> _CachedRow:
            row: Final = await load()
            return _CachedRow(values=row.model_dump() if row is not None else None)

        return await self.cached(key, _CachedRow, read)

    async def invalidate_cache(self) -> None:
        await self.invalidate_definition_cache()
        if self.cache is not None:
            await evict_and_broadcast(
                tuple(dict.fromkeys((*self._dirty_cache_keys, *self.customers.auth_cache_keys))), self.cache
            )
        self._dirty_cache_keys = ()

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

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def definition(self) -> AppDefinition:
        if self.cache is not None and self.cache_reads:
            cached: Final = await self.cache.async_get_cache(key=_DEFINITION, model_type=AppDefinition)
            if cached is not None:
                return cached
            lease: Final = await self.cache.lease(_DEFINITION)
            definition: Final = await self._read_definition()
            _ = await self.cache.async_set_leased(lease, definition, get_management_object_ttl(self.cache))
            return definition
        return await self._read_definition()

    async def _read_definition(self) -> AppDefinition:
        row: Final = await self.config.find_unique(where={"param_name": _DEFINITION_PARAM})
        state: Final = self.definition_state(row)
        return AppDefinition(
            resource_type=state.resource_type,
            auto_create_customers_without_app=state.auto_create_customers_without_app,
        )

    @with_service_target(AUTH_OBJECTS_TARGET)
    async def invalidate_definition_cache(self) -> None:
        if not self._definition_cache_dirty:
            return
        self._definition_cache_dirty = False
        if self.cache is not None:
            await evict_and_broadcast((_DEFINITION.key,), self.cache)

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
        self._definition_cache_dirty = True
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

        async def read() -> _CachedRow:
            row: Final = await self.keys.find_unique(where={"token": token_hash})
            key_id: Final = (
                await self.ensure_key_id(_ResourceRecord.model_validate(row.model_dump())) if row is not None else None
            )
            return _CachedRow(values={"key_id": key_id} if key_id is not None else None)

        cached: Final = await self.cached(AuthObject.customer_app_key_resource(token_hash), _CachedRow, read)
        return _ResourceRecord.model_validate(cached.values).key_id if cached.values is not None else None

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
        return (await self.association_budget_state(budget_id)).effective

    async def association_budget_state(self, budget_id: str | None) -> _AssociationBudget:
        if budget_id is None:
            return _AssociationBudget(None, None)

        async def read() -> PrismaRecord | None:
            return await self.customers.budgets.find_unique(where={"budget_id": budget_id})

        row: Final = await self.cached_row(AuthObject.customer_app_budget(budget_id), read)
        return _AssociationBudget(related_budget(row.values), related_budget(row.values, self.now))

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

    async def shared_profile_ids(self) -> frozenset[str]:
        async def read() -> _SharedProfiles:
            return _SharedProfiles(
                ids=(
                    *await budget_group_ids(self.customers.config, BudgetGroupScope.APP_ORGANIZATION),
                    *await organization_budget_group_ids(self.customers.config),
                )
            )

        return frozenset((await self.cached(AuthObject.customer_app_shared_profiles(), _SharedProfiles, read)).ids)

    async def valid_profile(self, budget_id: str | None, scope: BudgetGroupScope | None = None) -> bool:
        if budget_id is None:
            return True
        groups: Final = (
            await budget_group_ids(self.customers.config, scope)
            if scope is not None
            else await self.shared_profile_ids()
        )
        return budget_id in groups and await self.budget(budget_id) is not None

    async def validate_member_profile(self, budget_id: str | None) -> AppFailure | None:
        if budget_id is None:
            return None
        await self.customers.lock_customer_identities()
        if not await self.valid_profile(budget_id, BudgetGroupScope.APP_MEMBER):
            return AppFailure(422, "The member budget must be an existing app × customer member budget group.")
        return None

    async def validate_settings(self, data: AppPatch | AppCreate) -> AppFailure | None:
        if data.app_group_id is not None:
            from litellm.proxy.management_helpers.app_groups import AppGroupRepository

            if await AppGroupRepository(self).table.find_unique(where={"app_group_id": data.app_group_id}) is None:
                return AppFailure(404, "App group not found.")
        if isinstance(data, AppCreate) or data.budget_id is not None or data.default_tier_budget_id is not None:
            await self.customers.lock_customer_identities()
        if not await self.valid_profile(data.default_tier_budget_id):
            return AppFailure(422, "The default profile must be an existing customer budget group.")
        if not await self.valid_profile(data.budget_id, BudgetGroupScope.APP):
            return AppFailure(422, "The app budget must be an existing overall-app budget group.")
        return await self.validate_member_profile(data.default_customer_member_budget_id)

    async def get(self, app_id: str) -> App | AppFailure:
        row: Final = await self.apps.find_unique(where={"app_id": app_id})
        if row is None or not row.model_dump().get("resource_id"):
            return AppFailure(404, "App not found.")
        return await self.response(row)

    async def response(self, row: PrismaRecord, resource_alias: str | None = None) -> App:
        app: Final = App.model_validate(row.model_dump())
        resource: Final = (
            await self.resource(app.resource_type, app.resource_id)
            if resource_alias is None
            else AppResource(resource_type=app.resource_type, resource_id=app.resource_id, alias=resource_alias)
        )
        return app.model_copy(
            update={
                "alias": (resource.alias if isinstance(resource, AppResource) else app.resource_id),
                "budget": await self.budget(app.budget_id),
                "default_customer_member_budget": await self.budget(app.default_customer_member_budget_id),
            }
        )

    async def by_resource(
        self, resource_type: AppResourceType, resource_id: str, resource_alias: str | None = None
    ) -> App | None:
        async def read() -> PrismaRecord | None:
            row: Final = await self.apps.find_unique(
                where={
                    "resource_type_resource_id": {
                        "resource_type": resource_type.value,
                        "resource_id": resource_id,
                    }
                }
            )
            if row is None:
                return None
            app: Final = App.model_validate(row.model_dump())
            if resource_alias is not None:
                return app.model_copy(update={"alias": resource_alias})
            resource: Final = await self.resource(resource_type, resource_id)
            return app.model_copy(
                update={"alias": resource.alias if isinstance(resource, AppResource) else resource_id}
            )

        row: Final = await self.cached_row(AuthObject.customer_app(resource_type.value, resource_id), read)
        if row.values is None:
            return None
        app: Final = App.model_validate(row.values)
        return await self.response(app, resource_alias if resource_alias is not None else app.alias)

    async def list(self, offset: int, limit: int, app_ids: frozenset[str] | None = None) -> tuple[App, ...]:
        definition: Final = await self.definition()
        if definition.resource_type is None:
            return ()
        rows: Final = await self.apps.find_many(
            where={
                "resource_type": definition.resource_type.value,
                # bounded-ok: the apps granted to one caller through object permissions, one id per app
                **({} if app_ids is None else {"app_id": {"in": sorted(app_ids)}}),
            },
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
        failure: Final = await self.validate_settings(data)
        if failure is not None:
            return failure
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
        self._dirty_cache_keys += (AuthObject.customer_app(definition.resource_type.value, data.resource_id).key,)
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

    async def linked_budget_id(
        self,
        current_id: str | None,
        data: MemberPatch,
        actor: str,
        scope: BudgetGroupScope,
        inherited_id: str | None = None,
        *,
        apply_default: bool = False,
    ) -> str | None:
        if data.budget_id is not None:
            return data.budget_id
        if "budget" not in data.model_fields_set:
            if "budget_id" in data.model_fields_set:
                return data.budget_id
            return await default_budget_group_id(self.config, scope) if apply_default else current_id
        return await self.customers.private_budget_patch(
            current_id,
            data,
            actor,
            source_id=inherited_id,
            shared_ids=(
                *await budget_group_ids(self.customers.config, scope),
                *((inherited_id,) if inherited_id is not None else ()),
            ),
        )

    async def fields(self, data: AppPatch | AppCreate, actor: str, current: App | None = None) -> Mapping[str, object]:
        budget_id: Final = await self.linked_budget_id(
            current.budget_id if current else None,
            app_budget_selection(data),
            actor,
            BudgetGroupScope.APP,
            apply_default=current is None,
        )
        member_id: Final = await self.linked_budget_id(
            current.default_customer_member_budget_id if current else None,
            app_budget_selection(data, "default_customer_member_budget"),
            actor,
            BudgetGroupScope.APP_MEMBER,
        )
        return {
            **({"budget_id": budget_id} if current is None or {"budget", "budget_id"} & data.model_fields_set else {}),
            **(
                {"default_tier_budget_id": data.default_tier_budget_id}
                if "default_tier_budget_id" in data.model_fields_set
                else {}
            ),
            **(
                {"default_customer_member_budget_id": member_id}
                if {"default_customer_member_budget", "default_customer_member_budget_id"} & data.model_fields_set
                else {}
            ),
            **({"app_group_id": data.app_group_id} if "app_group_id" in data.model_fields_set else {}),
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
        failure: Final = await self.validate_settings(data)
        if failure is not None:
            return failure
        row: Final = await self.apps.update(where={"app_id": app_id}, data=await self.fields(data, actor, app))
        if row is None:
            return AppFailure(404, "App not found.")
        updated: Final = await self.response(row)
        self._dirty_cache_keys += (AuthObject.customer_app(app.resource_type.value, app.resource_id).key,)
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

        async def read() -> PrismaRecord:
            existing: Final = await self.organizations.find_unique(where={"app_id_customer_org_id": identity})
            if existing is not None:
                return existing
            return await self.organizations.upsert(
                where={"app_id_customer_org_id": identity},
                data={"create": identity, "update": {"app_id": app.app_id}},
            )

        row: Final = await self.cached_row(
            AuthObject.customer_app_association("organization", app.app_id, customer_org_id), read
        )
        return await self.organization_response(app, AppOrganization.model_validate(row.values))

    async def organization_response(self, app: App, row: PrismaRecord) -> AppOrganization:
        org: Final = AppOrganization.model_validate(row.model_dump())
        base_id: Final = org.budget_id or org.tier_budget_id or app.default_tier_budget_id
        shared: Final = await self.shared_profile_ids()
        base: Final = await self.association_budget_state(base_id)
        override: Final = await self.association_budget_state(org.budget_override_id)
        return org.model_copy(
            update={
                "budget": base.permanent if org.budget_id is not None and org.budget_id not in shared else None,
                "budget_override": override.permanent,
                "effective_budget": self.association_effective_budget(org, base.effective, override.effective),
                "effective_customer_member_budget": await self.budget(
                    org.customer_member_budget_id or app.default_customer_member_budget_id
                ),
            }
        )

    def override_active(self, row: AppOrganization | AppMember) -> bool:
        return row.budget_override_id is not None and (
            row.budget_override_expires_at is None
            or utc_datetime(row.budget_override_expires_at) > utc_datetime(self.now)
        )

    def association_effective_budget(
        self, row: AppOrganization | AppMember, base: Budget | None, override: Budget | None
    ) -> Budget | None:
        if base is None and not self.override_active(row):
            return None
        # Legacy nonexpiring app overrides replace the complete policy rather than inherit unset fields.
        if override is not None and row.budget_override_expires_at is None:
            return override
        return effective_member_budget(base, override, row.budget_override_expires_at, self.now)

    async def association_budget(
        self, current_id: str | None, inherited_id: str | None, patch: Budget | None, actor: str
    ) -> str | None:
        shared: Final = await self.shared_profile_ids() | (
            frozenset((inherited_id,)) if inherited_id is not None else frozenset[str]()
        )
        return await self.customers.private_budget_patch(
            current_id,
            MemberPatch(budget=patch),
            actor,
            source_id=inherited_id,
            shared_ids=shared,
        )

    async def association_override(
        self, current: AppOrganization | AppMember | None, data: AppOrganizationPut | AppMemberPut, actor: str
    ) -> _AssociationOverride:
        previous: Final = current.budget_override_id if current is not None else None
        patch: Final = MemberPatch.model_validate(
            data.model_dump(include={"budget_override", "budget_override_expires_at"}, exclude_unset=True)
        )
        override: Final = await self.customers.override_patch(previous, patch, actor)
        return _AssociationOverride(override, override_expiry(current, patch, override))

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
        invalid_member: Final = await self.validate_member_profile(data.customer_member_budget_id)
        if invalid_member is not None:
            return invalid_member
        selected: Final = data.budget_id or data.tier_budget_id
        if not await self.valid_profile(selected):
            return AppFailure(
                422,
                "The profile must be an existing customer organization budget group.",
            )
        if await self.customers.organizations.find_unique(where={"customer_org_id": customer_org_id}) is None:
            return AppFailure(404, "Customer organization not found.")
        await self.customers.lock_organization(customer_org_id)
        await self.db.query_raw(
            'SELECT app_id FROM "LiteLLM_AppCustomerOrgTable" WHERE app_id = $1 AND customer_org_id = $2 FOR UPDATE',
            app_id,
            customer_org_id,
        )
        identity: Final = {"app_id": app_id, "customer_org_id": customer_org_id}
        previous: Final = await self.organizations.find_unique(where={"app_id_customer_org_id": identity})
        original: Final = AppOrganization.model_validate(previous.model_dump()) if previous is not None else None
        current: Final = _association_base(original, data)
        current_id: Final = (current.budget_id or current.tier_budget_id) if current else None
        inherited_id: Final = (current.tier_budget_id if current else None) or app.default_tier_budget_id
        base_id: Final = (
            await self.association_budget(current_id, inherited_id, data.budget, actor)
            if "budget" in data.model_fields_set and selected is None
            else selected
            if {"budget_id", "tier_budget_id"} & data.model_fields_set
            else current_id
        )
        override: Final = await self.association_override(current, data, actor)
        member: Final = await self.linked_budget_id(
            current.customer_member_budget_id if current else None,
            app_budget_selection(data, "customer_member_budget"),
            actor,
            BudgetGroupScope.APP_MEMBER,
            app.default_customer_member_budget_id,
        )
        fields: Final = {
            "tier_budget_id": None,
            "budget_id": base_id,
            **override.fields(),
            "customer_member_budget_id": member,
        }
        row: Final = await self.organizations.upsert(
            where={"app_id_customer_org_id": identity},
            data={"create": {**identity, **fields}, "update": fields},
        )
        self._dirty_cache_keys += (AuthObject.customer_app_association("organization", app_id, customer_org_id).key,)
        if original is not None:
            await self.customers.delete_released_budget(original.budget_id or original.tier_budget_id, base_id)
            await self.customers.delete_released_budget(original.budget_override_id, override.budget_override_id)
            await self.customers.delete_released_budget(original.customer_member_budget_id, member)
        return await self.organization_response(app, row)

    async def member(self, app: App, customer_member_id: str, org: AppOrganization) -> AppMember:
        identity: Final = {
            "app_id": app.app_id,
            "customer_member_id": customer_member_id,
        }

        async def read() -> PrismaRecord:
            existing: Final = await self.members.find_unique(where={"app_id_customer_member_id": identity})
            if existing is not None:
                return existing
            return await self.members.upsert(
                where={"app_id_customer_member_id": identity},
                data={"create": identity, "update": {"app_id": app.app_id}},
            )

        row: Final = await self.cached_row(
            AuthObject.customer_app_association("member", app.app_id, customer_member_id), read
        )
        member: Final = AppMember.model_validate(row.values)
        return await self.member_response(app, member, org)

    async def member_response(self, app: App, member: AppMember, organization: AppOrganization | None) -> AppMember:
        inherited: Final = (
            organization.effective_customer_member_budget
            if organization is not None
            else app.default_customer_member_budget
        )
        organization_default_id: Final = organization.customer_member_budget_id if organization is not None else None
        base: Final = await self.association_budget_state(member.budget_id)
        override: Final = await self.association_budget_state(member.budget_override_id)
        return member.model_copy(
            update={
                "budget": base.permanent,
                "budget_override": override.permanent,
                "effective_budget": self.association_effective_budget(
                    member, base.effective if member.budget_id is not None else inherited, override.effective
                ),
                "budget_source": self.member_budget_source(member, app, organization_default_id),
            }
        )

    def member_budget_source(
        self, member: AppMember, app: App, customer_default_id: str | None
    ) -> AppMemberBudgetSource:
        if self.override_active(member):
            return AppMemberBudgetSource.MEMBER_OVERRIDE
        if member.budget_id is not None:
            return AppMemberBudgetSource.MEMBER_BUDGET
        if customer_default_id is not None:
            return AppMemberBudgetSource.CUSTOMER_DEFAULT
        if app.default_customer_member_budget_id is not None:
            return AppMemberBudgetSource.APP_DEFAULT
        return AppMemberBudgetSource.NONE

    async def get_member(
        self, app_id: str, customer_member_id: str, known_to_app_only: bool = False
    ) -> AppMember | AppFailure:
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
        if known_to_app_only and row is None and organization is None:
            return AppFailure(404, "Customer member not found.")
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
        invalid_member: Final = await self.validate_member_profile(data.budget_id)
        if invalid_member is not None:
            return invalid_member
        customer: Final = await self.customers.members.find_unique(where={"customer_member_id": customer_member_id})
        if customer is None:
            return AppFailure(404, "Customer member not found.")
        parent: Final = _PARENT_ORG_ID.validate_python(customer.model_dump()["parent_org_id"])
        await self.customers.lock_organization(parent)
        await self.db.query_raw(
            'SELECT app_id FROM "LiteLLM_AppCustomerMemberTable" WHERE app_id = $1 AND customer_member_id = $2 FOR UPDATE',
            app_id,
            customer_member_id,
        )
        identity: Final = {"app_id": app_id, "customer_member_id": customer_member_id}
        previous: Final = await self.members.find_unique(where={"app_id_customer_member_id": identity})
        original: Final = AppMember.model_validate(previous.model_dump()) if previous is not None else None
        current: Final = _association_base(original, data)
        organization: Final = await self.organization(app, parent)
        inherited_id: Final = organization.customer_member_budget_id or app.default_customer_member_budget_id
        base_id: Final = await self.linked_budget_id(
            current.budget_id if current else None,
            app_budget_selection(data),
            actor,
            BudgetGroupScope.APP_MEMBER,
            inherited_id,
        )
        override: Final = await self.association_override(current, data, actor)
        fields: Final = {"budget_id": base_id, **override.fields()}
        await self.members.upsert(
            where={"app_id_customer_member_id": identity},
            data={
                "create": {**identity, **fields},
                "update": fields,
            },
        )
        self._dirty_cache_keys += (AuthObject.customer_app_association("member", app_id, customer_member_id).key,)
        if original is not None:
            await self.customers.delete_released_budget(original.budget_id, base_id)
            await self.customers.delete_released_budget(original.budget_override_id, override.budget_override_id)
        return await self.member(app, customer_member_id, organization)
