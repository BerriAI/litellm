from collections.abc import AsyncGenerator, Mapping, Sequence
from contextlib import asynccontextmanager
from functools import partial
from typing import Final
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_helpers.apps import AppStore, app_scope_id, app_scope_identity
from litellm.proxy.management_helpers.customer_hierarchy import BudgetGroupScope, ErrorCode, Failure
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
    AppResourceType,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    BudgetGroupsUpdate,
    BudgetTemplate,
    MemberPatch,
)
from tests.unit.proxy.management_helpers.test_customer_hierarchy import (
    NOW,
    Database,
    Record,
    Table,
)


class AppTable(Table):
    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> Record:
        return await super().create({"app_id": str(uuid4()), **data}, include)

    async def find_unique(
        self,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> Record | None:
        nested: Final = where.get("resource_type_resource_id")
        criteria: Final = TypeAdapter(Mapping[str, object]).validate_python(nested) if nested is not None else where
        return next((row for row in self.rows.values() if self.matches(row, criteria)), None)

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> Record | None:
        row: Final = await self.find_unique(where)
        if row is None:
            return None
        return await super().update({self.id_field: row.model_dump()[self.id_field]}, data)

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        include: Mapping[str, object] | None = None,
        skip: int = 0,
        take: int | None = None,
    ) -> tuple[Record, ...]:
        return await super().find_many(
            where,
            (order,) if isinstance(order, Mapping) else order,
            include,
            skip,
            take,
        )


class CompositeTable(AppTable):
    async def find_unique(
        self,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> Record | None:
        identity: Final = TypeAdapter(Mapping[str, str]).validate_python(where[self.id_field])
        return next(
            (
                row
                for row in self.rows.values()
                if all(row.model_dump().get(key) == value for key, value in identity.items())
            ),
            None,
        )

    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> Record:
        customer_column: Final = self.id_field.removeprefix("app_id_")
        return await super().create(
            {
                **data,
                self.id_field: {
                    "app_id": data["app_id"],
                    customer_column: data[customer_column],
                },
            },
            include,
        )

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> Record | None:
        return await Table.update(self, where, data)


class AppConfigTable(Table):
    def __init__(self, database: Database) -> None:
        super().__init__("param_name", database)
        self.read_error: Exception | None = None

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> Record | None:
        self.read_count += 1
        if self.read_error is not None:
            raise self.read_error
        return await super().find_unique(where, include)


class AppDatabase(Database):
    def __init__(self) -> None:
        super().__init__()
        self.committed = False
        self.litellm_config = AppConfigTable(self)
        self.litellm_apptable: Final = AppTable("app_id", self)
        self.litellm_appcustomerorgtable: Final = CompositeTable("app_id_customer_org_id", self)
        self.litellm_appcustomermembertable: Final = CompositeTable("app_id_customer_member_id", self)
        self.litellm_teamtable: Final = AppTable("team_id", self)
        self.litellm_projecttable = AppTable("project_id", self)
        self.litellm_verificationtoken = AppTable("token", self)
        self.litellm_usertable: Final = AppTable("user_id", self)
        self.litellm_organizationtable: Final = AppTable("organization_id", self)
        self.litellm_organizationmembership: Final = AppTable("organization_id", self)
        self.litellm_objectpermissiontable: Final = AppTable("object_permission_id", self)

    @asynccontextmanager
    async def tx(self) -> AsyncGenerator["AppDatabase", None]:
        async with super().tx():
            yield self
        self.committed = True

    def has_budget_reference(self, budget_id: str, relation: str) -> bool:
        references: Final = {
            "apps": (self.litellm_apptable, "budget_id"),
            "app_default_tiers": (self.litellm_apptable, "default_tier_budget_id"),
            "app_default_members": (
                self.litellm_apptable,
                "default_customer_member_budget_id",
            ),
            "app_org_tiers": (self.litellm_appcustomerorgtable, "tier_budget_id"),
            "app_org_budgets": (self.litellm_appcustomerorgtable, "budget_id"),
            "app_member_budgets": (self.litellm_appcustomermembertable, "budget_id"),
            "app_org_overrides": (
                self.litellm_appcustomerorgtable,
                "budget_override_id",
            ),
            "app_org_members": (
                self.litellm_appcustomerorgtable,
                "customer_member_budget_id",
            ),
            "app_member_overrides": (
                self.litellm_appcustomermembertable,
                "budget_override_id",
            ),
        }
        reference: Final = references.get(relation)
        if reference is None:
            return super().has_budget_reference(budget_id, relation)
        table, field = reference
        return any(row.model_dump().get(field) == budget_id for row in table.rows.values())


class AppPrismaClient:
    def __init__(self, db: AppDatabase) -> None:
        self.db: Final = db


class CommitAwareCache(UserApiKeyCache):
    def __init__(self, database: AppDatabase) -> None:
        super().__init__()
        self.database: Final = database
        self.invalidated_after_commit: bool | None = None

    async def async_delete_cache(self, key: str) -> None:
        self.invalidated_after_commit = self.database.committed
        await super().async_delete_cache(key)


async def seed_resources(database: AppDatabase) -> None:
    await database.litellm_teamtable.create({"team_id": "chat", "team_alias": "Chat"})
    await database.litellm_teamtable.create({"team_id": "search", "team_alias": "Search"})
    await database.litellm_projecttable.create({"project_id": "project", "project_alias": "Project"})
    await database.litellm_verificationtoken.create({"token": "token-a", "key_alias": "Service key", "key_id": None})


async def profile(
    store: AppStore, name: str, maximum: float, scope: BudgetGroupScope = BudgetGroupScope.ORGANIZATION
) -> str:
    from prisma import Json

    budget_id: Final = await store.customers.create_budget(BudgetTemplate(name=name, max_budget=maximum), "admin", name)
    current: Final = await store.customers.config.find_unique(where={"param_name": scope.value})
    ids: Final = TypeAdapter(tuple[str, ...]).validate_python(current.model_dump()["param_value"]) if current else ()
    await store.customers.config.upsert(
        where={"param_name": scope.value},
        data={
            "create": {
                "param_name": scope.value,
                "param_value": Json([budget_id]),
            },
            "update": {"param_value": Json([*ids, budget_id])},
        },
    )
    return budget_id


@pytest.fixture
async def app_store() -> tuple[AppDatabase, AppStore, App]:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    await database.litellm_customerorganizationtable.create({"customer_org_id": "org"})
    await database.litellm_customermembertable.create({"customer_member_id": "member", "parent_org_id": "org"})
    app: Final = await store.create(AppCreate(resource_id="chat"), "admin")
    assert isinstance(app, App)
    return database, store, app


@pytest.mark.asyncio
async def test_apps_are_opt_in_and_discovery_does_not_configure_a_resource() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await seed_resources(database)
    assert await store.definition() == AppDefinition()
    assert await store.resources(0, 100) == ()
    assert await store.list(0, 100) == ()
    preview: Final = await store.resources(0, 100, AppResourceType.TEAM)
    assert tuple(resource.resource_id for resource in preview) == ("chat", "search")
    assert all(resource.app_id is None for resource in preview)
    rejected: Final = await store.create(AppCreate(resource_id="chat"), "admin")
    assert isinstance(rejected, AppFailure) and rejected.status == 409
    assert not database.litellm_apptable.rows
    assert await store.definition() == AppDefinition()


@pytest.mark.asyncio
async def test_definition_cache_caches_an_absent_config() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW, cache=UserApiKeyCache())

    assert await store.definition() == AppDefinition()
    assert await store.definition() == AppDefinition()
    assert database.litellm_config.read_count == 1


@pytest.mark.asyncio
async def test_definition_cache_is_invalidated_after_a_config_update() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW, cache=UserApiKeyCache())
    assert await store.definition() == AppDefinition()

    stored: Final = await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    assert stored == AppDefinition(resource_type=AppResourceType.TEAM)
    await store.invalidate_definition_cache()

    assert await store.definition() == AppDefinition(resource_type=AppResourceType.TEAM)
    assert await store.definition() == AppDefinition(resource_type=AppResourceType.TEAM)
    assert database.litellm_config.read_count == 3


@pytest.mark.asyncio
async def test_app_store_dependency_invalidates_after_commit(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.proxy import proxy_server
    from litellm.proxy.management_endpoints.management_v1.apps import get_app_store

    database: Final = AppDatabase()
    cache: Final = CommitAwareCache(database)
    monkeypatch.setattr(proxy_server, "prisma_client", AppPrismaClient(database))
    monkeypatch.setattr(proxy_server, "user_api_key_cache", cache)
    await AppStore(database, NOW, cache=cache).definition()

    dependency: Final = get_app_store()
    store: Final = await anext(dependency)
    definition: Final = AppDefinition(resource_type=AppResourceType.TEAM)
    assert await store.put_definition(definition) == definition

    with pytest.raises(StopAsyncIteration):
        await anext(dependency)

    assert cache.invalidated_after_commit is True
    assert await AppStore(database, NOW, cache=cache).definition() == definition


@pytest.mark.asyncio
async def test_definition_cache_does_not_cache_database_errors() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW, cache=UserApiKeyCache())
    database.litellm_config.read_error = RuntimeError("database read failed")

    with pytest.raises(RuntimeError, match="database read failed"):
        await store.definition()

    database.litellm_config.read_error = None
    assert await store.definition() == AppDefinition()
    assert database.litellm_config.read_count == 2


@pytest.mark.asyncio
async def test_binding_is_idempotent_and_locks_the_gateway_definition(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    database, store, app = app_store
    again: Final = await store.create(AppCreate(resource_id="chat"), "admin")
    assert isinstance(again, App) and again.app_id == app.app_id
    assert app.resource_type == AppResourceType.TEAM and app.resource_id == "chat" and app.alias == "Chat"
    rejected: Final = await store.put_definition(AppDefinition(resource_type=AppResourceType.PROJECT))
    assert isinstance(rejected, AppFailure) and rejected.status == 409
    same: Final = await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    assert same == AppDefinition(resource_type=AppResourceType.TEAM)
    await database.litellm_apptable.delete(where={"app_id": app.app_id})
    disabled: Final = await store.put_definition(AppDefinition())
    assert isinstance(disabled, AppFailure) and disabled.status == 409


@pytest.mark.asyncio
async def test_unknown_and_wrong_type_resources_cannot_create_policy_records(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    _, store, _ = app_store
    wrong_type: Final = await store.create(
        AppCreate(resource_id="project", resource_type=AppResourceType.PROJECT), "admin"
    )
    missing: Final = await store.create(AppCreate(resource_id="unknown"), "admin")
    assert isinstance(wrong_type, AppFailure) and wrong_type.status == 422
    assert isinstance(missing, AppFailure) and missing.status == 404
    assert len(await store.list(0, 100)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("selector", ("tier_budget_id", "budget_id"))
async def test_org_override_wins_then_shared_profile_then_default(
    app_store: tuple[AppDatabase, AppStore, App],
    selector: str,
) -> None:
    _, store, app = app_store
    default: Final = await profile(store, "Standard", 10)
    selected: Final = await profile(store, "Pro", 20)
    configured: Final = await store.patch(app.app_id, AppPatch(default_tier_budget_id=default), "admin")
    assert isinstance(configured, App)
    inherited: Final = await store.organization(configured, "org")
    assert inherited.effective_budget == Budget(max_budget=10, allowed_models=())
    assigned: Final = await store.put_organization(
        app.app_id, "org", AppOrganizationPut.model_validate({selector: selected, "budget": None}), "admin"
    )
    assert isinstance(assigned, AppOrganization) and assigned.effective_budget == Budget(
        max_budget=20, allowed_models=()
    )
    overridden: Final = await store.put_organization(
        app.app_id,
        "org",
        AppOrganizationPut(tier_budget_id=selected, budget_override=Budget(max_budget=3)),
        "admin",
    )
    assert isinstance(overridden, AppOrganization) and overridden.effective_budget == Budget(
        max_budget=3, allowed_models=()
    )


@pytest.mark.asyncio
async def test_shared_profile_edits_are_inherited_without_pooling_app_spend(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    database, store, chat = app_store
    group: Final = await profile(store, "Standard", 10)
    search: Final = await store.create(AppCreate(resource_id="search"), "admin")
    assert isinstance(search, App)
    await store.put_organization(chat.app_id, "org", AppOrganizationPut(tier_budget_id=group), "admin")
    await store.put_organization(search.app_id, "org", AppOrganizationPut(tier_budget_id=group), "admin")
    identity: Final = {"app_id": chat.app_id, "customer_org_id": "org"}
    await database.litellm_appcustomerorgtable.upsert(
        where={"app_id_customer_org_id": identity},
        data={"create": identity, "update": {"spend": 4}},
    )
    await database.litellm_budgettable.update(where={"budget_id": group}, data={"max_budget": 30})
    chat_budget: Final = await store.organization(chat, "org")
    search_budget: Final = await store.organization(search, "org")
    assert chat_budget.effective_budget == search_budget.effective_budget == Budget(max_budget=30, allowed_models=())
    assert chat_budget.spend == 4 and search_budget.spend == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("assignment", ("default", "customer", "app", "member-default", "customer-members", "member"))
async def test_profiles_in_use_by_apps_cannot_be_removed(
    app_store: tuple[AppDatabase, AppStore, App], assignment: str
) -> None:
    database, store, app = app_store
    scope: Final = (
        BudgetGroupScope.APP_MEMBER
        if assignment in ("member-default", "customer-members", "member")
        else BudgetGroupScope.APP
        if assignment == "app"
        else BudgetGroupScope.ORGANIZATION
    )
    group: Final = await profile(store, "Standard", 10, scope)
    if assignment == "app":
        await store.patch(app.app_id, AppPatch(budget_id=group), "admin")
    elif assignment == "default":
        await store.patch(app.app_id, AppPatch(default_tier_budget_id=group), "admin")
    elif assignment == "member-default":
        await store.patch(app.app_id, AppPatch(default_customer_member_budget_id=group), "admin")
    elif assignment == "customer-members":
        await store.put_organization(app.app_id, "org", AppOrganizationPut(customer_member_budget_id=group), "admin")
    elif assignment == "member":
        await store.put_member(app.app_id, "member", AppMemberPut(budget_id=group), "admin")
    else:
        await store.put_organization(app.app_id, "org", AppOrganizationPut(tier_budget_id=group), "admin")
    groups: Final = await store.customers.get_budget_groups(None, scope=scope)
    assert not isinstance(groups, Failure)
    rejected: Final = await store.customers.configure_templates(
        None,
        BudgetGroupsUpdate(
            expected_revision=groups.budget_groups_revision,
            default_budgets=(),
            default_budget_template=None,
        ),
        "admin",
        scope=scope,
    )
    assert rejected == Failure(ErrorCode.TEMPLATE_IN_USE)
    assert group in database.litellm_budgettable.rows


@pytest.mark.asyncio
async def test_app_presets_stay_linked_until_customized_without_changing_other_apps(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    database, store, app = app_store
    group: Final = await profile(store, "App Standard", 10, BudgetGroupScope.APP)
    await store.customers.update_budget(group, Budget(rpm_limit=4, budget_duration="1mo"), "admin")
    linked: Final = await store.patch(app.app_id, AppPatch(budget_id=group), "admin")
    second: Final = await store.create(AppCreate(resource_id="search", budget_id=group), "admin")
    assert isinstance(linked, App) and isinstance(second, App) and linked.budget_id == second.budget_id == group
    await store.customers.update_budget(group, Budget(max_budget=20), "admin")
    loaded: Final = await store.get(app.app_id)
    other: Final = await store.get(second.app_id)
    expected: Final = Budget(max_budget=20, rpm_limit=4, budget_duration="1mo", allowed_models=())
    assert isinstance(loaded, App) and isinstance(other, App) and loaded.budget == other.budget == expected
    unchanged: Final = await store.patch(app.app_id, AppPatch(customer_margin=0.1), "admin")
    assert isinstance(unchanged, App) and unchanged.budget_id == group
    forked: Final = await store.patch(app.app_id, AppPatch(budget=Budget(max_budget=30)), "admin")
    assert isinstance(forked, App) and forked.budget_id is not None and forked.budget_id != group
    assert forked.budget == expected.model_copy(update={"max_budget": 30})
    assert await store.budget(group) == expected
    preset_row: Final = database.litellm_budgettable.rows[group].model_dump()
    custom_row: Final = database.litellm_budgettable.rows[forked.budget_id].model_dump()
    assert custom_row["budget_reset_at"] == preset_row["budget_reset_at"]
    cleared: Final = await store.patch(app.app_id, AppPatch(budget=Budget(max_budget=None)), "admin")
    assert isinstance(cleared, App) and cleared.budget_id == forked.budget_id
    assert cleared.budget == expected.model_copy(update={"max_budget": None})
    restored: Final = await store.patch(app.app_id, AppPatch(budget_id=group), "admin")
    assert isinstance(restored, App) and restored.budget_id == group and await store.budget(forked.budget_id) is None
    unlimited: Final = await store.patch(app.app_id, AppPatch(budget_id=None), "admin")
    assert isinstance(unlimited, App) and unlimited.budget_id is None and unlimited.budget is None
    await store.patch(second.app_id, AppPatch(budget_id=None), "admin")
    assert await store.budget(group) == expected


@pytest.mark.asyncio
async def test_member_override_is_scoped_and_can_return_to_the_org_default(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    _, store, app = app_store
    configured: Final = await store.patch(
        app.app_id,
        AppPatch(default_customer_member_budget=Budget(max_budget=10)),
        "admin",
    )
    assert isinstance(configured, App)
    org: Final = await store.put_organization(
        app.app_id,
        "org",
        AppOrganizationPut(customer_member_budget=Budget(max_budget=5)),
        "admin",
    )
    assert isinstance(org, AppOrganization)
    unused: Final = await store.get_member(app.app_id, "member")
    assert isinstance(unused, AppMember) and unused.effective_budget == Budget(max_budget=5, allowed_models=())
    assert unused.budget_source == AppMemberBudgetSource.CUSTOMER_DEFAULT
    override: Final = await store.put_member(
        app.app_id,
        "member",
        AppMemberPut(budget_override=Budget(max_budget=1)),
        "admin",
    )
    assert isinstance(override, AppMember) and override.effective_budget == Budget(max_budget=1, allowed_models=())
    assert override.budget_source == AppMemberBudgetSource.MEMBER_OVERRIDE
    restored: Final = await store.put_member(app.app_id, "member", AppMemberPut(budget_override=None), "admin")
    assert isinstance(restored, AppMember) and restored.effective_budget == Budget(max_budget=5, allowed_models=())


@pytest.mark.asyncio
async def test_arbitrary_budget_rows_are_not_profiles(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    _, store, app = app_store
    private_budget: Final = await store.customers.create_budget(Budget(max_budget=10), "admin")
    assigned: Final = await store.put_organization(
        app.app_id, "org", AppOrganizationPut(tier_budget_id=private_budget), "admin"
    )
    default: Final = await store.patch(app.app_id, AppPatch(default_tier_budget_id=private_budget), "admin")
    created: Final = await store.create(AppCreate(resource_id="search", default_tier_budget_id=private_budget), "admin")
    assert isinstance(assigned, AppFailure) and assigned.status == 422
    assert isinstance(default, AppFailure) and default.status == 422
    assert isinstance(created, AppFailure) and created.status == 422
    assert await store.by_resource(AppResourceType.TEAM, "search") is None


@pytest.mark.asyncio
async def test_virtual_key_discovery_uses_a_nonsecret_identity_that_survives_rotation() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await seed_resources(database)
    resources: Final = await store.resources(0, 100, AppResourceType.VIRTUAL_KEY)
    resource: Final = resources[0]
    assert resource.resource_id and resource.resource_id != "token-a"
    assert "token-a" not in resource.model_dump_json() and "token" not in resource.model_dump()
    await store.put_definition(AppDefinition(resource_type=AppResourceType.VIRTUAL_KEY))
    bound: Final = await store.create(AppCreate(resource_id=resource.resource_id), "admin")
    assert isinstance(bound, App)
    await database.litellm_verificationtoken.update(where={"token": "token-a"}, data={"token": "rotated-token"})
    assert await store.key_resource_id("rotated-token") == resource.resource_id
    rotated: Final = await store.resources(0, 100)
    assert rotated[0].resource_id == resource.resource_id and rotated[0].app_id == bound.app_id
    assert await store.key_resource_id("token-a") is None


@pytest.mark.asyncio
@pytest.mark.parametrize("duration", ("30s", "1m", "1h", "24h", "7d", "1mo", "12mo"))
async def test_app_period_usage_reads_the_current_window_instead_of_lifetime_spend(
    app_store: tuple[AppDatabase, AppStore, App], duration: str
) -> None:
    from datetime import timedelta

    from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters
    from litellm.types.proxy.auth.customer_hierarchy import app_budget_limits

    database, store, app = app_store
    configured: Final = await store.patch(
        app.app_id,
        AppPatch(budget=Budget(max_budget=10, budget_duration=duration)),
        "admin",
    )
    assert isinstance(configured, App)
    await database.litellm_apptable.update(where={"app_id": app.app_id}, data={"spend": 1000})
    counter: Final = customer_window_counters(
        "app",
        app.app_id,
        app_budget_limits(configured.budget),
        now=NOW,
        reset_settings=store.customers.settings,
    )[0]
    database.litellm_budgetwindowspend.rows = {
        "current": Record(
            entity_type="app",
            entity_id=app.app_id,
            window_duration=duration,
            window_start=counter.window_start,
            spend=7,
        ),
    }
    current: Final = await store.get(app.app_id)
    assert isinstance(current, App)
    result: Final = await store.with_budget_usage(current)
    assert result.spend == 1000 and result.budget_usage is not None
    assert len(result.budget_usage) == 1 and result.budget_usage[0].spend == 7
    assert result.budget_usage[0].source == "window" and result.budget_usage[0].max_budget == 10
    assert result.budget_usage[0].window_start == counter.window_start
    assert result.budget_usage[0].reset_at == counter.reset_at
    next_period: Final = AppStore(database, NOW + timedelta(days=400))
    reset: Final = await next_period.with_budget_usage(current)
    assert reset.spend == 1000 and reset.budget_usage is not None
    assert reset.budget_usage[0].spend == 0


@pytest.mark.asyncio
async def test_scoped_usage_reports_all_effective_windows_and_keeps_app_balances_independent(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    from litellm.models.budget import BudgetLimit
    from litellm.proxy.spend_tracking.budget_reservation import customer_window_counters

    database, store, chat = app_store
    budget: Final = Budget(
        budget_limits=(
            BudgetLimit(budget_duration="1mo", max_budget=10),
            BudgetLimit(budget_duration="12mo", max_budget=100),
        )
    )
    chat_org: Final = await store.put_organization(
        chat.app_id, "org", AppOrganizationPut(budget_override=budget), "admin"
    )
    search: Final = await store.create(AppCreate(resource_id="search"), "admin")
    assert isinstance(chat_org, AppOrganization) and isinstance(search, App)
    search_org: Final = await store.put_organization(
        search.app_id, "org", AppOrganizationPut(budget_override=budget), "admin"
    )
    assert isinstance(search_org, AppOrganization)
    identity: Final = f"{chat.app_id}:org"
    counters: Final = customer_window_counters(
        "app_customer_org",
        identity,
        budget.budget_limits or (),
        now=NOW,
        reset_settings=store.customers.settings,
    )
    database.litellm_budgetwindowspend.rows = {
        counter.window_duration or "": Record(
            entity_type="app_customer_org",
            entity_id=identity,
            window_duration=counter.window_duration,
            window_start=counter.window_start,
            spend=amount,
        )
        for counter, amount in zip(counters, (7, 62), strict=True)
    }
    chat_usage: Final = await store.with_budget_usage(chat_org)
    search_usage: Final = await store.with_budget_usage(search_org)
    assert chat_usage.budget_usage is not None and search_usage.budget_usage is not None
    assert tuple((usage.spend, usage.max_budget) for usage in chat_usage.budget_usage) == ((7, 10), (62, 100))
    assert tuple(usage.spend for usage in search_usage.budget_usage) == (0, 0)


def test_should_preserve_customer_identifier_colons_in_app_scope_keys() -> None:
    assert app_scope_identity(app_scope_id("chat", "customer:external")) == ("chat", "customer:external")


@pytest.mark.asyncio
@pytest.mark.parametrize("state, status", (("unset", 409), ("missing", 404), ("mismatched", 409)))
async def test_should_reject_budget_changes_when_the_app_binding_is_unavailable(
    app_store: tuple[AppDatabase, AppStore, App], state: str, status: int
) -> None:
    database, store, app = app_store
    if state == "unset":
        await database.litellm_config.delete(where={"param_name": "customer_app_definition"})
    elif state == "missing":
        await database.litellm_apptable.delete(where={"app_id": app.app_id})
    else:
        await database.litellm_apptable.update(
            where={"app_id": app.app_id}, data={"resource_type": AppResourceType.PROJECT.value}
        )
    results: Final = (
        await store.patch(app.app_id, AppPatch(budget=Budget(max_budget=7)), "admin"),
        await store.put_organization(
            app.app_id, "org", AppOrganizationPut(budget_override=Budget(max_budget=3)), "admin"
        ),
        await store.put_member(app.app_id, "member", AppMemberPut(budget_override=Budget(max_budget=1)), "admin"),
    )
    assert all(isinstance(result, AppFailure) and result.status == status for result in results)
    assert not database.litellm_budgettable.rows
    assert not database.litellm_appcustomerorgtable.rows
    assert not database.litellm_appcustomermembertable.rows
    if state == "missing":
        assert await store.get_member(app.app_id, "member") == AppFailure(404, "App not found.")


@pytest.mark.asyncio
async def test_should_reject_budget_assignments_to_missing_customers(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    database, store, app = app_store
    organization: Final = await store.put_organization(
        app.app_id, "missing-org", AppOrganizationPut(budget_override=Budget(max_budget=3)), "admin"
    )
    member: Final = await store.put_member(
        app.app_id, "missing-member", AppMemberPut(budget_override=Budget(max_budget=1)), "admin"
    )
    assert organization == AppFailure(404, "Customer organization not found.")
    assert member == AppFailure(404, "Customer member not found.")
    assert await store.get_member(app.app_id, "missing-member") == member
    assert not database.litellm_budgettable.rows
    assert not database.litellm_appcustomerorgtable.rows
    assert not database.litellm_appcustomermembertable.rows


@pytest.mark.asyncio
async def test_should_report_a_binding_removed_before_its_budget_update(
    app_store: tuple[AppDatabase, AppStore, App], monkeypatch: pytest.MonkeyPatch
) -> None:
    database, store, app = app_store
    update: Final = database.litellm_apptable.update

    async def remove_before_update(where: Mapping[str, object], data: Mapping[str, object]) -> Record | None:
        await database.litellm_apptable.delete(where=where)
        return await update(where=where, data=data)

    monkeypatch.setattr(database.litellm_apptable, "update", remove_before_update)
    assert await store.patch(app.app_id, AppPatch(), "admin") == AppFailure(404, "App not found.")
    assert await store.get(app.app_id) == AppFailure(404, "App not found.")


@pytest.mark.asyncio
async def test_should_skip_a_key_rotated_during_identity_backfill(monkeypatch: pytest.MonkeyPatch) -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    await seed_resources(database)
    update_many: Final = database.litellm_verificationtoken.update_many

    async def rotate_before_backfill(where: Mapping[str, object], data: Mapping[str, object]) -> int:
        await database.litellm_verificationtoken.update(where={"token": "token-a"}, data={"token": "rotated-token"})
        return await update_many(where=where, data=data)

    with monkeypatch.context() as backfill:
        backfill.setattr(database.litellm_verificationtoken, "update_many", rotate_before_backfill)
        assert await store.resources(0, 100, AppResourceType.VIRTUAL_KEY) == ()
    discovered: Final = await store.resources(0, 100, AppResourceType.VIRTUAL_KEY)
    assert len(discovered) == 1 and discovered[0].alias == "Service key"
    assert "rotated-token" not in discovered[0].model_dump_json()
    assert await store.key_resource_id("rotated-token") == discovered[0].resource_id


async def put_budget_association(
    store: AppStore, app: App, scope: str, data: MemberPatch
) -> AppMember | AppOrganization:
    values: Final = data.model_dump(exclude_unset=True)
    result: Final = (
        await store.put_member(app.app_id, "member", AppMemberPut.model_validate(values), "admin")
        if scope == "member"
        else await store.put_organization(app.app_id, "org", AppOrganizationPut.model_validate(values), "admin")
    )
    assert isinstance(result, (AppMember, AppOrganization))
    return result


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ("member", "organization"))
@pytest.mark.parametrize(
    "legacy",
    (None, Budget(max_budget=15, model_rpm_limit={"model": 5}, cap_policy="pause_allow_increase", allowed_models=())),
)
async def test_app_custom_policy_survives_edits_grants_and_expiry(
    app_store: tuple[AppDatabase, AppStore, App], scope: str, legacy: Budget | None
) -> None:
    database, store, app = app_store
    preset: Final = await profile(store, "Standard", 10)
    policy: Final = Budget(max_budget=10, rpm_limit=4, budget_duration="1mo", allowed_models=("base",))
    await store.customers.update_budget(preset, policy, "admin")
    await store.patch(
        app.app_id, AppPatch(default_tier_budget_id=preset, default_customer_member_budget=policy), "admin"
    )
    global_member: Final = database.litellm_customermembertable.rows["member"]
    put: Final = partial(put_budget_association, store, app, scope)
    initial: Final = await put(MemberPatch(budget_override=legacy) if legacy else MemberPatch())
    expected: Final = legacy or policy
    assert initial.effective_budget == expected
    created: Final = await put(MemberPatch(budget=Budget(max_budget=20), budget_override=None))
    assert created.budget == expected.model_copy(update={"max_budget": 20})
    assert created.budget_id != preset
    await database.litellm_budgettable.update(
        where={"budget_id": created.budget_id},
        data={"temp_budget_increase": 5, "temp_budget_expiry": NOW.replace(year=2027)},
    )
    loaded: Final = await put(MemberPatch())
    assert loaded.budget == created.budget
    assert loaded.effective_budget == created.budget.model_copy(update={"max_budget": 25})
    saved: Final = await put(MemberPatch(budget=loaded.budget))
    assert saved.effective_budget == loaded.effective_budget
    grant: Final = await put(
        MemberPatch(budget_override=Budget(max_budget=30), budget_override_expires_at=NOW.replace(year=2027)),
    )
    edited: Final = await put(MemberPatch(budget=Budget(rpm_limit=None, model_rpm_limit=None, tpm_limit=0)))
    assert edited.budget_id == created.budget_id
    assert edited.budget == created.budget.model_copy(
        update={"rpm_limit": None, "model_rpm_limit": None, "tpm_limit": 0}
    )
    assert edited.budget_override_id == grant.budget_override_id
    assert edited.effective_budget == edited.budget.model_copy(update={"max_budget": 30})
    expired: Final = await put_budget_association(AppStore(database, NOW.replace(year=2028)), app, scope, MemberPatch())
    assert expired.effective_budget == edited.budget
    assert database.litellm_customermembertable.rows["member"] == global_member
    assert await store.budget(preset) == policy
    cleared: Final = await put(MemberPatch(budget=None, budget_override=None))
    assert cleared.effective_budget == policy
    assert await store.budget(created.budget_id) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", (BudgetGroupScope.ORGANIZATION, BudgetGroupScope.APP_ORGANIZATION))
async def test_app_org_custom_forks_shared_profile_and_expiry_restores_private_budget(
    app_store: tuple[AppDatabase, AppStore, App], scope: BudgetGroupScope
) -> None:
    database, store, app = app_store
    tier: Final = await profile(store, "Standard", 10, scope)
    await store.customers.update_budget(tier, Budget(rpm_limit=4, budget_duration="1mo"), "admin")
    selected: Final = await store.put_organization(app.app_id, "org", AppOrganizationPut(budget_id=tier), "admin")
    assert isinstance(selected, AppOrganization) and selected.budget_id == tier and selected.budget is None
    forked: Final = await store.put_organization(
        app.app_id, "org", AppOrganizationPut(budget=Budget(max_budget=12)), "admin"
    )
    assert isinstance(forked, AppOrganization) and forked.budget is not None
    assert forked.budget_id != tier and forked.budget.rpm_limit == 4 and forked.budget.max_budget == 12
    assert await store.budget(tier) == selected.effective_budget
    original_row: Final = await store.customers.budgets.find_unique(where={"budget_id": tier})
    private_row: Final = await store.customers.budgets.find_unique(where={"budget_id": forked.budget_id})
    assert original_row is not None and private_row is not None
    assert private_row.model_dump()["budget_reset_at"] == original_row.model_dump()["budget_reset_at"]
    patched: Final = await store.put_organization(
        app.app_id, "org", AppOrganizationPut(budget=Budget(rpm_limit=None)), "admin"
    )
    assert isinstance(patched, AppOrganization) and patched.budget is not None
    assert (
        patched.budget_id == forked.budget_id and patched.budget.rpm_limit is None and patched.budget.max_budget == 12
    )
    increased: Final = await store.put_organization(
        app.app_id,
        "org",
        AppOrganizationPut(budget_override=Budget(max_budget=20), budget_override_expires_at=NOW.replace(year=2027)),
        "admin",
    )
    assert isinstance(increased, AppOrganization) and increased.effective_budget == Budget(
        max_budget=20, budget_duration="1mo", allowed_models=()
    )
    expired: Final = await AppStore(database, NOW.replace(year=2028)).organization(app, "org")
    assert expired.effective_budget == await store.budget(forked.budget_id)
    restored: Final = await store.put_organization(
        app.app_id, "org", AppOrganizationPut(budget_id=tier, budget_override=None), "admin"
    )
    assert isinstance(restored, AppOrganization) and restored.budget_id == tier
    assert await store.budget(forked.budget_id) is None
    assert await store.budget(tier) is not None
