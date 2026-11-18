from collections.abc import Mapping, Sequence
from typing import Final
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from litellm.proxy.management_helpers.apps import AppStore
from litellm.proxy.management_helpers.customer_hierarchy import ErrorCode, Failure
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


class AppDatabase(Database):
    def __init__(self) -> None:
        super().__init__()
        self.litellm_apptable: Final = AppTable("app_id", self)
        self.litellm_appcustomerorgtable: Final = CompositeTable("app_id_customer_org_id", self)
        self.litellm_appcustomermembertable: Final = CompositeTable("app_id_customer_member_id", self)
        self.litellm_teamtable: Final = AppTable("team_id", self)
        self.litellm_projecttable = AppTable("project_id", self)
        self.litellm_verificationtoken = AppTable("token", self)

    def has_budget_reference(self, budget_id: str, relation: str) -> bool:
        references: Final = {
            "apps": (self.litellm_apptable, "budget_id"),
            "app_default_tiers": (self.litellm_apptable, "default_tier_budget_id"),
            "app_default_members": (
                self.litellm_apptable,
                "default_customer_member_budget_id",
            ),
            "app_org_tiers": (self.litellm_appcustomerorgtable, "tier_budget_id"),
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


async def seed_resources(database: AppDatabase) -> None:
    await database.litellm_teamtable.create({"team_id": "chat", "team_alias": "Chat"})
    await database.litellm_teamtable.create({"team_id": "search", "team_alias": "Search"})
    await database.litellm_projecttable.create({"project_id": "project", "project_alias": "Project"})
    await database.litellm_verificationtoken.create({"token": "token-a", "key_alias": "Service key", "key_id": None})


async def profile(store: AppStore, name: str, maximum: float) -> str:
    from prisma import Json

    budget_id: Final = await store.customers.create_budget(BudgetTemplate(name=name, max_budget=maximum), "admin", name)
    current: Final = await store.customers.config.find_unique(
        where={"param_name": "customer_organization_budget_group_ids"}
    )
    ids: Final = TypeAdapter(tuple[str, ...]).validate_python(current.model_dump()["param_value"]) if current else ()
    await store.customers.config.upsert(
        where={"param_name": "customer_organization_budget_group_ids"},
        data={
            "create": {
                "param_name": "customer_organization_budget_group_ids",
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
    preview: Final = await store.resources(0, 100, AppResourceType.TEAM)
    assert tuple(resource.resource_id for resource in preview) == ("chat", "search")
    assert all(resource.app_id is None for resource in preview)
    rejected: Final = await store.create(AppCreate(resource_id="chat"), "admin")
    assert isinstance(rejected, AppFailure) and rejected.status == 409
    assert not database.litellm_apptable.rows
    assert await store.definition() == AppDefinition()


@pytest.mark.asyncio
async def test_binding_is_idempotent_and_locks_the_gateway_definition(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    _, store, app = app_store
    again: Final = await store.create(AppCreate(resource_id="chat"), "admin")
    assert isinstance(again, App) and again.app_id == app.app_id
    assert app.resource_type == AppResourceType.TEAM and app.resource_id == "chat" and app.alias == "Chat"
    rejected: Final = await store.put_definition(AppDefinition(resource_type=AppResourceType.PROJECT))
    assert isinstance(rejected, AppFailure) and rejected.status == 409
    same: Final = await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    assert same == AppDefinition(resource_type=AppResourceType.TEAM)
    assert await store.delete(app.app_id) is None
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
async def test_org_override_wins_then_shared_profile_then_default(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    _, store, app = app_store
    default: Final = await profile(store, "Standard", 10)
    selected: Final = await profile(store, "Pro", 20)
    configured: Final = await store.patch(app.app_id, AppPatch(default_tier_budget_id=default), "admin")
    assert isinstance(configured, App)
    inherited: Final = await store.organization(configured, "org")
    assert inherited.effective_budget == Budget(max_budget=10, allowed_models=())
    assigned: Final = await store.put_organization(
        app.app_id, "org", AppOrganizationPut(tier_budget_id=selected), "admin"
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
@pytest.mark.parametrize("assignment", ("default", "customer"))
async def test_profiles_in_use_by_apps_cannot_be_removed(
    app_store: tuple[AppDatabase, AppStore, App], assignment: str
) -> None:
    database, store, app = app_store
    group: Final = await profile(store, "Standard", 10)
    if assignment == "default":
        await store.patch(app.app_id, AppPatch(default_tier_budget_id=group), "admin")
    else:
        await store.put_organization(app.app_id, "org", AppOrganizationPut(tier_budget_id=group), "admin")
    groups: Final = await store.customers.get_budget_groups(None)
    assert not isinstance(groups, Failure)
    rejected: Final = await store.customers.configure_templates(
        None,
        BudgetGroupsUpdate(
            expected_revision=groups.budget_groups_revision,
            default_budgets=(),
            default_budget_template=None,
        ),
        "admin",
    )
    assert rejected == Failure(ErrorCode.TEMPLATE_IN_USE)
    assert group in database.litellm_budgettable.rows


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
    restored: Final = await store.put_member(app.app_id, "member", AppMemberPut(), "admin")
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
    assert isinstance(assigned, AppFailure) and assigned.status == 422
    assert isinstance(default, AppFailure) and default.status == 422


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
