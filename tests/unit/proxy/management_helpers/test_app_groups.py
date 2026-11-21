from typing import Final

import pytest

from litellm.proxy.management_helpers.app_groups import AppGroupStore
from litellm.proxy.management_helpers.apps import AppStore
from litellm.proxy.management_helpers.customer_hierarchy import BudgetGroupScope
from litellm.types.proxy.management_endpoints.app_groups import (
    AppGroup,
    AppGroupApp,
    AppGroupBudgetPut,
    AppGroupCreate,
    AppGroupOrganization,
    AppGroupPatch,
)
from litellm.types.proxy.management_endpoints.apps import (
    App,
    AppCreate,
    AppDefinition,
    AppFailure,
    AppPatch,
    AppResourceType,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget
from tests.unit.proxy.management_helpers.test_apps import (
    AppDatabase,
    configure_default_profile,
    profile,
    seed_resources,
)
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ("default", "none", "custom", "preset"))
async def test_new_groups_and_selected_apps_use_independent_defaults_without_changing_existing_caps(
    choice: str,
) -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    groups: Final = AppGroupStore(store)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    existing: Final = await store.create(AppCreate(resource_id="chat", budget=Budget(max_budget=50)), "admin")
    assert isinstance(existing, App)
    group_catalog: Final = await configure_default_profile(store, BudgetGroupScope.APP_GROUP)
    app_catalog: Final = await configure_default_profile(store, BudgetGroupScope.APP)
    other: Final = await profile(store, "Other", 200, BudgetGroupScope.APP_GROUP)
    selection: Final = {
        "default": AppGroupCreate(name="Default"),
        "none": AppGroupCreate(name="None", budget_id=None),
        "custom": AppGroupCreate(name="Custom", budget=Budget(max_budget=300)),
        "preset": AppGroupCreate(name="Preset", budget_id=other),
    }[choice]
    created: Final = await groups.create(
        selection.model_copy(
            update={
                "apps": tuple(
                    AppGroupApp(resource_type=AppResourceType.TEAM, resource_id=id) for id in ("chat", "search")
                )
            }
        ),
        "admin",
    )
    assert isinstance(created, AppGroup)
    assert (created.budget.max_budget if created.budget else None) == {
        "default": 100,
        "none": None,
        "custom": 300,
        "preset": 200,
    }[choice]
    if choice == "default":
        assert created.budget_id == group_catalog.default_budget_template_id
    search: Final = await store.by_resource(AppResourceType.TEAM, "search")
    assert isinstance(search, App) and search.budget_id == app_catalog.default_budget_template_id
    chat: Final = await store.get(existing.app_id)
    assert isinstance(chat, App) and chat.budget_id == existing.budget_id and chat.budget == existing.budget
    renamed: Final = await groups.patch(created.app_group_id, AppGroupPatch(name="Renamed"), "admin")
    assert isinstance(renamed, AppGroup) and renamed.budget_id == created.budget_id


@pytest.mark.asyncio
async def test_group_and_app_caps_are_independent_and_membership_can_be_cleared() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    groups: Final = AppGroupStore(store)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    group: Final = await groups.create(AppGroupCreate(name="Product", budget=Budget(max_budget=10000)), "admin")
    assert isinstance(group, AppGroup)
    first: Final = await store.create(
        AppCreate(resource_id="chat", app_group_id=group.app_group_id, budget=Budget(max_budget=2000)), "admin"
    )
    second: Final = await store.create(
        AppCreate(resource_id="search", app_group_id=group.app_group_id, budget=Budget(max_budget=9000)), "admin"
    )
    assert isinstance(first, App) and isinstance(second, App)
    assert first.budget is not None and first.budget.max_budget == 2000
    assert second.budget is not None and second.budget.max_budget == 9000
    assert group.budget is not None and group.budget.max_budget == 10000
    rejected: Final = await store.patch(first.app_id, AppPatch(app_group_id="missing"), "admin")
    assert rejected == AppFailure(404, "App group not found.")
    cleared: Final = await store.patch(first.app_id, AppPatch(app_group_id=None), "admin")
    assert isinstance(cleared, App) and cleared.app_group_id is None
    assert cleared.budget == first.budget
    unchanged: Final = await store.get(second.app_id)
    assert isinstance(unchanged, App) and unchanged.app_group_id == group.app_group_id


@pytest.mark.asyncio
async def test_groups_reuse_existing_presets_and_customizing_does_not_change_other_consumers() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    groups: Final = AppGroupStore(store)
    preset: Final = await profile(store, "Shared policy", 10000, BudgetGroupScope.APP_GROUP)
    app_preset: Final = await profile(store, "App policy", 2000, BudgetGroupScope.APP)
    rejected: Final = await groups.create(AppGroupCreate(name="Wrong catalog", budget_id=app_preset), "admin")
    assert isinstance(rejected, AppFailure) and rejected.status == 422
    first: Final = await groups.create(AppGroupCreate(name="First", budget_id=preset), "admin")
    second: Final = await groups.create(AppGroupCreate(name="Second", budget_id=preset), "admin")
    assert isinstance(first, AppGroup) and isinstance(second, AppGroup)
    assert first.budget_id == second.budget_id == preset
    customized: Final = await groups.patch(first.app_group_id, AppGroupPatch(budget=Budget(max_budget=2000)), "admin")
    assert isinstance(customized, AppGroup) and customized.budget_id != preset
    assert customized.budget is not None and customized.budget.max_budget == 2000
    await database.litellm_budgettable.update({"budget_id": preset}, {"max_budget": 8000})
    other: Final = await groups.get(second.app_group_id)
    assert isinstance(other, AppGroup) and other.budget is not None and other.budget.max_budget == 8000
    cleared: Final = await groups.patch(first.app_group_id, AppGroupPatch(budget_id=None), "admin")
    assert isinstance(cleared, AppGroup) and cleared.budget is None
    assert await database.litellm_budgettable.find_unique({"budget_id": customized.budget_id}) is None
    assert await database.litellm_budgettable.find_unique({"budget_id": preset}) is not None


@pytest.mark.asyncio
async def test_customer_budget_is_scoped_to_one_group_and_uses_the_existing_customer_app_catalog() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    groups: Final = AppGroupStore(store)
    await database.litellm_customerorganizationtable.create({"customer_org_id": "customer"})
    group: Final = await groups.create(AppGroupCreate(name="Product"), "admin")
    assert isinstance(group, AppGroup)
    preset: Final = await profile(store, "Customer policy", 3000, BudgetGroupScope.APP_ORGANIZATION)
    wrong: Final = await groups.patch(group.app_group_id, AppGroupPatch(budget_id=preset), "admin")
    assert isinstance(wrong, AppFailure) and wrong.status == 422
    attached: Final = await groups.put_organization(
        group.app_group_id, "customer", AppGroupBudgetPut(budget_id=preset), "admin"
    )
    assert isinstance(attached, AppGroupOrganization) and attached.budget_id == preset
    assert attached.budget is not None and attached.budget.max_budget == 3000
    assert await groups.list_organizations(group.app_group_id, 0, 100) == (attached,)
    missing: Final = await groups.put_organization(group.app_group_id, "missing", AppGroupBudgetPut(), "admin")
    assert missing == AppFailure(404, "Customer organization not found.")
    duplicate: Final = await groups.create(AppGroupCreate(name="Product"), "admin")
    assert isinstance(duplicate, AppFailure) and duplicate.status == 409


@pytest.mark.asyncio
async def test_group_membership_selects_existing_and_unconfigured_apps_and_preserves_budgets() -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    groups: Final = AppGroupStore(store)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    existing: Final = await store.create(AppCreate(resource_id="chat", budget=Budget(max_budget=2000)), "admin")
    assert isinstance(existing, App)
    selection: Final = tuple(
        AppGroupApp(resource_type=AppResourceType.TEAM, resource_id=id) for id in ("chat", "search")
    )
    created: Final = await groups.create(
        AppGroupCreate(name="Selected apps", apps=selection, budget=Budget(max_budget=10000)), "admin"
    )
    assert isinstance(created, AppGroup)
    assigned: Final = await store.list(0, 100)
    assert {app.resource_id for app in assigned if app.app_group_id == created.app_group_id} == {"chat", "search"}
    chat: Final = await store.get(existing.app_id)
    assert isinstance(chat, App) and chat.budget == existing.budget
    other: Final = await groups.create(AppGroupCreate(name="Other", apps=selection[:1]), "admin")
    assert isinstance(other, AppGroup)
    moved: Final = await store.get(existing.app_id)
    assert isinstance(moved, App) and moved.app_group_id == other.app_group_id and moved.budget == existing.budget
    unchanged: Final = await groups.patch(created.app_group_id, AppGroupPatch(name="Renamed"), "admin")
    assert isinstance(unchanged, AppGroup)
    search: Final = await store.by_resource(AppResourceType.TEAM, "search")
    assert isinstance(search, App) and search.app_group_id == created.app_group_id
    cleared: Final = await groups.patch(created.app_group_id, AppGroupPatch(apps=()), "admin")
    assert isinstance(cleared, AppGroup)
    ungrouped: Final = await store.get(search.app_id)
    assert isinstance(ungrouped, App) and ungrouped.app_group_id is None
    assert await store.get(existing.app_id) == moved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resource_type,resource_id", ((AppResourceType.TEAM, "missing"), (AppResourceType.PROJECT, "project"))
)
async def test_invalid_group_selection_leaves_group_and_existing_app_unchanged(
    resource_type: AppResourceType, resource_id: str
) -> None:
    database: Final = AppDatabase()
    store: Final = AppStore(database, NOW)
    groups: Final = AppGroupStore(store)
    await seed_resources(database)
    await store.put_definition(AppDefinition(resource_type=AppResourceType.TEAM))
    existing: Final = await store.create(AppCreate(resource_id="chat"), "admin")
    assert isinstance(existing, App)
    selection: Final = (
        AppGroupApp(resource_type=AppResourceType.TEAM, resource_id="chat"),
        AppGroupApp(resource_type=resource_type, resource_id=resource_id),
    )
    rejected: Final = await groups.create(AppGroupCreate(name="Invalid", apps=selection), "admin")
    assert isinstance(rejected, AppFailure) and rejected.status in (404, 422)
    assert await groups.list(0, 100) == ()
    assert await store.get(existing.app_id) == existing
