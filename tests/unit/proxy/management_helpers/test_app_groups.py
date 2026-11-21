from typing import Final

import pytest

from litellm.proxy.management_helpers.app_groups import AppGroupStore
from litellm.proxy.management_helpers.apps import AppStore
from litellm.proxy.management_helpers.customer_hierarchy import BudgetGroupScope
from litellm.types.proxy.management_endpoints.app_groups import (
    AppGroup,
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
from tests.unit.proxy.management_helpers.test_apps import AppDatabase, profile, seed_resources
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW


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
    preset: Final = await profile(store, "Shared policy", 10000, BudgetGroupScope.APP)
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
