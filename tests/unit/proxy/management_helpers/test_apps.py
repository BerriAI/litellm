from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, cast
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from litellm.proxy.management_helpers.apps import AppStore
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
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import Budget, BudgetTemplate
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW, Database, Record, Table

if TYPE_CHECKING:
    from prisma import Prisma


class AppTable(Table):
    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> Record:
        return await super().create({"app_id": str(uuid4()), **data}, include)

    async def find_unique(
        self,
        where: Mapping[str, object],
        include: Mapping[str, object] | None = None,
    ) -> Record | None:
        return next((row for row in self.rows.values() if self.matches(row, where)), None)

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        order: Mapping[str, object] | Sequence[Mapping[str, object]] | None = None,
        include: Mapping[str, object] | None = None,
        skip: int = 0,
        take: int | None = None,
    ) -> tuple[Record, ...]:
        return await super().find_many(where, (order,) if isinstance(order, Mapping) else order, include, skip, take)


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


class AppDatabase(Database):
    def __init__(self) -> None:
        super().__init__()
        self.litellm_apptable: Final = AppTable("app_id", self)
        self.litellm_appcustomerorgtable: Final = CompositeTable("app_id_customer_org_id", self)
        self.litellm_appcustomermembertable: Final = CompositeTable("app_id_customer_member_id", self)


@pytest.fixture
async def app_store() -> tuple[AppDatabase, AppStore, App]:
    database: Final = AppDatabase()
    store: Final = AppStore(cast("Prisma", database), NOW)  # cast-ok: in-memory Prisma edge
    await database.litellm_customerorganizationtable.create({"customer_org_id": "org"})
    await database.litellm_customermembertable.create({"customer_member_id": "member", "parent_org_id": "org"})
    app: Final = await store.create(AppCreate(alias="assistant"), "admin")
    assert isinstance(app, App)
    return database, store, app


@pytest.mark.asyncio
async def test_org_override_wins_then_tier_then_default(app_store: tuple[AppDatabase, AppStore, App]) -> None:
    _, store, app = app_store
    default: Final = await store.create_tier(app.app_id, BudgetTemplate(name="default", max_budget=10), "admin")
    tier: Final = await store.create_tier(app.app_id, BudgetTemplate(name="large", max_budget=20), "admin")
    assert isinstance(default, AppTier) and isinstance(tier, AppTier)
    configured: Final = await store.patch(app.app_id, AppPatch(default_tier_budget_id=default.budget_id), "admin")
    assert isinstance(configured, App)
    initial: Final = await store.organization(configured, "org")
    assert initial.effective_budget == Budget(max_budget=10, allowed_models=())
    selected: Final = await store.put_organization(
        app.app_id,
        "org",
        AppOrganizationPut(tier_budget_id=tier.budget_id),
        "admin",
    )
    assert isinstance(selected, AppOrganization) and selected.effective_budget == Budget(
        max_budget=20, allowed_models=()
    )
    overridden: Final = await store.put_organization(
        app.app_id,
        "org",
        AppOrganizationPut(tier_budget_id=tier.budget_id, budget_override=Budget(max_budget=3)),
        "admin",
    )
    assert isinstance(overridden, AppOrganization) and overridden.effective_budget == Budget(
        max_budget=3, allowed_models=()
    )


@pytest.mark.asyncio
async def test_member_override_wins_over_org_and_app_defaults(app_store: tuple[AppDatabase, AppStore, App]) -> None:
    _, store, app = app_store
    configured: Final = await store.patch(
        app.app_id,
        AppPatch(default_customer_member_budget=Budget(max_budget=10)),
        "admin",
    )
    assert isinstance(configured, App)
    initial: Final = await store.member(configured, "member", await store.organization(configured, "org"))
    assert initial.effective_budget == Budget(max_budget=10, allowed_models=())
    org: Final = await store.put_organization(
        app.app_id,
        "org",
        AppOrganizationPut(customer_member_budget=Budget(max_budget=5)),
        "admin",
    )
    assert isinstance(org, AppOrganization)
    inherited: Final = await store.member(configured, "member", org)
    assert inherited.effective_budget == Budget(max_budget=5, allowed_models=())
    override: Final = await store.put_member(
        app.app_id,
        "member",
        AppMemberPut(budget_override=Budget(max_budget=1)),
        "admin",
    )
    assert isinstance(override, AppMember) and override.effective_budget == Budget(max_budget=1, allowed_models=())
    restored: Final = await store.put_member(app.app_id, "member", AppMemberPut(), "admin")
    assert isinstance(restored, AppMember) and restored.effective_budget == Budget(max_budget=5, allowed_models=())


@pytest.mark.asyncio
async def test_default_and_referenced_tiers_cannot_be_deleted(app_store: tuple[AppDatabase, AppStore, App]) -> None:
    _, store, app = app_store
    tier: Final = await store.create_tier(app.app_id, BudgetTemplate(name="tier", max_budget=10), "admin")
    assert isinstance(tier, AppTier)
    await store.patch(app.app_id, AppPatch(default_tier_budget_id=tier.budget_id), "admin")
    assert await store.delete_tier(app.app_id, tier.budget_id, "admin") == AppFailure(
        409,
        "A default or referenced tier cannot be deleted.",
    )
    await store.patch(app.app_id, AppPatch(default_tier_budget_id=None), "admin")
    await store.put_organization(app.app_id, "org", AppOrganizationPut(tier_budget_id=tier.budget_id), "admin")
    assert await store.delete_tier(app.app_id, tier.budget_id, "admin") == AppFailure(
        409,
        "A default or referenced tier cannot be deleted.",
    )
    await store.put_organization(app.app_id, "org", AppOrganizationPut(), "admin")
    assert await store.delete_tier(app.app_id, tier.budget_id, "admin") is None
    remaining: Final = await store.get(app.app_id)
    assert isinstance(remaining, App) and remaining.tier_budget_ids == ()


@pytest.mark.asyncio
async def test_unknown_and_foreign_tiers_are_rejected_without_assigning_rows(
    app_store: tuple[AppDatabase, AppStore, App],
) -> None:
    database, store, app = app_store
    other: Final = await store.create(AppCreate(alias="other"), "admin")
    assert isinstance(other, App)
    tier: Final = await store.create_tier(other.app_id, BudgetTemplate(name="other"), "admin")
    assert isinstance(tier, AppTier)
    rejected: Final = await store.put_organization(
        app.app_id,
        "org",
        AppOrganizationPut(tier_budget_id=tier.budget_id),
        "admin",
    )
    assert rejected == AppFailure(422, "Tier does not belong to this app.")
    assert database.litellm_appcustomerorgtable.rows == {}
    missing: Final = await store.by_alias("missing")
    assert missing == AppFailure(400, "Unknown app alias.")
    assert len(await store.list(0, 100)) == 2
