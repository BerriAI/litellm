from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import reduce
from typing import TYPE_CHECKING, Final, cast

import pytest
from pydantic import BaseModel, ConfigDict

from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy, ErrorCode, Failure, Page
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    BudgetTemplate,
    CustomerMember,
    MemberCreate,
    MemberPatch,
    Organization,
    OrganizationCreate,
    OrganizationPatch,
    Team,
    TeamCreate,
    TeamPatch,
)

if TYPE_CHECKING:
    from prisma import Prisma

NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


class Record(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")


class Table:
    def __init__(self, id_field: str, database: "Database") -> None:
        self.id_field = id_field
        self.database = database
        self.rows: Mapping[str, Record] = {}

    async def include(self, row: Record, include: Mapping[str, object] | None) -> Record:
        if not include:
            return row
        if "default_budgets" in include:
            templates: Final = await self.database.litellm_customerorganizationbudgettable.find_many(
                where={"customer_org_id": row.model_dump()["customer_org_id"]},
                include={"budget": True},
                order=[{"created_at": "asc"}, {"budget_id": "asc"}],
            )
            return row.model_copy(update={"default_budgets": tuple(template.model_dump() for template in templates)})
        if "budget" in include:
            budget_id: Final = row.model_dump().get("budget_id")
            budget: Final = self.database.litellm_budgettable.rows.get(str(budget_id))
            return row.model_copy(update={"budget": budget.model_dump() if budget else None})
        return row

    async def find_unique(
        self, where: Mapping[str, object], include: Mapping[str, object] | None = None
    ) -> Record | None:
        row: Final = self.rows.get(str(where[self.id_field]))
        return await self.include(row, include) if row else None

    async def create(self, data: Mapping[str, object], include: Mapping[str, object] | None = None) -> Record:
        from prisma.errors import UniqueViolationError

        if str(data[self.id_field]) in self.rows:
            raise UniqueViolationError({})
        row: Final = Record.model_validate({"created_at": NOW, "updated_at": NOW, **data})
        self.rows = {**self.rows, str(data[self.id_field]): row}
        return await self.include(row, include)

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> Record | None:
        row_id: Final = str(where[self.id_field])
        if row_id not in self.rows:
            return None
        row: Final = self.rows[row_id].model_copy(update=dict(data))
        self.rows = {**self.rows, row_id: row}
        return row

    async def find_many(
        self,
        where: Mapping[str, object] | None = None,
        order: Sequence[Mapping[str, object]] | None = None,
        include: Mapping[str, object] | None = None,
        skip: int = 0,
        take: int | None = None,
    ) -> tuple[Record, ...]:
        matches: Final = tuple(
            row
            for row in self.rows.values()
            if all(row.model_dump().get(key) == value for key, value in (where or {}).items())
        )
        ordered: Final = reduce(
            lambda rows, key: tuple(sorted(rows, key=lambda row: str(row.model_dump().get(next(iter(key)))))),
            reversed(order or ()),
            matches,
        )
        selected: Final = ordered[skip : skip + take] if take is not None else ordered[skip:]
        return tuple([await self.include(row, include) for row in selected])

    async def count(self, where: Mapping[str, object] | None = None) -> int:
        return len(await self.find_many(where=where))

    async def delete(self, where: Mapping[str, object]) -> Record | None:
        row_id: Final = str(where[self.id_field])
        row: Final = self.rows.get(row_id)
        self.rows = {key: value for key, value in self.rows.items() if key != row_id}
        return row

    async def delete_many(self, where: Mapping[str, object]) -> int:
        matched: Final = await self.find_many(where=where)
        self.rows = {key: row for key, row in self.rows.items() if row not in matched}
        return len(matched)


class Database:
    def __init__(self) -> None:
        self.litellm_customerorganizationtable = Table("customer_org_id", self)
        self.litellm_customerteamtable = Table("customer_team_id", self)
        self.litellm_customermembertable = Table("customer_member_id", self)
        self.litellm_budgettable = Table("budget_id", self)
        self.litellm_customerorganizationbudgettable = Table("budget_id", self)

    @asynccontextmanager
    async def tx(self) -> AsyncIterator["Database"]:
        tables: Final = (
            self.litellm_customerorganizationtable,
            self.litellm_customerteamtable,
            self.litellm_customermembertable,
            self.litellm_budgettable,
            self.litellm_customerorganizationbudgettable,
        )
        snapshots: Final = tuple(table.rows for table in tables)
        try:
            yield self
        except Exception:
            for table, rows in zip(tables, snapshots):
                table.rows = rows
            raise


@pytest.fixture
def database() -> Database:
    return Database()


@pytest.fixture
def hierarchy(database: Database) -> CustomerHierarchy:
    return CustomerHierarchy(cast("Prisma", database))  # cast-ok: in-memory implementation of the Prisma edge


async def family(hierarchy: CustomerHierarchy, budget: Budget | None = None) -> tuple[Organization, Team]:
    organization: Final = await hierarchy.create_organization(
        OrganizationCreate(
            customer_org_id="org",
            default_budgets=(BudgetTemplate(**budget.model_dump()),) if budget is not None else (),
        ),
        "admin",
    )
    team: Final = await hierarchy.create_team(
        TeamCreate(customer_team_id="team", parent_org_id="org", budget=Budget(max_budget=999)), "admin"
    )
    assert isinstance(organization, Organization)
    assert isinstance(team, Team)
    return organization, team


@pytest.mark.asyncio
async def test_member_copies_default_not_team_budget_and_overrides_only_supplied_fields(
    hierarchy: CustomerHierarchy,
) -> None:
    organization, team = await family(
        hierarchy, Budget(max_budget=100, tpm_limit=1000, rpm_limit=10, allowed_models=("model",), budget_duration="1d")
    )
    copied: Final = await hierarchy.create_member(MemberCreate(parent_team_id=team.customer_team_id), "admin")
    overridden: Final = await hierarchy.create_member(
        MemberCreate(parent_team_id=team.customer_team_id, budget=Budget(rpm_limit=None, max_budget=5)), "admin"
    )
    assert isinstance(copied, CustomerMember) and isinstance(overridden, CustomerMember)
    assert len({copied.budget_id, overridden.budget_id, organization.default_budget_id}) == 3
    assert copied.budget == Budget(
        max_budget=100, tpm_limit=1000, rpm_limit=10, allowed_models=("model",), budget_duration="1d"
    )
    assert overridden.budget == Budget(
        max_budget=5, tpm_limit=1000, rpm_limit=None, allowed_models=("model",), budget_duration="1d"
    )
    assert await hierarchy.get_organization("org") == organization


@pytest.mark.asyncio
async def test_member_patch_changes_only_its_budget(hierarchy: CustomerHierarchy) -> None:
    organization, team = await family(hierarchy, Budget(max_budget=100, allowed_models=("model",)))
    member: Final = await hierarchy.create_member(MemberCreate(parent_team_id=team.customer_team_id), "admin")
    assert isinstance(member, CustomerMember)
    updated: Final = await hierarchy.update_member(
        member.customer_member_id, MemberPatch(budget=Budget(max_budget=5)), "editor"
    )
    assert isinstance(updated, CustomerMember)
    assert updated.budget_id == member.budget_id
    assert updated.budget == Budget(max_budget=5, allowed_models=("model",))
    assert updated.updated_by == "editor"
    assert await hierarchy.get_organization("org") == organization


@pytest.mark.asyncio
@pytest.mark.parametrize("templates", ((), (BudgetTemplate(max_budget=50),)))
async def test_org_replacement_deletes_old_budget_rows_but_not_member_copies(
    hierarchy: CustomerHierarchy, database: Database, templates: tuple[BudgetTemplate, ...]
) -> None:
    organization, team = await family(hierarchy, Budget(max_budget=100))
    member: Final = await hierarchy.create_member(MemberCreate(parent_team_id=team.customer_team_id), "admin")
    updated: Final = await hierarchy.update_organization("org", OrganizationPatch(default_budgets=templates), "admin")
    assert isinstance(member, CustomerMember) and isinstance(updated, Organization)
    assert organization.default_budget_id not in database.litellm_budgettable.rows
    assert updated.default_budget_id != organization.default_budget_id
    assert len(updated.default_budgets) == len(templates)
    assert await hierarchy.get_member(member.customer_member_id) == member
    assert member.budget_id in database.litellm_budgettable.rows
    assert len(database.litellm_budgettable.rows) == len(templates) + 2


@pytest.mark.asyncio
async def test_member_without_templates_owns_empty_budget(hierarchy: CustomerHierarchy) -> None:
    _, team = await family(hierarchy)
    member: Final = await hierarchy.create_member(MemberCreate(parent_team_id=team.customer_team_id), "admin")
    assert isinstance(member, CustomerMember)
    assert member.budget == Budget(allowed_models=())
    assert member.budget_id


@pytest.mark.asyncio
async def test_template_selection_override_and_wrong_org_rejection(hierarchy: CustomerHierarchy) -> None:
    organization: Final = await hierarchy.create_organization(
        OrganizationCreate(
            customer_org_id="org",
            default_budgets=(
                BudgetTemplate(max_budget=100, is_default=True),
                BudgetTemplate(max_budget=20, rpm_limit=8),
            ),
        ),
        "admin",
    )
    team: Final = await hierarchy.create_team(TeamCreate(parent_org_id="org"), "admin")
    assert isinstance(organization, Organization) and isinstance(team, Team)
    template: Final = next(budget for budget in organization.default_budgets if not budget.is_default)
    selected: Final = await hierarchy.create_member(
        MemberCreate(
            parent_team_id=team.customer_team_id, budget_template_id=template.budget_id, budget=Budget(rpm_limit=4)
        ),
        "admin",
    )
    assert isinstance(selected, CustomerMember)
    assert selected.budget == Budget(max_budget=20, rpm_limit=4, allowed_models=())
    assert selected.budget_id != template.budget_id
    other: Final = await hierarchy.create_organization(OrganizationCreate(default_budgets=(BudgetTemplate(),)), "admin")
    assert isinstance(other, Organization)
    assert await hierarchy.create_member(
        MemberCreate(parent_team_id=team.customer_team_id, budget_template_id=other.default_budget_id), "admin"
    ) == Failure(ErrorCode.TEMPLATE_NOT_IN_ORG)


@pytest.mark.asyncio
async def test_missing_parents_and_entities_return_typed_failures(hierarchy: CustomerHierarchy) -> None:
    assert await hierarchy.create_team(TeamCreate(parent_org_id="missing"), "admin") == Failure(
        ErrorCode.PARENT_NOT_FOUND
    )
    assert await hierarchy.create_member(MemberCreate(parent_team_id="missing"), "admin") == Failure(
        ErrorCode.PARENT_NOT_FOUND
    )
    assert await hierarchy.get_organization("missing") == Failure(ErrorCode.NOT_FOUND)
    assert await hierarchy.update_member("missing", MemberPatch(budget=Budget()), "admin") == Failure(
        ErrorCode.NOT_FOUND
    )


@pytest.mark.asyncio
async def test_alias_patch_preserves_templates_and_team_budget_patch_preserves_other_fields(
    hierarchy: CustomerHierarchy,
) -> None:
    organization, team = await family(hierarchy, Budget(max_budget=100))
    patched: Final = await hierarchy.update_organization("org", OrganizationPatch(alias="renamed"), "admin")
    assert isinstance(patched, Organization)
    assert patched.default_budgets == organization.default_budgets
    assert patched.alias == "renamed"
    updated: Final = await hierarchy.update_team(team.customer_team_id, TeamPatch(budget=Budget(rpm_limit=7)), "admin")
    assert isinstance(updated, Team)
    assert updated.budget_id == team.budget_id
    assert updated.budget == Budget(max_budget=999, rpm_limit=7, allowed_models=())


@pytest.mark.asyncio
async def test_delete_rejects_parents_with_children_and_cleans_all_owned_budgets(
    hierarchy: CustomerHierarchy, database: Database
) -> None:
    organization, team = await family(hierarchy, Budget(max_budget=100))
    member: Final = await hierarchy.create_member(MemberCreate(parent_team_id=team.customer_team_id), "admin")
    assert isinstance(member, CustomerMember)
    assert await hierarchy.delete_organization("org") == Failure(ErrorCode.HAS_CHILDREN)
    assert await hierarchy.delete_team("team") == Failure(ErrorCode.HAS_CHILDREN)
    assert await hierarchy.get_organization("org") == organization
    assert await hierarchy.delete_member(member.customer_member_id) is None
    assert member.budget_id not in database.litellm_budgettable.rows
    assert await hierarchy.get_member(member.customer_member_id) == Failure(ErrorCode.NOT_FOUND)
    assert await hierarchy.delete_team("team") is None
    assert team.budget_id not in database.litellm_budgettable.rows
    assert await hierarchy.delete_organization("org") is None
    assert database.litellm_budgettable.rows == {}
    assert database.litellm_customerorganizationbudgettable.rows == {}


@pytest.mark.asyncio
async def test_delete_missing_entities_is_not_found(hierarchy: CustomerHierarchy) -> None:
    assert await hierarchy.delete_member("missing") == Failure(ErrorCode.NOT_FOUND)
    assert await hierarchy.delete_team("missing") == Failure(ErrorCode.NOT_FOUND)
    assert await hierarchy.delete_organization("missing") == Failure(ErrorCode.NOT_FOUND)


@pytest.mark.asyncio
async def test_lists_filter_by_parent_paginate_and_resolve_budgets(hierarchy: CustomerHierarchy) -> None:
    organization, team = await family(hierarchy, Budget(max_budget=100))
    other: Final = await hierarchy.create_organization(OrganizationCreate(customer_org_id="other"), "admin")
    other_team: Final = await hierarchy.create_team(TeamCreate(parent_org_id="other"), "admin")
    assert isinstance(other, Organization) and isinstance(other_team, Team)
    member: Final = await hierarchy.create_member(MemberCreate(customer_member_id="a", parent_team_id="team"), "admin")
    second: Final = await hierarchy.create_member(MemberCreate(customer_member_id="b", parent_team_id="team"), "admin")
    assert isinstance(member, CustomerMember) and isinstance(second, CustomerMember)
    assert await hierarchy.list_teams("org") == Page((team,), 1)
    assert await hierarchy.list_members("team", 1, 1) == Page((member,), 2)
    assert await hierarchy.list_members("team", 2, 1) == Page((second,), 2)
    assert await hierarchy.list_members("team", 3, 1) == Page((), 2)
    assert await hierarchy.list_organizations(1, 1) == Page((organization,), 2)
    assert await hierarchy.list_teams("unknown") == Page((), 0)
    assert await hierarchy.list_members(other_team.customer_team_id) == Page((), 0)
