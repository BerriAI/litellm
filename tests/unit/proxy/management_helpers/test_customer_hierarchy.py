from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Final, cast

import pytest
from pydantic import BaseModel, ConfigDict

from litellm.proxy.management_helpers.customer_hierarchy import (
    Budget,
    BudgetTemplate,
    CustomerHierarchy,
    CustomerMember,
    Failure,
    Kind,
    MemberCreate,
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
    def __init__(self, id_field: str) -> None:
        self.id_field = id_field
        self.rows: Mapping[str, Record] = {}

    async def find_unique(self, where: Mapping[str, object]) -> Record | None:
        return self.rows.get(str(where[self.id_field]))

    async def create(self, data: Mapping[str, object]) -> Record:
        from prisma.errors import UniqueViolationError

        if str(data[self.id_field]) in self.rows:
            raise UniqueViolationError({})
        row: Final = Record.model_validate({"created_at": NOW, "updated_at": NOW, **data})
        self.rows = {**self.rows, str(data[self.id_field]): row}
        return row

    async def update(self, where: Mapping[str, object], data: Mapping[str, object]) -> Record | None:
        row_id: Final = str(where[self.id_field])
        if row_id not in self.rows:
            return None
        row: Final = self.rows[row_id].model_copy(update=dict(data))
        self.rows = {**self.rows, row_id: row}
        return row

    async def find_many(
        self, where: Mapping[str, object], order: Mapping[str, object] | None = None
    ) -> tuple[Record, ...]:
        return tuple(row for row in self.rows.values() if all(row.model_dump().get(k) == v for k, v in where.items()))

    async def delete_many(self, where: Mapping[str, object]) -> int:
        matched: Final = await self.find_many(where)
        self.rows = {key: row for key, row in self.rows.items() if row not in matched}
        return len(matched)


class Database:
    def __init__(self) -> None:
        self.litellm_customerorganizationtable = Table("customer_org_id")
        self.litellm_customerteamtable = Table("customer_team_id")
        self.litellm_customermembertable = Table("customer_member_id")
        self.litellm_budgettable = Table("budget_id")
        self.litellm_customerorganizationbudgettable = Table("budget_id")

    @asynccontextmanager
    async def tx(self) -> AsyncIterator["Database"]:
        yield self


@pytest.fixture
def database() -> Database:
    return Database()


@pytest.fixture
def hierarchy(database: Database) -> CustomerHierarchy:
    return CustomerHierarchy(cast("Prisma", database))  # cast-ok: in-memory implementation of the Prisma edge


async def family(hierarchy: CustomerHierarchy, budget: Budget | None = None) -> tuple[Organization, Team]:
    org: Final = await hierarchy.write(
        Kind.ORGANIZATION,
        OrganizationCreate(
            customer_org_id="org",
            default_budgets=(BudgetTemplate(**budget.model_dump()),) if budget is not None else (),
        ),
        "admin",
    )
    team: Final = await hierarchy.write(
        Kind.TEAM, TeamCreate(customer_team_id="team", parent_org_id="org", budget=Budget(max_budget=999)), "admin"
    )
    assert isinstance(org, Organization)
    assert isinstance(team, Team)
    return org, team


@pytest.mark.asyncio
async def test_member_copies_org_default_not_team_budget(hierarchy: CustomerHierarchy) -> None:
    org, team = await family(
        hierarchy, Budget(max_budget=100, tpm_limit=1000, rpm_limit=10, allowed_models=("model",), budget_duration="1d")
    )
    member: Final = await hierarchy.write(Kind.MEMBER, MemberCreate(parent_team_id=team.customer_team_id), "admin")
    assert isinstance(member, CustomerMember)
    assert member.budget_id is not None
    assert member.budget_id != org.default_budget_id
    assert member.budget == Budget(
        max_budget=100, tpm_limit=1000, rpm_limit=10, allowed_models=("model",), budget_duration="1d"
    )


@pytest.mark.asyncio
async def test_member_request_overrides_copied_defaults(hierarchy: CustomerHierarchy) -> None:
    org, team = await family(hierarchy, Budget(max_budget=100, rpm_limit=10, allowed_models=("model",)))
    member: Final = await hierarchy.write(
        Kind.MEMBER,
        MemberCreate(parent_team_id=team.customer_team_id, budget=Budget(max_budget=12, rpm_limit=None)),
        "admin",
    )
    assert isinstance(member, CustomerMember)
    assert member.budget == Budget(max_budget=12, rpm_limit=None, allowed_models=("model",))
    assert await hierarchy.get(Kind.ORGANIZATION, org.customer_org_id) == org


@pytest.mark.asyncio
async def test_member_patch_only_updates_its_own_budget(hierarchy: CustomerHierarchy) -> None:
    org, team = await family(hierarchy, Budget(max_budget=100, allowed_models=("model",)))
    member: Final = await hierarchy.write(Kind.MEMBER, MemberCreate(parent_team_id=team.customer_team_id), "admin")
    assert isinstance(member, CustomerMember)
    patched: Final = await hierarchy.write(
        Kind.MEMBER, TeamPatch(budget=Budget(max_budget=5)), "admin", member.customer_member_id
    )
    assert isinstance(patched, CustomerMember)
    assert patched.budget_id == member.budget_id
    assert patched.budget == Budget(max_budget=5, allowed_models=("model",))
    assert await hierarchy.get(Kind.ORGANIZATION, org.customer_org_id) == org


@pytest.mark.asyncio
async def test_org_patch_leaves_existing_members_unchanged(hierarchy: CustomerHierarchy) -> None:
    org, team = await family(hierarchy, Budget(max_budget=100))
    member: Final = await hierarchy.write(Kind.MEMBER, MemberCreate(parent_team_id=team.customer_team_id), "admin")
    assert isinstance(member, CustomerMember)
    patched: Final = await hierarchy.write(
        Kind.ORGANIZATION,
        OrganizationPatch(default_budgets=(BudgetTemplate(max_budget=50),)),
        "admin",
        org.customer_org_id,
    )
    assert isinstance(patched, Organization)
    assert len(patched.default_budgets) == 1
    assert patched.default_budgets[0].max_budget == 50
    assert patched.default_budget_id != org.default_budget_id
    assert await hierarchy.get(Kind.MEMBER, member.customer_member_id) == member
    new_member: Final = await hierarchy.write(Kind.MEMBER, MemberCreate(parent_team_id=team.customer_team_id), "admin")
    assert isinstance(new_member, CustomerMember)
    assert new_member.budget == Budget(max_budget=50, allowed_models=())


@pytest.mark.asyncio
async def test_member_without_defaults_still_gets_own_budget(hierarchy: CustomerHierarchy) -> None:
    _, team = await family(hierarchy)
    member: Final = await hierarchy.write(Kind.MEMBER, MemberCreate(parent_team_id=team.customer_team_id), "admin")
    assert isinstance(member, CustomerMember)
    assert member.budget_id is not None
    assert member.budget == Budget(allowed_models=())


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "payload"),
    (
        (Kind.TEAM, TeamCreate(parent_org_id="missing")),
        (Kind.MEMBER, MemberCreate(parent_team_id="missing")),
    ),
)
async def test_missing_parent_returns_404(
    hierarchy: CustomerHierarchy, kind: Kind, payload: TeamCreate | MemberCreate
) -> None:
    assert await hierarchy.write(kind, payload, "admin") == Failure(404, "Customer entity not found.")


@pytest.mark.asyncio
async def test_patch_missing_member_returns_404(hierarchy: CustomerHierarchy) -> None:
    assert await hierarchy.write(Kind.MEMBER, TeamPatch(budget=Budget()), "admin", "missing") == Failure(
        404, "Customer entity not found."
    )


@pytest.mark.asyncio
async def test_patch_creates_budget_for_member_without_one(hierarchy: CustomerHierarchy, database: Database) -> None:
    await family(hierarchy)
    await database.litellm_customermembertable.create(
        data={"customer_member_id": "legacy", "parent_team_id": "team", "created_by": "admin", "updated_by": "admin"}
    )
    member: Final = await hierarchy.write(Kind.MEMBER, TeamPatch(budget=Budget(rpm_limit=7)), "admin", "legacy")
    assert isinstance(member, CustomerMember)
    assert member.budget_id is not None
    assert member.budget == Budget(rpm_limit=7, allowed_models=())


@pytest.mark.asyncio
async def test_team_patch_preserves_other_budget_fields(hierarchy: CustomerHierarchy) -> None:
    _, team = await family(hierarchy)
    patched: Final = await hierarchy.write(
        Kind.TEAM, TeamPatch(budget=Budget(rpm_limit=7)), "admin", team.customer_team_id
    )
    assert isinstance(patched, Team)
    assert patched.budget_id == team.budget_id
    assert patched.budget == Budget(max_budget=999, rpm_limit=7, allowed_models=())


@pytest.mark.asyncio
async def test_member_can_select_a_non_default_org_template(hierarchy: CustomerHierarchy) -> None:
    org: Final = await hierarchy.write(
        Kind.ORGANIZATION,
        OrganizationCreate(
            customer_org_id="org",
            default_budgets=(
                BudgetTemplate(max_budget=100, is_default=True),
                BudgetTemplate(max_budget=20, rpm_limit=8),
            ),
        ),
        "admin",
    )
    assert isinstance(org, Organization)
    template: Final = next(budget for budget in org.default_budgets if not budget.is_default)
    team: Final = await hierarchy.write(Kind.TEAM, TeamCreate(parent_org_id="org"), "admin")
    assert isinstance(team, Team)
    member: Final = await hierarchy.write(
        Kind.MEMBER,
        MemberCreate(
            parent_team_id=team.customer_team_id, budget_template_id=template.budget_id, budget=Budget(rpm_limit=4)
        ),
        "admin",
    )
    assert isinstance(member, CustomerMember)
    assert member.budget_id not in tuple(budget.budget_id for budget in org.default_budgets)
    assert member.budget == Budget(max_budget=20, rpm_limit=4, allowed_models=())


@pytest.mark.asyncio
async def test_member_rejects_template_from_another_org(hierarchy: CustomerHierarchy) -> None:
    _, team = await family(hierarchy, Budget(max_budget=100))
    other: Final = await hierarchy.write(
        Kind.ORGANIZATION,
        OrganizationCreate(default_budgets=(BudgetTemplate(max_budget=1),)),
        "admin",
    )
    assert isinstance(other, Organization)
    rejected: Final = await hierarchy.write(
        Kind.MEMBER,
        MemberCreate(parent_team_id=team.customer_team_id, budget_template_id=other.default_budget_id),
        "admin",
    )
    assert rejected == Failure(422, "Budget template must belong to the member's organization.")


@pytest.mark.asyncio
async def test_replacing_org_templates_unlinks_old_set_without_changing_member(hierarchy: CustomerHierarchy) -> None:
    org, team = await family(hierarchy, Budget(max_budget=100))
    member: Final = await hierarchy.write(Kind.MEMBER, MemberCreate(parent_team_id=team.customer_team_id), "admin")
    assert isinstance(member, CustomerMember)
    cleared: Final = await hierarchy.write(
        Kind.ORGANIZATION, OrganizationPatch(default_budgets=()), "admin", org.customer_org_id
    )
    assert isinstance(cleared, Organization)
    assert cleared.default_budgets == ()
    assert cleared.default_budget_id is None
    assert await hierarchy.get(Kind.MEMBER, member.customer_member_id) == member
    rejected: Final = await hierarchy.write(
        Kind.MEMBER,
        MemberCreate(parent_team_id=team.customer_team_id, budget_template_id=org.default_budget_id),
        "admin",
    )
    assert rejected == Failure(422, "Budget template must belong to the member's organization.")


@pytest.mark.asyncio
async def test_org_alias_patch_keeps_templates(hierarchy: CustomerHierarchy) -> None:
    org, _ = await family(hierarchy, Budget(max_budget=100))
    patched: Final = await hierarchy.write(
        Kind.ORGANIZATION, OrganizationPatch(alias="renamed"), "admin", org.customer_org_id
    )
    assert isinstance(patched, Organization)
    assert patched.alias == "renamed"
    assert patched.default_budgets == org.default_budgets
    assert patched.default_budget_id == org.default_budget_id
