from typing import TYPE_CHECKING, Final, cast

import pytest

from litellm.proxy.auth.customer_member_auto_create import default_team_id, ensure_customer_member
from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy, ErrorCode, Failure
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    BudgetTemplate,
    CustomerMember,
    OrganizationCreate,
    TeamCreate,
)
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW, Database

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


def _prisma(database: Database) -> "PrismaClient":
    return cast("PrismaClient", database)  # cast-ok: in-memory Prisma edge


async def _organization(database: Database, org_id: str, default_budget: bool = True) -> None:
    templates: Final = (BudgetTemplate(name="std", max_budget=10.0, is_default=True),) if default_budget else ()
    created: Final = await CustomerHierarchy(_prisma(database), clock=lambda: NOW).create_organization(
        OrganizationCreate(customer_org_id=org_id, default_budgets=templates), "admin"
    )
    assert not isinstance(created, Failure)


@pytest.mark.asyncio
async def test_a_new_member_joins_the_organizations_default_team_on_the_organizations_default_budget() -> None:
    database: Final = Database()
    await _organization(database, "org-1")

    member: Final = await ensure_customer_member("org-1", "user-7", _prisma(database))

    assert isinstance(member, CustomerMember)
    assert (member.customer_member_id, member.parent_team_id) == ("user-7", default_team_id("org-1"))
    assert member.effective_budget.max_budget == 10.0
    team: Final = database.litellm_customerteamtable.rows[default_team_id("org-1")].model_dump()
    assert (team["parent_org_id"], team["alias"]) == ("org-1", "default")


@pytest.mark.asyncio
async def test_registering_the_same_member_twice_keeps_one_row_and_reuses_an_existing_default_team() -> None:
    database: Final = Database()
    await _organization(database, "org-1")
    hierarchy: Final = CustomerHierarchy(_prisma(database), clock=lambda: NOW)
    existing_team: Final = await hierarchy.create_team(
        TeamCreate(customer_team_id=default_team_id("org-1"), alias="pre-existing", parent_org_id="org-1"), "admin"
    )
    assert not isinstance(existing_team, Failure)

    first: Final = await ensure_customer_member("org-1", "user-7", _prisma(database))
    second: Final = await ensure_customer_member("org-1", "user-7", _prisma(database))

    assert isinstance(first, CustomerMember) and isinstance(second, CustomerMember)
    assert first.parent_team_id == second.parent_team_id == default_team_id("org-1")
    assert tuple(database.litellm_customermembertable.rows) == ("user-7",)
    assert tuple(database.litellm_customerteamtable.rows) == (default_team_id("org-1"),)


@pytest.mark.asyncio
async def test_an_organization_without_a_default_budget_cannot_register_members() -> None:
    database: Final = Database()
    await _organization(database, "org-1", default_budget=False)

    result: Final = await ensure_customer_member("org-1", "user-7", _prisma(database))

    assert result == Failure(ErrorCode.BUDGET_REQUIRED)
    assert "user-7" not in database.litellm_customermembertable.rows


@pytest.mark.asyncio
async def test_an_unknown_organization_cannot_register_members() -> None:
    assert await ensure_customer_member("ghost", "user-7", _prisma(Database())) == Failure(ErrorCode.PARENT_NOT_FOUND)
