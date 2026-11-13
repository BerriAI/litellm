from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import pytest

from litellm.proxy.auth.customer_hierarchy_auth import load_customer_hierarchy
from litellm.proxy.auth.customer_member_auto_create import (
    default_team_id,
    ensure_customer_member,
    ensure_customer_organization,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy, ErrorCode, Failure
from litellm.types.proxy.auth.customer_hierarchy import CustomerRequestIdentity
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    BudgetTemplate,
    CustomerMember,
    MemberCreate,
    MemberPatch,
    OrganizationCreate,
    TeamCreate,
)
from tests.unit.proxy.management_helpers.test_customer_hierarchy import NOW, Database

if TYPE_CHECKING:
    from litellm.proxy.utils import PrismaClient


def _prisma(database: Database) -> "PrismaClient":
    return cast("PrismaClient", database)  # cast-ok: in-memory Prisma edge


@pytest.mark.asyncio
@pytest.mark.parametrize(("org_id", "member_id"), (("org-1", "foo/bar"), ("foo/bar", "member")))
async def test_slash_ids_cannot_leave_a_default_team(org_id: str, member_id: str) -> None:
    database: Final = Database()
    await _organization(database, "org-1")
    result: Final = await ensure_customer_member(org_id, member_id, _prisma(database))
    assert result == Failure(ErrorCode.NOT_FOUND)
    assert not database.litellm_customerteamtable.rows
    assert not database.litellm_customermembertable.rows


@pytest.mark.asyncio
async def test_slash_alias_cannot_register_an_unreachable_organization() -> None:
    database: Final = Database()
    result: Final = await ensure_customer_organization("foo/bar", _prisma(database))
    assert result == Failure(ErrorCode.NOT_FOUND)
    assert not database.litellm_customerorganizationtable.rows


@pytest.mark.asyncio
@pytest.mark.parametrize("known_org", (False, True))
async def test_unregistrable_member_traffic_keeps_the_known_organization_or_runs_untracked(known_org: bool) -> None:
    database: Final = Database()
    created: Final = await CustomerHierarchy(_prisma(database), clock=lambda: NOW).create_organization(
        OrganizationCreate(customer_org_id="org-1", alias="org-1"), "admin"
    )
    assert not isinstance(created, Failure)
    prisma: Final = cast("PrismaClient", SimpleNamespace(db=database))  # cast-ok: in-memory Prisma edge
    result: Final = await load_customer_hierarchy(
        CustomerRequestIdentity(customer_org_alias="org-1" if known_org else "foo/bar", customer_member_id="foo/bar"),
        prisma,
        UserApiKeyCache(),
    )
    assert (result.customer_org_id if result is not None else None) == ("org-1" if known_org else None)
    assert tuple(database.litellm_customerorganizationtable.rows) == ("org-1",)
    assert not database.litellm_customerteamtable.rows
    assert not database.litellm_customermembertable.rows


async def _organization(database: Database, org_id: str, default_budget: bool = True) -> None:
    templates: Final = (BudgetTemplate(name="std", max_budget=10.0),) if default_budget else ()
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
    assert (member.customer_member_id, member.parent_team_ids) == ("user-7", (default_team_id("org-1"),))
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
    assert first.parent_team_ids == second.parent_team_ids == (default_team_id("org-1"),)
    assert tuple(database.litellm_customermembertable.rows) == ("user-7",)
    assert tuple(database.litellm_customerteamtable.rows) == (default_team_id("org-1"),)


@pytest.mark.asyncio
async def test_a_new_member_of_an_organization_without_a_default_budget_is_registered_uncapped() -> None:
    database: Final = Database()
    await _organization(database, "org-1", default_budget=False)

    member: Final = await ensure_customer_member("org-1", "user-7", _prisma(database))

    assert isinstance(member, CustomerMember), member
    assert (member.parent_org_id, member.parent_team_ids) == ("org-1", (default_team_id("org-1"),))
    assert (member.budget_id, member.budget_override_id, member.effective_budget.max_budget) == (None, None, None)


@pytest.mark.asyncio
async def test_an_admin_created_member_still_needs_a_budget_and_an_uncapped_member_stays_editable() -> None:
    database: Final = Database()
    await _organization(database, "org-1", default_budget=False)
    hierarchy: Final = CustomerHierarchy(_prisma(database), clock=lambda: NOW)
    registered: Final = await ensure_customer_member("org-1", "user-7", _prisma(database))
    assert isinstance(registered, CustomerMember), registered

    by_admin: Final = await hierarchy.create_member(
        MemberCreate(customer_member_id="user-8", parent_team_ids=registered.parent_team_ids), "admin"
    )
    renamed: Final = await hierarchy.update_member("user-7", MemberPatch(alias="Seven"), "admin")

    assert by_admin == Failure(ErrorCode.BUDGET_REQUIRED)
    assert isinstance(renamed, CustomerMember) and renamed.alias == "Seven", renamed


@pytest.mark.asyncio
async def test_an_unknown_organization_cannot_register_members() -> None:
    assert await ensure_customer_member("ghost", "user-7", _prisma(Database())) == Failure(ErrorCode.PARENT_NOT_FOUND)


@pytest.mark.asyncio
async def test_an_unknown_alias_becomes_an_organization_once_and_a_known_one_is_reused() -> None:
    database: Final = Database()

    created: Final = await ensure_customer_organization("acme", _prisma(database))
    again: Final = await ensure_customer_organization("acme", _prisma(database))

    assert not isinstance(created, Failure) and not isinstance(again, Failure)
    assert (created.customer_org_id, created.alias, again.customer_org_id) == ("acme", "acme", "acme")
    listed: Final = await CustomerHierarchy(_prisma(database), clock=lambda: NOW).list_organizations(
        page=1, page_size=10
    )
    assert not isinstance(listed, Failure) and len(listed.data) == 1


@pytest.mark.asyncio
async def test_renaming_a_member_keeps_a_budget_assigned_after_the_rename_read_the_member() -> None:
    database: Final = Database()
    await _organization(database, "org-1", default_budget=False)
    hierarchy: Final = CustomerHierarchy(_prisma(database), clock=lambda: NOW)
    registered: Final = await ensure_customer_member("org-1", "user-7", _prisma(database))
    assert isinstance(registered, CustomerMember), registered
    capped: Final = await hierarchy.update_member("user-7", MemberPatch(budget_override=Budget(max_budget=3.0)), "a")
    assert isinstance(capped, CustomerMember), capped

    written: Final = await hierarchy.update_member("user-7", MemberPatch(alias="Seven"), "admin")

    assert isinstance(written, CustomerMember), written
    row: Final = database.litellm_customermembertable.rows["user-7"].model_dump()
    assert (written.alias, row["budget_override_id"], written.effective_budget.max_budget) == (
        "Seven",
        capped.budget_override_id,
        3.0,
    )
