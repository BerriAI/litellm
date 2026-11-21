import asyncio
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import pytest
import pytest_asyncio
from pydantic import TypeAdapter

from litellm.proxy.management_helpers.customer_hierarchy import (
    CustomerHierarchy,
    CustomerHierarchyStore,
    ErrorCode,
    Failure,
    delete_unattached_customer_budget,
)
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    MemberCreate,
    MemberPatch,
    OrganizationCreate,
    TeamCreate,
    TeamPatch,
)

if TYPE_CHECKING:
    from prisma import Prisma

_WAITERS: Final = "SELECT count(*)::int AS count FROM pg_stat_activity WHERE $1 = ANY(pg_blocking_pids(pid))"
_ROWS: Final = TypeAdapter(tuple[Mapping[str, int], ...])
_NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


async def _wait_for_contention(db: "Prisma", pid: int) -> None:
    while not _ROWS.validate_python(await db.query_raw(_WAITERS, pid))[0]["count"]:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_member_creation_and_team_move_both_commit_under_contention(
    deletion_family: tuple["Prisma", CustomerHierarchy, str, str, str],
) -> None:
    db, hierarchy, org, source, member = deletion_family
    free_members: Final = CustomerHierarchy(db, premium_user=False, clock=lambda: _NOW)
    target: Final = f"target-{org}"
    assert not isinstance(
        await hierarchy.create_team(TeamCreate(customer_team_id=target, parent_org_id=org), org), Failure
    )
    async with db.tx(timeout=timedelta(seconds=30)) as transaction:
        store: Final = CustomerHierarchyStore(transaction, _NOW, premium_user=False)
        await store.lock_customer_identities()
        await store.lock_member(member)
        pid: Final = _ROWS.validate_python(await transaction.query_raw("SELECT pg_backend_pid() AS pid"))[0]["pid"]
        creating: Final = asyncio.create_task(
            free_members.create_member(
                MemberCreate(customer_member_id=f"new-{org}", parent_team_ids=(target,), budget=Budget(max_budget=100)),
                org,
            )
        )
        try:
            await asyncio.wait_for(_wait_for_contention(db, pid), timeout=10)
            moved: Final = await store.update_member(member, MemberPatch(parent_team_ids=(target,)), "admin")
            assert not isinstance(moved, Failure), moved
            assert moved.parent_team_ids == (target,)
        except BaseException:
            creating.cancel()
            await asyncio.gather(creating, return_exceptions=True)
            raise
    created: Final = await creating
    assert not isinstance(created, Failure), created
    assert created.parent_team_ids == (target,)
    stored: Final = await hierarchy.get_member(member)
    assert not isinstance(stored, Failure), stored
    assert stored.parent_team_ids == (target,)


@pytest_asyncio.fixture(loop_scope="function")
async def deletion_family() -> AsyncIterator[tuple["Prisma", CustomerHierarchy, str, str, str]]:
    from prisma import Prisma

    db: Final = Prisma()
    await db.connect()
    hierarchy: Final = CustomerHierarchy(db, premium_user=True, clock=lambda: _NOW)
    suffix: Final = uuid4().hex
    org, team, member = (f"{name}-{suffix}" for name in ("org", "team", "member"))
    try:
        created: Final = (
            await hierarchy.create_organization(
                OrganizationCreate(customer_org_id=org, budget=Budget(max_budget=1000)), org
            ),
            await hierarchy.create_team(TeamCreate(customer_team_id=team, parent_org_id=org), org),
            await hierarchy.create_member(
                MemberCreate(customer_member_id=member, parent_team_ids=(team,), budget=Budget(max_budget=100)), org
            ),
        )
        assert all(not isinstance(row, Failure) for row in created), created
        yield db, hierarchy, org, team, member
    finally:
        await db.litellm_apptable.delete_many(where={"created_by": org})
        await db.litellm_customermembertable.delete_many(where={"parent_org_id": org})
        await hierarchy.delete_organization(org, cascade=True)
        await db.litellm_budgettable.delete_many(where={"created_by": org})
        await db.disconnect()


async def _delete_family(hierarchy: CustomerHierarchy, resource: str, org: str, team: str, member: str) -> None:
    result: Final = (
        await hierarchy.delete_member(member)
        if resource == "member"
        else await hierarchy.delete_team(team, cascade=True)
        if resource == "team"
        else await hierarchy.delete_organization(org, cascade=True)
    )
    assert result is None, result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("resource", "edit_team"),
    (("member", False), ("team", False), ("organization", False), ("organization_no_team", False), ("team", True)),
)
async def test_deletion_reclaims_existing_and_inflight_budgets_but_preserves_shared_profiles(
    deletion_family: tuple["Prisma", CustomerHierarchy, str, str, str], resource: str, edit_team: bool
) -> None:
    db, hierarchy, org, team, member = deletion_family
    if resource == "organization_no_team":
        await db.litellm_customermembertable.update(where={"customer_member_id": member}, data={"parent_team_ids": []})
        await db.litellm_customerteamtable.delete(where={"customer_team_id": team})
    store: Final = CustomerHierarchyStore(db, _NOW, premium_user=True)
    shared, member_base, member_override, org_base, org_override, org_member = tuple(
        [await store.create_budget(Budget(max_budget=maximum), org) for maximum in range(10, 16)]
    )
    application: Final = await db.litellm_apptable.create(
        data={"alias": org, "default_tier_budget_id": shared, "created_by": org, "updated_by": org}
    )
    await db.litellm_appcustomerorgtable.create(
        data={
            "app_id": application.app_id,
            "customer_org_id": org,
            "tier_budget_id": shared,
            "budget_id": org_base,
            "budget_override_id": org_override,
            "customer_member_budget_id": org_member,
        }
    )
    await db.litellm_appcustomermembertable.create(
        data={
            "app_id": application.app_id,
            "customer_member_id": member,
            "budget_id": member_base,
            "budget_override_id": member_override,
        }
    )
    global_org: Final = await hierarchy.get_organization(org)
    assert not isinstance(global_org, Failure)
    if resource == "organization_no_team":
        assert await hierarchy.delete_organization(org) == Failure(ErrorCode.HAS_CHILDREN)
        assert await db.litellm_customermembertable.find_unique(where={"customer_member_id": member}) is not None
    surviving: Final = frozenset({shared}) | (
        frozenset({org_base, org_override, org_member, global_org.budget_id})
        if not resource.startswith("organization")
        else frozenset()
    )
    async with db.tx(timeout=timedelta(seconds=30)) as transaction:
        editing: Final = CustomerHierarchyStore(transaction, _NOW, premium_user=True)
        await editing.lock_member(member)
        if edit_team:
            await transaction.query_raw(
                'SELECT 1 FROM "LiteLLM_CustomerTeamTable" WHERE customer_team_id = $1 FOR UPDATE', team
            )
        pid: Final = _ROWS.validate_python(await transaction.query_raw("SELECT pg_backend_pid() AS pid"))[0]["pid"]
        deleting: Final = asyncio.create_task(_delete_family(hierarchy, resource, org, team, member))
        try:
            await asyncio.wait_for(_wait_for_contention(db, pid), timeout=10)
            edited: Final = (
                await editing.update_team(team, TeamPatch(budget=Budget(max_budget=300)), org)
                if edit_team
                else await editing.update_member(member, MemberPatch(budget_override=Budget(max_budget=300)), org)
            )
            assert not isinstance(edited, Failure), edited
            new_budget: Final = edited.budget_id if edit_team else edited.budget_override_id
            assert new_budget is not None
        except BaseException:
            deleting.cancel()
            await asyncio.gather(deleting, return_exceptions=True)
            raise
    await asyncio.wait_for(deleting, timeout=10)
    assert await db.litellm_appcustomermembertable.count(where={"app_id": application.app_id}) == 0
    assert (
        frozenset(row.budget_id for row in await db.litellm_budgettable.find_many(where={"created_by": org}))
        == surviving
    )


@pytest.mark.asyncio
async def test_budget_delete_rejects_locked_app_budget_without_blocking_edit(
    deletion_family: tuple["Prisma", CustomerHierarchy, str, str, str],
) -> None:
    db, _, org, _, member = deletion_family
    store: Final = CustomerHierarchyStore(db, _NOW, premium_user=True)
    budget_id: Final = await store.create_budget(Budget(max_budget=10), org)
    application: Final = await db.litellm_apptable.create(data={"alias": org, "created_by": org, "updated_by": org})
    await db.litellm_appcustomermembertable.create(
        data={
            "app_id": application.app_id,
            "customer_member_id": member,
            "budget_id": budget_id,
        }
    )
    async with db.tx(timeout=timedelta(seconds=30)) as transaction:
        editing: Final = CustomerHierarchyStore(transaction, _NOW, premium_user=True)
        await editing.lock_organization(org)
        await transaction.query_raw(
            'SELECT 1 FROM "LiteLLM_AppCustomerMemberTable" WHERE app_id = $1 AND customer_member_id = $2 FOR UPDATE',
            application.app_id,
            member,
        )
        rejected: Final = await asyncio.wait_for(delete_unattached_customer_budget(store.budgets, budget_id), timeout=5)
        assert rejected == Failure(ErrorCode.HAS_CHILDREN)
        await editing.update_budget(budget_id, Budget(max_budget=20), org)
    kept: Final = await db.litellm_budgettable.find_unique(where={"budget_id": budget_id})
    assert kept is not None and kept.max_budget == 20
