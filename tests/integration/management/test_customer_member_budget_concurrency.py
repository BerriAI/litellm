import asyncio
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Final
from uuid import uuid4

import pytest
from pydantic import TypeAdapter

from litellm.proxy.management_helpers.customer_hierarchy import CustomerHierarchy, CustomerHierarchyStore, Failure
from litellm.types.proxy.management_endpoints.customer_hierarchy import (
    Budget,
    MemberCreate,
    MemberPatch,
    OrganizationCreate,
    TeamCreate,
)

if TYPE_CHECKING:
    from prisma import Prisma

_WAITERS: Final = (
    "SELECT count(*)::int AS count FROM pg_stat_activity "
    "WHERE $1 = ANY(pg_blocking_pids(pid)) AND query LIKE '%LiteLLM_CustomerOrganizationTable%'"
)
_ROWS: Final = TypeAdapter(tuple[Mapping[str, int], ...])
_NOW: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)


async def _wait_for_organization_contention(db: "Prisma", pid: int) -> None:
    while not _ROWS.validate_python(await db.query_raw(_WAITERS, pid))[0]["count"]:
        await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_member_creation_and_team_move_both_commit_under_contention() -> None:
    from prisma import Prisma

    db: Final = Prisma()
    await db.connect()
    hierarchy: Final = CustomerHierarchy(db, premium_user=True, clock=lambda: _NOW)
    free_members: Final = CustomerHierarchy(db, premium_user=False, clock=lambda: _NOW)
    suffix: Final = uuid4().hex
    org, source, target, member = (f"{name}-{suffix}" for name in ("org", "source", "target", "member"))
    try:
        assert not isinstance(
            await hierarchy.create_organization(OrganizationCreate(customer_org_id=org), "admin"), Failure
        )
        for team in (source, target):
            assert not isinstance(
                await hierarchy.create_team(TeamCreate(customer_team_id=team, parent_org_id=org), "admin"), Failure
            )
        assert not isinstance(
            await hierarchy.create_member(
                MemberCreate(customer_member_id=member, parent_team_ids=(source,), budget=Budget(max_budget=750)),
                "admin",
            ),
            Failure,
        )
        async with db.tx(timeout=timedelta(seconds=30)) as transaction:
            store: Final = CustomerHierarchyStore(transaction, _NOW, premium_user=False)
            await store.lock_customer_identities()
            await store.lock_member(member)
            pid: Final = _ROWS.validate_python(await transaction.query_raw("SELECT pg_backend_pid() AS pid"))[0]["pid"]
            creating: Final = asyncio.create_task(
                free_members.create_member(
                    MemberCreate(
                        customer_member_id=f"new-{suffix}", parent_team_ids=(target,), budget=Budget(max_budget=100)
                    ),
                    "admin",
                )
            )
            try:
                await asyncio.wait_for(_wait_for_organization_contention(db, pid), timeout=10)
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
    finally:
        await hierarchy.delete_organization(org, cascade=True)
        await db.disconnect()


@pytest.mark.asyncio
async def test_member_edit_and_organization_cascade_both_commit_under_contention() -> None:
    from prisma import Prisma

    db: Final = Prisma()
    await db.connect()
    hierarchy: Final = CustomerHierarchy(db, premium_user=True, clock=lambda: _NOW)
    suffix: Final = uuid4().hex
    org, team, member = (f"{name}-{suffix}" for name in ("org", "team", "member"))
    try:
        assert not isinstance(
            await hierarchy.create_organization(OrganizationCreate(customer_org_id=org), "admin"), Failure
        )
        assert not isinstance(
            await hierarchy.create_team(TeamCreate(customer_team_id=team, parent_org_id=org), "admin"), Failure
        )
        assert not isinstance(
            await hierarchy.create_member(
                MemberCreate(customer_member_id=member, parent_team_ids=(team,), budget=Budget(max_budget=750)),
                "admin",
            ),
            Failure,
        )
        async with db.tx(timeout=timedelta(seconds=30)) as transaction:
            store: Final = CustomerHierarchyStore(transaction, _NOW, premium_user=True)
            await store.lock_organization(org)
            pid: Final = _ROWS.validate_python(await transaction.query_raw("SELECT pg_backend_pid() AS pid"))[0]["pid"]
            deleting: Final = asyncio.create_task(hierarchy.delete_organization(org, cascade=True))
            try:
                await asyncio.wait_for(_wait_for_organization_contention(db, pid), timeout=10)
                edited: Final = await store.update_member(member, MemberPatch(alias="Renamed"), "admin")
                assert not isinstance(edited, Failure), edited
                assert edited.alias == "Renamed"
            except BaseException:
                deleting.cancel()
                await asyncio.gather(deleting, return_exceptions=True)
                raise
        assert await deleting is None
        assert isinstance(await hierarchy.get_organization(org), Failure)
        assert isinstance(await hierarchy.get_member(member), Failure)
    finally:
        await hierarchy.delete_organization(org, cascade=True)
        await db.disconnect()
