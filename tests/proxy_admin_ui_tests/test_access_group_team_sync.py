"""
Real-Postgres coverage for the team -> access group mirror.

`sync_team_access_group_membership` reconciles `assigned_team_ids` with two raw
statements, and a mocked prisma cannot tell whether that SQL is right: a fake has to
reimplement the array semantics in Python, so it passes no matter what the SQL says.
These tests run the statements against the same Postgres CI seeds for the admin UI
suite, which is the only place a `NOT (... = ANY(...))` guard going missing shows up.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest


from litellm.proxy.management_helpers.access_group_team_sync import (
    reconcile_team_access_group_membership,
    sync_team_access_group_membership,
)

TEAM = "ags-team-a"
OTHER_TEAM = "ags-team-b"
GROUPS = ("ags-group-1", "ags-group-2", "ags-group-3")
_DELETE_SEEDED = 'DELETE FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = ANY($1::TEXT[])'
_DELETE_TEAMS = 'DELETE FROM "LiteLLM_TeamTable" WHERE team_id = ANY($1::TEXT[])'


@asynccontextmanager
async def _clean_db():
    """Connects inside the running test's loop. An async fixture would be torn up on a
    different loop than the test body, which prisma's engine lock refuses outright."""
    from prisma import Prisma

    if not os.getenv("DATABASE_URL"):
        pytest.fail("DATABASE_URL is required; these tests must not silently skip")

    db = Prisma()
    await db.connect()
    try:
        await db.execute_raw(_DELETE_SEEDED, list(GROUPS))
        await db.execute_raw(_DELETE_TEAMS, [TEAM, OTHER_TEAM])
        yield db
    finally:
        await db.execute_raw(_DELETE_SEEDED, list(GROUPS))
        await db.execute_raw(_DELETE_TEAMS, [TEAM, OTHER_TEAM])
        await db.disconnect()


async def _seed(db, assignments):
    for group_id, team_ids in assignments.items():
        await db.litellm_accessgrouptable.create(
            data={
                "access_group_id": group_id,
                "access_group_name": group_id,
                "assigned_team_ids": team_ids,
            }
        )


async def _read(db):
    rows = await db.query_raw(
        'SELECT access_group_id, assigned_team_ids FROM "LiteLLM_AccessGroupTable" '
        "WHERE access_group_id = ANY($1::TEXT[])",
        list(GROUPS),
    )
    return {row["access_group_id"]: sorted(row["assigned_team_ids"] or []) for row in rows}


async def _set_team_groups(db, team_id, access_group_ids):
    """The mirror reads the committed team row, so the desired state is written there."""
    if access_group_ids is None:
        await db.execute_raw(_DELETE_TEAMS, [team_id])
        return
    await db.litellm_teamtable.upsert(
        where={"team_id": team_id},
        data={
            "create": {"team_id": team_id, "access_group_ids": list(access_group_ids)},
            "update": {"access_group_ids": list(access_group_ids)},
        },
    )


async def _sync(db, team_id, access_group_ids):
    await _set_team_groups(db, team_id, access_group_ids)
    with patch(
        "litellm.proxy.management_helpers.access_group_team_sync.invalidate_access_group_cache",
        new_callable=AsyncMock,
    ) as invalidate:
        await sync_team_access_group_membership(prisma_client=SimpleNamespace(db=db), team_id=team_id)
    return {call.args[0] for call in invalidate.call_args_list}


@pytest.mark.asyncio
async def test_reconcile_attaches_and_detaches_without_touching_other_teams():
    """The detach must be scoped to groups the team dropped. Losing that scope would
    strip the team from the very groups it just kept, silently revoking live grants."""
    async with _clean_db() as db:
        await _seed(db, {GROUPS[0]: [TEAM, OTHER_TEAM], GROUPS[1]: [TEAM], GROUPS[2]: [OTHER_TEAM]})

        invalidated = await _sync(db, TEAM, [GROUPS[1], GROUPS[2]])

        assert await _read(db) == {
            GROUPS[0]: [OTHER_TEAM],
            GROUPS[1]: [TEAM],
            GROUPS[2]: sorted([TEAM, OTHER_TEAM]),
        }
        assert invalidated == {GROUPS[0], GROUPS[1], GROUPS[2]}


@pytest.mark.asyncio
async def test_reconcile_is_idempotent_so_a_retry_heals_rather_than_duplicates():
    """Reconciling to the same desired state twice must leave the rows alone and still name
    the team's groups for the cache step, so a retry after a failed cache drop reaches them.
    A delta-based mirror would instead go quiet once the rows match, leaving the caches
    serving a grant the admin already revoked."""
    async with _clean_db() as db:
        await _seed(db, {GROUPS[0]: [], GROUPS[1]: [TEAM], GROUPS[2]: []})

        first = await _sync(db, TEAM, [GROUPS[0], GROUPS[1]])
        after_first = await _read(db)
        second = await _sync(db, TEAM, [GROUPS[0], GROUPS[1]])

        assert after_first == {GROUPS[0]: [TEAM], GROUPS[1]: [TEAM], GROUPS[2]: []}
        assert await _read(db) == after_first
        assert first == {GROUPS[0], GROUPS[1]}
        assert second == first


@pytest.mark.asyncio
async def test_reconcile_handles_a_null_array_column():
    """`assigned_team_ids` is nullable in Postgres. Without COALESCE both statements
    evaluate their guard to NULL, skip the row, and the grant silently never syncs."""
    async with _clean_db() as db:
        await _seed(db, {GROUPS[0]: [], GROUPS[1]: []})
        await db.execute_raw(
            'UPDATE "LiteLLM_AccessGroupTable" SET assigned_team_ids = NULL WHERE access_group_id = $1',
            GROUPS[0],
        )

        await _sync(db, TEAM, [GROUPS[0]])

        assert await _read(db) == {GROUPS[0]: [TEAM], GROUPS[1]: []}


@pytest.mark.asyncio
async def test_passing_none_detaches_the_team_from_every_group():
    """Team deletion. A group the deleted row never listed must still let the team go,
    otherwise the id dangles under Attached Teams and grants again if it is reused."""
    async with _clean_db() as db:
        await _seed(db, {GROUPS[0]: [TEAM, OTHER_TEAM], GROUPS[1]: [TEAM], GROUPS[2]: [OTHER_TEAM]})

        invalidated = await _sync(db, TEAM, None)

        assert await _read(db) == {GROUPS[0]: [OTHER_TEAM], GROUPS[1]: [], GROUPS[2]: [OTHER_TEAM]}
        assert invalidated == {GROUPS[0], GROUPS[1]}


@pytest.mark.asyncio
async def test_a_failed_mirror_takes_the_new_team_row_with_it():
    """`/team/new` inserts the team and mirrors it in one transaction. Mirroring in a
    transaction of its own instead leaves a committed team whose groups never learned about
    it, and the retry with that same team id comes back as a duplicate."""
    async with _clean_db() as db:
        await _seed(db, {GROUPS[0]: [], GROUPS[1]: [OTHER_TEAM]})

        async def _blow_up_after_reconcile():
            async with db.tx() as tx:
                await tx.litellm_teamtable.create(data={"team_id": TEAM, "access_group_ids": [GROUPS[0]]})
                await reconcile_team_access_group_membership(tx, TEAM)
                raise RuntimeError("the cache handoff blew up")

        with pytest.raises(RuntimeError):
            await _blow_up_after_reconcile()

        assert await _read(db) == {GROUPS[0]: [], GROUPS[1]: [OTHER_TEAM]}
        assert await db.litellm_teamtable.find_unique(where={"team_id": TEAM}) is None


@pytest.mark.asyncio
async def test_a_concurrent_writer_cannot_replay_a_stale_team_row_over_a_newer_one():
    """
    Two writers edit one team at once. Whichever team row commits last is the admin's
    final intent and the mirror must match it, so the mirror has to hold the team's
    advisory lock across its read and its writes.

    A second connection holds that lock and changes the team underneath, which pins the
    interleaving instead of hoping a sleep lands in the gap. With the lock the sync waits
    and then reads the new row. Without it the sync reads the old row and writes a group
    the admin already moved off, which keeps granting to that team.
    """
    from prisma import Prisma

    async with _clean_db() as db:
        await _seed(db, {GROUPS[0]: [], GROUPS[1]: []})
        await _sync(db, TEAM, [GROUPS[0]])
        assert await _read(db) == {GROUPS[0]: [TEAM], GROUPS[1]: []}

        blocker = Prisma()
        await blocker.connect()
        sync_started = asyncio.Event()

        async def competing_sync():
            sync_started.set()
            with patch(
                "litellm.proxy.management_helpers.access_group_team_sync.invalidate_access_group_cache",
                new_callable=AsyncMock,
            ):
                await sync_team_access_group_membership(prisma_client=SimpleNamespace(db=db), team_id=TEAM)

        try:
            async with blocker.tx(timeout=timedelta(seconds=30)) as held:
                await held.query_raw("SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked", TEAM)
                task = asyncio.create_task(competing_sync())
                await sync_started.wait()
                await asyncio.sleep(0.2)
                assert not task.done(), "the mirror did not wait on the team's advisory lock"
                await held.execute_raw(
                    'UPDATE "LiteLLM_TeamTable" SET access_group_ids = $1 WHERE team_id = $2',
                    [GROUPS[1]],
                    TEAM,
                )
            await asyncio.wait_for(task, timeout=30)
        finally:
            await blocker.disconnect()

        assert await _read(db) == {GROUPS[0]: [], GROUPS[1]: [TEAM]}
