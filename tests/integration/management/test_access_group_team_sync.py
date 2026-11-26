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
from typing import Final
from uuid import uuid4

import psycopg
import pytest
from fastapi import HTTPException

from litellm.proxy import proxy_server
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.management_endpoints.access_group_endpoints import create_access_group, update_access_group
from litellm.proxy.management_helpers.access_group_key_sync import sync_key_access_group_membership
from litellm.proxy.management_helpers.access_group_model_sync import sync_access_groups_for_deleted_model
from litellm.proxy.management_helpers.access_group_team_sync import (
    reconcile_team_access_group_membership,
    sync_team_access_group_membership,
)
from litellm.redis.stores.auth_cache import AuthObject
from litellm.types.access_group import AccessGroupCreateRequest, AccessGroupUpdateRequest
from tests.integration._support.client import Gateway, Scenario, eventually, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.prisma_clients import connected_proxy_prisma

_XDIST_WORKER = os.environ.get("PYTEST_XDIST_WORKER", "master")
TEAM = f"ags-team-a-{_XDIST_WORKER}"
OTHER_TEAM = f"ags-team-b-{_XDIST_WORKER}"
GROUPS = tuple(f"ags-group-{n}-{_XDIST_WORKER}" for n in (1, 2, 3))
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


async def _sync(db, team_id, access_group_ids) -> frozenset[str]:
    from litellm.proxy.proxy_server import user_api_key_cache

    await _set_team_groups(db, team_id, access_group_ids)
    for group in GROUPS:
        user_api_key_cache.set_cache(key=AuthObject.access_group(group), value={"access_group_id": group})
    await sync_team_access_group_membership(prisma_client=SimpleNamespace(db=db), team_id=team_id)
    return frozenset(
        group for group in GROUPS if user_api_key_cache.get_cache(key=AuthObject.access_group(group)) is None
    )


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
            await sync_team_access_group_membership(prisma_client=SimpleNamespace(db=db), team_id=TEAM)

        try:
            async with blocker.tx(timeout=timedelta(seconds=30)) as held:
                await held.query_raw("SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked", TEAM)
                task = asyncio.create_task(competing_sync())
                await sync_started.wait()
                blocked: Final = await asyncio.to_thread(
                    eventually,
                    lambda: read_rows(
                        "SELECT count(*) AS blocked FROM pg_stat_activity "
                        "WHERE wait_event_type = 'Lock' AND query LIKE %s",
                        ("%pg_advisory_xact_lock%",),
                    ),
                    lambda rows: rows[0]["blocked"] == 1,
                    seconds=5,
                )
                assert blocked[0]["blocked"] == 1
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


@pytest.mark.asyncio
async def test_group_editors_assign_the_same_teams_in_opposite_request_order(monkeypatch: pytest.MonkeyPatch) -> None:
    actor: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    async with _clean_db() as db, connected_proxy_prisma() as client:
        monkeypatch.setattr(proxy_server, "prisma_client", client)
        for team in (TEAM, OTHER_TEAM):
            await _set_team_groups(db, team, [])
        await _seed(db, {GROUPS[0]: [], GROUPS[1]: []})

        async with asyncio.TaskGroup() as editors:
            async with db.tx(timeout=timedelta(seconds=30)) as held:
                (holder,) = await held.query_raw(
                    'SELECT pg_backend_pid() AS pid FROM "LiteLLM_TeamTable" WHERE team_id = $1 FOR UPDATE', TEAM
                )
                editors.create_task(
                    update_access_group(
                        GROUPS[0],
                        AccessGroupUpdateRequest(assigned_team_ids=[TEAM, OTHER_TEAM]),
                        actor,
                    )
                )
                editors.create_task(
                    update_access_group(
                        GROUPS[1],
                        AccessGroupUpdateRequest(assigned_team_ids=[OTHER_TEAM, TEAM]),
                        actor,
                    )
                )
                await asyncio.to_thread(
                    eventually,
                    lambda: read_rows(
                        "WITH RECURSIVE wait_chain(pid) AS (VALUES (%s::int) UNION "
                        "SELECT a.pid FROM wait_chain JOIN pg_stat_activity a "
                        "ON wait_chain.pid = ANY(pg_blocking_pids(a.pid))) "
                        "SELECT count(*) - 1 AS n FROM wait_chain",
                        (holder["pid"],),
                    ),
                    lambda rows: rows[0]["n"] == 2,
                    seconds=3,
                )

        members: Final = await db.litellm_teamtable.find_many(where={"team_id": {"in": [TEAM, OTHER_TEAM]}})
        assert {row.team_id: sorted(row.access_group_ids) for row in members} == {
            TEAM: sorted(GROUPS[:2]),
            OTHER_TEAM: sorted(GROUPS[:2]),
        }
        assert await _read(db) == {group: sorted((TEAM, OTHER_TEAM)) for group in GROUPS[:2]}


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["missing", "duplicate", "settled"])
async def test_busy_team_preserves_unrelated_group_responses(operation: str, monkeypatch: pytest.MonkeyPatch) -> None:
    actor: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    async with _clean_db() as db, connected_proxy_prisma() as client:
        monkeypatch.setattr(proxy_server, "prisma_client", client)
        await _set_team_groups(db, TEAM, [GROUPS[0]])
        await _seed(db, {GROUPS[0]: [TEAM]})
        async with db.tx(timeout=timedelta(seconds=30)) as held:
            await held.query_raw('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = $1 FOR UPDATE', TEAM)
            if operation == "settled":
                result: Final = await update_access_group(
                    GROUPS[0], AccessGroupUpdateRequest(assigned_team_ids=[TEAM], description="settled"), actor
                )
                assert (tuple(result.assigned_team_ids or ()), result.description) == ((TEAM,), "settled")
            elif operation == "missing":
                with pytest.raises(HTTPException) as error:
                    await update_access_group(GROUPS[1], AccessGroupUpdateRequest(assigned_team_ids=[TEAM]), actor)
                assert (error.value.status_code, error.value.detail) == (404, f"Access group '{GROUPS[1]}' not found")
            else:
                with pytest.raises(HTTPException) as error:
                    await create_access_group(
                        AccessGroupCreateRequest(access_group_name=GROUPS[0], assigned_team_ids=[TEAM]), actor
                    )
                assert (error.value.status_code, error.value.detail) == (
                    409,
                    f"Access group '{GROUPS[0]}' already exists",
                )
        assert await _read(db) == {GROUPS[0]: [TEAM]}


@pytest.mark.asyncio
async def test_settled_request_restores_assignment_removed_while_waiting_for_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor: Final = UserAPIKeyAuth(user_role=LitellmUserRoles.PROXY_ADMIN)
    async with _clean_db() as db, connected_proxy_prisma() as client:
        monkeypatch.setattr(proxy_server, "prisma_client", client)
        await _set_team_groups(db, TEAM, [GROUPS[0]])
        await _seed(db, {GROUPS[0]: [TEAM]})
        async with asyncio.TaskGroup() as editors:
            async with db.tx(timeout=timedelta(seconds=30)) as removing:
                await removing.litellm_teamtable.update(where={"team_id": TEAM}, data={"access_group_ids": []})
                await removing.litellm_accessgrouptable.update(
                    where={"access_group_id": GROUPS[0]}, data={"assigned_team_ids": []}
                )
                (backend,) = await removing.query_raw("SELECT pg_backend_pid() AS pid")
                editors.create_task(
                    update_access_group(GROUPS[0], AccessGroupUpdateRequest(assigned_team_ids=[TEAM]), actor)
                )
                await asyncio.to_thread(
                    eventually,
                    lambda: read_rows(
                        "SELECT count(*) AS n FROM pg_stat_activity WHERE %s = ANY(pg_blocking_pids(pid))",
                        (backend["pid"],),
                    ),
                    lambda rows: rows[0]["n"] == 1,
                    seconds=3,
                )
        team: Final = await db.litellm_teamtable.find_unique_or_raise(where={"team_id": TEAM})
        assert (team.access_group_ids, await _read(db)) == ([GROUPS[0]], {GROUPS[0]: [TEAM]})


def _store_first_in_scan_order(first: str, second: str, model: str) -> None:
    """Store two access groups so a sequential scan reads ``first`` before ``second``. Postgres puts a
    new row in any free slot of the page, so ``second`` is stored again until it lands after ``first``;
    the slots it leaves stay taken until the transaction commits, so every retry moves it forward."""
    insert: Final = (
        'INSERT INTO "LiteLLM_AccessGroupTable" (access_group_id, access_group_name, access_model_names) '
        "VALUES (%s, %s, ARRAY[%s]) RETURNING ctid"
    )
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        (first_slot,) = connection.execute(insert, (first, first, model)).fetchall()
        (second_slot,) = connection.execute(insert, (second, second, model)).fetchall()
        while _slot(second_slot[0]) < _slot(first_slot[0]):
            connection.execute('DELETE FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = %s', (second,))
            (second_slot,) = connection.execute(insert, (second, second, model)).fetchall()


def _slot(ctid: str) -> tuple[int, int]:
    page, _, offset = ctid.strip("()").partition(",")
    return int(page), int(offset)


@pytest.mark.asyncio
@pytest.mark.parametrize("writer", ["model delete", "key attach"])
async def test_group_writer_and_team_sync_on_the_same_groups_both_commit(writer: str) -> None:
    model: Final = f"ags-model-{_XDIST_WORKER}"
    key: Final = f"ags-key-{_XDIST_WORKER}"
    low, high = GROUPS[0], GROUPS[1]
    async with _clean_db() as db, connected_proxy_prisma() as client:
        await asyncio.to_thread(_store_first_in_scan_order, high, low, model)
        await _set_team_groups(db, TEAM, [low, high])

        with psycopg.connect(os.environ["DATABASE_URL"]) as held:
            (holder,) = held.execute(
                'SELECT pg_backend_pid() FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = %s FOR UPDATE', (high,)
            ).fetchall()
            group_writer: Final = asyncio.create_task(
                sync_access_groups_for_deleted_model(client, model_id=uuid4().hex, model_name=model, llm_router=None)
                if writer == "model delete"
                else sync_key_access_group_membership(client, key, None, [low, high])
            )
            await asyncio.to_thread(
                eventually,
                lambda: read_rows(
                    "SELECT count(*) AS n FROM pg_stat_activity WHERE %s = ANY(pg_blocking_pids(pid))", (holder[0],)
                ),
                lambda rows: rows[0]["n"] == 1,
                seconds=5,
            )
            team_sync: Final = asyncio.create_task(
                sync_team_access_group_membership(prisma_client=SimpleNamespace(db=db), team_id=TEAM)
            )
            await asyncio.to_thread(
                eventually,
                lambda: read_rows(
                    "WITH RECURSIVE wait_chain(pid) AS (VALUES (%s::int) UNION "
                    "SELECT a.pid FROM wait_chain JOIN pg_stat_activity a "
                    "ON wait_chain.pid = ANY(pg_blocking_pids(a.pid))) "
                    "SELECT count(*) - 1 AS n FROM wait_chain",
                    (holder[0],),
                ),
                lambda rows: rows[0]["n"] == 2,
                seconds=5,
            )
        await asyncio.wait_for(asyncio.gather(group_writer, team_sync), timeout=30)

        groups: Final = await db.query_raw(
            "SELECT access_group_id, access_model_names, assigned_key_ids, assigned_team_ids "
            'FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = ANY($1::text[]) ORDER BY access_group_id',
            [low, high],
        )
        models: Final = [] if writer == "model delete" else [model]
        keys: Final = [key] if writer == "key attach" else []
        assert groups == [
            {
                "access_group_id": group,
                "access_model_names": models,
                "assigned_key_ids": keys,
                "assigned_team_ids": [TEAM],
            }
            for group in (low, high)
        ]


def _assigned_group(scenario: Scenario, team_id: str) -> str:
    created: Final = scenario.gateway.request(
        "POST", "/v1/access_group", {"access_group_name": uuid4().hex, "assigned_team_ids": [team_id]}
    )
    assert created.status_code == 201, created.text
    group_id: Final = string_value(created.json()["access_group_id"])
    scenario.cleanups.callback(scenario.gateway.request, "DELETE", f"/v1/access_group/{group_id}")
    return group_id


def test_team_rpm_edit_succeeds_while_its_settled_group_is_busy(gateway: Gateway, scenario: Scenario) -> None:
    team: Final = scenario.team()
    group: Final = _assigned_group(scenario, team)
    with psycopg.connect(os.environ["DATABASE_URL"]) as held:
        held.execute('SELECT 1 FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = %s FOR UPDATE', (group,))
        response: Final = gateway.request("POST", "/team/update", {"team_id": team, "rpm_limit": 40})

    assert response.status_code == 200, response.text
    assert gateway.rows(
        'SELECT t.rpm_limit, t.access_group_ids, g.assigned_team_ids FROM "LiteLLM_TeamTable" t '
        'JOIN "LiteLLM_AccessGroupTable" g ON g.access_group_id = %s WHERE t.team_id = %s',
        (group, team),
    ) == [{"rpm_limit": 40, "access_group_ids": [group], "assigned_team_ids": [team]}]


def test_duplicate_group_rename_is_reported_while_a_removed_team_is_busy(gateway: Gateway, scenario: Scenario) -> None:
    team: Final = scenario.team()
    group: Final = _assigned_group(scenario, team)
    taken: Final = uuid4().hex
    owner: Final = gateway.request("POST", "/v1/access_group", {"access_group_name": taken})
    assert owner.status_code == 201, owner.text
    owner_id: Final = string_value(owner.json()["access_group_id"])
    scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/access_group/{owner_id}")

    with psycopg.connect(os.environ["DATABASE_URL"]) as held:
        held.execute('SELECT 1 FROM "LiteLLM_TeamTable" WHERE team_id = %s FOR UPDATE', (team,))
        response: Final = gateway.request(
            "PUT", f"/v1/access_group/{group}", {"access_group_name": taken, "assigned_team_ids": []}
        )

    assert response.status_code == 409, response.text
    assert response.json() == {"detail": f"Access group '{taken}' already exists"}
    assert gateway.rows(
        'SELECT assigned_team_ids FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = %s', (group,)
    ) == [{"assigned_team_ids": [team]}]


@pytest.mark.parametrize("method", ["POST", "PUT"])
def test_unknown_team_is_reported_while_a_valid_assignment_is_busy(
    gateway: Gateway, scenario: Scenario, method: str
) -> None:
    team: Final = scenario.team()
    group: Final = _assigned_group(scenario, team)
    missing: Final = uuid4().hex
    name: Final = uuid4().hex
    with psycopg.connect(os.environ["DATABASE_URL"]) as held:
        held.execute('SELECT 1 FROM "LiteLLM_TeamTable" WHERE team_id = %s FOR UPDATE', (team,))
        response: Final = (
            gateway.request(
                method, "/v1/access_group", {"access_group_name": name, "assigned_team_ids": [team, missing]}
            )
            if method == "POST"
            else gateway.request(method, f"/v1/access_group/{group}", {"assigned_team_ids": [missing]})
        )

    assert response.status_code == 400, response.text
    assert response.json() == {"detail": f"Unknown team ids: {missing}"}
    assert gateway.rows(
        'SELECT assigned_team_ids FROM "LiteLLM_AccessGroupTable" WHERE access_group_id = %s OR access_group_name = %s',
        (group, name),
    ) == [{"assigned_team_ids": [team]}]
