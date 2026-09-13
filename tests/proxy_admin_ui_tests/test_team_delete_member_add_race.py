"""
Real-Postgres coverage for the /team/member_add vs /team/delete race (LIT-5544), and for
/team/member_delete's participation in the same lock.

A member_add that validated the team before a delete began could previously still commit
its writes after the delete's reference sweeps had already run, leaving a user record and
a membership row pointing at a team id that no longer exists. Neither side of that race can
be forced by a sequential script: it needs one request to be genuinely mid-flight while the
other commits. A mocked prisma cannot arbitrate that either, since the property under test
is whether Postgres's own advisory lock actually serializes the two requests.

These tests pin the interleaving without a timing assumption: a second real connection holds
the team's advisory lock in its own transaction, and the test then waits for Postgres itself
to report the endpoint queued behind that exact lock. A sleep can only guess whether the
endpoint has reached the lock yet; pg_locks answers it.
"""

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import timedelta
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from litellm.proxy._types import (
    DeleteTeamRequest,
    LitellmUserRoles,
    Member,
    TeamMemberAddRequest,
    UserAPIKeyAuth,
)
from litellm.caching.caching import DualCache
from litellm.proxy.utils import PrismaClient, ProxyLogging

_DELETE_SEEDED = 'DELETE FROM "LiteLLM_TeamMembership" WHERE team_id = $1'
_DELETE_USER = 'DELETE FROM "LiteLLM_UserTable" WHERE user_id = $1'
_DELETE_TEAM = 'DELETE FROM "LiteLLM_TeamTable" WHERE team_id = $1'
_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked"
_HELD_LOCK_KEY_SQL = (
    "SELECT classid::bigint AS classid, objid::bigint AS objid FROM pg_locks "
    "WHERE locktype = 'advisory' AND granted AND pid = pg_backend_pid()"
)
_LOCK_WAITER_SQL = (
    "SELECT count(*)::int AS waiters FROM pg_locks "
    "WHERE locktype = 'advisory' AND NOT granted "
    "AND classid::bigint = $1 AND objid::bigint = $2"
)
_LOCK_WAIT_TIMEOUT_SECONDS = 20.0
_LOCK_POLL_SECONDS = 0.01


async def _hold_team_lock(held, team_id: str) -> tuple[int, int]:
    """Take the team's advisory lock and return its pg_locks key.

    Reading the key back off our own backend avoids re-deriving hashtext()'s signed
    32-bit split here, and pins the watcher to this lock rather than to any advisory
    lock another xdist worker happens to hold on the same database."""
    await held.query_raw(_LOCK_SQL, team_id)
    rows = await held.query_raw(_HELD_LOCK_KEY_SQL)
    assert len(rows) == 1, f"expected exactly one advisory lock on the blocking connection, got {rows}"
    return rows[0]["classid"], rows[0]["objid"]


async def _await_lock_contention(watcher, lock_key: tuple[int, int], task, what: str) -> None:
    """Block until Postgres reports `task` queued behind the held lock.

    This is the assertion that the endpoint serializes on the team's advisory lock, and it
    is what a fixed sleep was standing in for: the endpoint is only provably waiting once a
    non-granted advisory lock on the same key exists. `watcher` must be a connection that is
    not itself blocked, so it can observe the queue."""
    classid, objid = lock_key
    deadline = time.monotonic() + _LOCK_WAIT_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if task.done():
            raise AssertionError(f"{what} returned without waiting on the team's advisory lock") from task.exception()
        if (await watcher.query_raw(_LOCK_WAITER_SQL, classid, objid))[0]["waiters"]:
            return
        await asyncio.sleep(_LOCK_POLL_SECONDS)
    raise AssertionError(f"{what} never queued on the team's advisory lock within {_LOCK_WAIT_TIMEOUT_SECONDS}s")


def _race_ids() -> tuple[str, str]:
    """Unique per test: xdist workers share one Postgres, so a shared id lets one worker's
    cleanup delete the team another worker is mid-race on."""
    suffix = uuid.uuid4().hex[:8]
    return f"lit5544-race-team-{suffix}", f"lit5544-race-user-{suffix}"


@asynccontextmanager
async def _clean_db(team_id: str, user_id: str):
    """Connects inside the running test's loop: an async fixture would be torn up on a
    different loop than the test body, which prisma's engine lock refuses outright."""
    from prisma import Prisma

    if not os.getenv("DATABASE_URL"):
        pytest.fail("DATABASE_URL is required; these tests must not silently skip")

    db = Prisma()
    await db.connect()
    try:
        await db.execute_raw(_DELETE_SEEDED, team_id)
        await db.execute_raw(_DELETE_USER, user_id)
        await db.execute_raw(_DELETE_TEAM, team_id)
        yield db
    finally:
        await db.execute_raw(_DELETE_SEEDED, team_id)
        await db.execute_raw(_DELETE_USER, user_id)
        await db.execute_raw(_DELETE_TEAM, team_id)
        await db.disconnect()


@asynccontextmanager
async def _real_prisma_client():
    """The full app-level PrismaClient, not the raw generated client: add_new_member reads
    and writes through PrismaClient.get_data/insert_data, which the raw client doesn't have."""
    proxy_logging_obj = ProxyLogging(user_api_key_cache=DualCache())
    client = PrismaClient(database_url=os.environ["DATABASE_URL"], proxy_logging_obj=proxy_logging_obj)
    await client.connect()
    try:
        yield client
    finally:
        await client.db.disconnect()


def _admin_auth():
    return UserAPIKeyAuth(user_id="lit5544-admin", api_key="sk-lit5544", user_role=LitellmUserRoles.PROXY_ADMIN.value)


@pytest.mark.asyncio
async def test_member_add_blocked_by_delete_writes_no_dangling_reference():
    """
    member_add re-reads the team under the advisory lock before writing anything. When a
    delete already holds that lock and then removes the row, member_add's re-read must see
    the row gone and raise, without ever calling the write that appends the user/membership
    references, which is the only way this leaves zero trace after the delete wins.
    """
    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.proxy.management_endpoints.team_endpoints import (
        _add_team_members_to_team,
    )

    team_id, user_id = _race_ids()
    async with _clean_db(team_id, user_id) as db:
        await db.litellm_teamtable.create(data={"team_id": team_id, "team_alias": team_id, "members_with_roles": "[]"})

        async with _real_prisma_client() as prisma_client:
            from prisma import Prisma

            blocker = Prisma()
            await blocker.connect()

            async def add_member():
                await _add_team_members_to_team(
                    data=TeamMemberAddRequest(
                        team_id=team_id,
                        member=Member(user_id=user_id, role="user"),
                        max_budget_in_team=5.0,
                    ),
                    complete_team_data=LiteLLM_TeamTable(team_id=team_id, members_with_roles=[]),
                    prisma_client=prisma_client,
                    user_api_key_dict=_admin_auth(),
                    litellm_proxy_admin_name="lit5544-admin",
                )

            try:
                async with blocker.tx(timeout=timedelta(seconds=30)) as held:
                    lock_key = await _hold_team_lock(held, team_id)
                    task = asyncio.create_task(add_member())
                    await _await_lock_contention(db, lock_key, task, "member_add")

                    # the delete wins the race: strip the team row while the lock is held
                    await held.execute_raw(_DELETE_TEAM, team_id)

                with pytest.raises(HTTPException) as exc_info:
                    await asyncio.wait_for(task, timeout=30)
                assert exc_info.value.status_code == 404
            finally:
                await blocker.disconnect()

        user_row = await db.litellm_usertable.find_unique(where={"user_id": user_id})
        assert user_row is None, "member_add must not have written a user row for a team that was gone under its lock"

        membership_row = await db.litellm_teammembership.find_first(where={"team_id": team_id, "user_id": user_id})
        assert membership_row is None


@pytest.mark.asyncio
async def test_member_delete_blocked_by_member_add_removes_from_the_fresh_roster():
    """
    team_member_delete takes the same advisory lock and re-reads the roster under it, so a
    member_add that committed while member_delete was waiting on the lock is not silently
    undone. Without the re-read, member_delete would compute its new roster from the stale
    snapshot it validated against before the lock, and its write would overwrite the
    member_add's addition right back out even though member_add's request already succeeded.
    """
    import litellm.proxy.proxy_server as proxy_server_module
    from litellm.proxy._types import TeamMemberDeleteRequest
    from litellm.proxy.management_endpoints.team_endpoints import team_member_delete

    team_id, user_id = _race_ids()
    other_user = f"{user_id}-other"
    seeded_roster = '[{"user_id": "%s", "user_email": null, "role": "user"}]' % user_id
    winning_add_roster = (
        '[{"user_id": "%s", "user_email": null, "role": "user"}, '
        '{"user_id": "%s", "user_email": null, "role": "user"}]' % (user_id, other_user)
    )

    async with _clean_db(team_id, user_id) as db:
        await db.litellm_teamtable.create(
            data={"team_id": team_id, "team_alias": team_id, "members_with_roles": seeded_roster}
        )

        async with _real_prisma_client() as prisma_client:
            original_prisma_client = proxy_server_module.prisma_client
            proxy_server_module.prisma_client = prisma_client

            try:
                from prisma import Prisma

                blocker = Prisma()
                await blocker.connect()

                async def run_delete():
                    return await team_member_delete(
                        data=TeamMemberDeleteRequest(team_id=team_id, user_id=user_id),
                        user_api_key_dict=_admin_auth(),
                    )

                try:
                    async with blocker.tx(timeout=timedelta(seconds=30)) as held:
                        lock_key = await _hold_team_lock(held, team_id)
                        task = asyncio.create_task(run_delete())
                        await _await_lock_contention(db, lock_key, task, "member_delete")

                        # member_add wins the race: it adds `other_user` while holding the lock
                        await held.litellm_teamtable.update(
                            where={"team_id": team_id},
                            data={"members_with_roles": winning_add_roster},
                        )

                    await asyncio.wait_for(task, timeout=30)
                finally:
                    await blocker.disconnect()
            finally:
                proxy_server_module.prisma_client = original_prisma_client

        team_row = await db.litellm_teamtable.find_unique(where={"team_id": team_id})
        raw_roster = team_row.members_with_roles
        parsed_roster = json.loads(raw_roster) if isinstance(raw_roster, str) else raw_roster
        remaining_ids = {m["user_id"] for m in parsed_roster}
        assert remaining_ids == {other_user}, (
            "member_delete must remove only the user it targeted from the roster it actually "
            "committed to, not silently drop the member the winning add just committed"
        )


@pytest.mark.asyncio
async def test_delete_blocked_by_member_add_sweeps_the_fresh_reference():
    """
    A member_add that wins the lock race writes its reference and releases the lock; the
    delete that was waiting on it must then run its locked sweep against the row as it
    actually is, not a stale snapshot, and reap that reference rather than leaving it
    stranded on a team id the delete is about to remove.
    """
    import litellm.proxy.proxy_server as proxy_server_module
    from litellm.proxy._types import LiteLLM_TeamTable
    from litellm.proxy.management_endpoints.team_endpoints import delete_team

    team_id, user_id = _race_ids()
    async with _clean_db(team_id, user_id) as db:
        await db.litellm_teamtable.create(data={"team_id": team_id, "team_alias": team_id, "members_with_roles": "[]"})

        async with _real_prisma_client() as prisma_client:
            proxy_logging_obj = prisma_client.proxy_logging_obj
            original_prisma_client = proxy_server_module.prisma_client
            original_admin_name = proxy_server_module.litellm_proxy_admin_name
            original_proxy_logging_obj = proxy_server_module.proxy_logging_obj
            original_cache = proxy_server_module.user_api_key_cache
            original_router = proxy_server_module.llm_router
            proxy_server_module.prisma_client = prisma_client
            proxy_server_module.litellm_proxy_admin_name = "lit5544-admin"
            proxy_server_module.proxy_logging_obj = proxy_logging_obj
            proxy_server_module.user_api_key_cache = original_cache or proxy_logging_obj.internal_usage_cache
            proxy_server_module.llm_router = None

            async def restore():
                proxy_server_module.prisma_client = original_prisma_client
                proxy_server_module.litellm_proxy_admin_name = original_admin_name
                proxy_server_module.proxy_logging_obj = original_proxy_logging_obj
                proxy_server_module.user_api_key_cache = original_cache
                proxy_server_module.llm_router = original_router

            try:
                from prisma import Prisma

                blocker = Prisma()
                await blocker.connect()

                async def run_delete():
                    return await delete_team(
                        data=DeleteTeamRequest(team_ids=[team_id]),
                        http_request=MagicMock(),
                        user_api_key_dict=_admin_auth(),
                        litellm_changed_by="lit5544-admin",
                    )

                try:
                    async with blocker.tx(timeout=timedelta(seconds=30)) as held:
                        lock_key = await _hold_team_lock(held, team_id)
                        task = asyncio.create_task(run_delete())
                        await _await_lock_contention(db, lock_key, task, "delete_team")

                        # member_add wins the race: write the reference while holding the lock
                        await held.litellm_usertable.upsert(
                            where={"user_id": user_id},
                            data={
                                "create": {"user_id": user_id, "teams": [team_id]},
                                "update": {"teams": {"push": [team_id]}},
                            },
                        )
                        await held.litellm_teammembership.create(data={"team_id": team_id, "user_id": user_id})
                        await held.litellm_teamtable.update(
                            where={"team_id": team_id},
                            data={"members_with_roles": '[{"user_id": "%s", "role": "user"}]' % user_id},
                        )

                    await asyncio.wait_for(task, timeout=30)
                finally:
                    await blocker.disconnect()
            finally:
                await restore()

        team_row = await db.litellm_teamtable.find_unique(where={"team_id": team_id})
        assert team_row is None

        user_row = await db.litellm_usertable.find_unique(where={"user_id": user_id})
        assert user_row is not None and team_id not in user_row.teams, (
            "delete_team's locked sweep must reap the reference member_add wrote just before losing the lock"
        )

        membership_row = await db.litellm_teammembership.find_first(where={"team_id": team_id, "user_id": user_id})
        assert membership_row is None
