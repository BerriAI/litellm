"""
Real-Postgres coverage for the /team/member_add vs /team/delete race (LIT-5544), and for
/team/member_delete's participation in the same lock.

A member_add that validated the team before a delete began could previously still commit
its writes after the delete's reference sweeps had already run, leaving a user record and
a membership row pointing at a team id that no longer exists. Neither side of that race can
be forced by a sequential script: it needs one request to be genuinely mid-flight while the
other commits. A mocked prisma cannot arbitrate that either, since the property under test
is whether Postgres's own advisory lock actually serializes the two requests.

These tests pin the interleaving the same way test_access_group_team_sync.py does: a second
real connection holds the team's advisory lock in its own transaction, so the function under
test is provably blocked on it rather than hoping a sleep lands in the right gap.
"""

import asyncio
import json
import os
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

TEAM = "lit5544-race-team"
USER = "lit5544-race-user"
_DELETE_SEEDED = 'DELETE FROM "LiteLLM_TeamMembership" WHERE team_id = $1'
_DELETE_USER = 'DELETE FROM "LiteLLM_UserTable" WHERE user_id = $1'
_DELETE_TEAM = 'DELETE FROM "LiteLLM_TeamTable" WHERE team_id = $1'
_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext($1)) IS NULL AS locked"


@asynccontextmanager
async def _clean_db():
    """Connects inside the running test's loop: an async fixture would be torn up on a
    different loop than the test body, which prisma's engine lock refuses outright."""
    from prisma import Prisma

    if not os.getenv("DATABASE_URL"):
        pytest.fail("DATABASE_URL is required; these tests must not silently skip")

    db = Prisma()
    await db.connect()
    try:
        await db.execute_raw(_DELETE_SEEDED, TEAM)
        await db.execute_raw(_DELETE_USER, USER)
        await db.execute_raw(_DELETE_TEAM, TEAM)
        yield db
    finally:
        await db.execute_raw(_DELETE_SEEDED, TEAM)
        await db.execute_raw(_DELETE_USER, USER)
        await db.execute_raw(_DELETE_TEAM, TEAM)
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

    async with _clean_db() as db:
        await db.litellm_teamtable.create(data={"team_id": TEAM, "team_alias": TEAM, "members_with_roles": "[]"})

        async with _real_prisma_client() as prisma_client:
            from prisma import Prisma

            blocker = Prisma()
            await blocker.connect()
            lock_acquired = asyncio.Event()

            async def add_member():
                lock_acquired.set()
                await _add_team_members_to_team(
                    data=TeamMemberAddRequest(
                        team_id=TEAM,
                        member=Member(user_id=USER, role="user"),
                        max_budget_in_team=5.0,
                    ),
                    complete_team_data=LiteLLM_TeamTable(team_id=TEAM, members_with_roles=[]),
                    prisma_client=prisma_client,
                    user_api_key_dict=_admin_auth(),
                    litellm_proxy_admin_name="lit5544-admin",
                )

            try:
                async with blocker.tx(timeout=timedelta(seconds=30)) as held:
                    await held.query_raw(_LOCK_SQL, TEAM)
                    task = asyncio.create_task(add_member())
                    await lock_acquired.wait()
                    await asyncio.sleep(0.2)
                    assert not task.done(), "member_add did not wait on the team's advisory lock"

                    # the delete wins the race: strip the team row while the lock is held
                    await held.execute_raw(_DELETE_TEAM, TEAM)

                with pytest.raises(HTTPException) as exc_info:
                    await asyncio.wait_for(task, timeout=30)
                assert exc_info.value.status_code == 404
            finally:
                await blocker.disconnect()

        user_row = await db.litellm_usertable.find_unique(where={"user_id": USER})
        assert user_row is None, "member_add must not have written a user row for a team that was gone under its lock"

        membership_row = await db.litellm_teammembership.find_first(where={"team_id": TEAM, "user_id": USER})
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

    other_user = f"{USER}-other"
    seeded_roster = '[{"user_id": "%s", "user_email": null, "role": "user"}]' % USER
    winning_add_roster = (
        '[{"user_id": "%s", "user_email": null, "role": "user"}, '
        '{"user_id": "%s", "user_email": null, "role": "user"}]' % (USER, other_user)
    )

    async with _clean_db() as db:
        await db.litellm_teamtable.create(
            data={"team_id": TEAM, "team_alias": TEAM, "members_with_roles": seeded_roster}
        )

        async with _real_prisma_client() as prisma_client:
            original_prisma_client = proxy_server_module.prisma_client
            proxy_server_module.prisma_client = prisma_client

            try:
                from prisma import Prisma

                blocker = Prisma()
                await blocker.connect()
                lock_acquired = asyncio.Event()

                async def run_delete():
                    lock_acquired.set()
                    return await team_member_delete(
                        data=TeamMemberDeleteRequest(team_id=TEAM, user_id=USER),
                        user_api_key_dict=_admin_auth(),
                    )

                try:
                    async with blocker.tx(timeout=timedelta(seconds=30)) as held:
                        await held.query_raw(_LOCK_SQL, TEAM)
                        task = asyncio.create_task(run_delete())
                        await lock_acquired.wait()
                        await asyncio.sleep(0.2)
                        assert not task.done(), "member_delete did not wait on the team's advisory lock"

                        # member_add wins the race: it adds `other_user` while holding the lock
                        await held.litellm_teamtable.update(
                            where={"team_id": TEAM},
                            data={"members_with_roles": winning_add_roster},
                        )

                    await asyncio.wait_for(task, timeout=30)
                finally:
                    await blocker.disconnect()
            finally:
                proxy_server_module.prisma_client = original_prisma_client

        team_row = await db.litellm_teamtable.find_unique(where={"team_id": TEAM})
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

    async with _clean_db() as db:
        await db.litellm_teamtable.create(data={"team_id": TEAM, "team_alias": TEAM, "members_with_roles": "[]"})

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
                lock_acquired = asyncio.Event()

                async def run_delete():
                    lock_acquired.set()
                    return await delete_team(
                        data=DeleteTeamRequest(team_ids=[TEAM]),
                        http_request=MagicMock(),
                        user_api_key_dict=_admin_auth(),
                        litellm_changed_by="lit5544-admin",
                    )

                try:
                    async with blocker.tx(timeout=timedelta(seconds=30)) as held:
                        await held.query_raw(_LOCK_SQL, TEAM)
                        task = asyncio.create_task(run_delete())
                        await lock_acquired.wait()
                        await asyncio.sleep(0.3)
                        assert not task.done(), "delete_team did not wait on the team's advisory lock"

                        # member_add wins the race: write the reference while holding the lock
                        await held.litellm_usertable.upsert(
                            where={"user_id": USER},
                            data={
                                "create": {"user_id": USER, "teams": [TEAM]},
                                "update": {"teams": {"push": [TEAM]}},
                            },
                        )
                        await held.litellm_teammembership.create(data={"team_id": TEAM, "user_id": USER})
                        await held.litellm_teamtable.update(
                            where={"team_id": TEAM},
                            data={"members_with_roles": '[{"user_id": "%s", "role": "user"}]' % USER},
                        )

                    await asyncio.wait_for(task, timeout=30)
                finally:
                    await blocker.disconnect()
            finally:
                await restore()

        team_row = await db.litellm_teamtable.find_unique(where={"team_id": TEAM})
        assert team_row is None

        user_row = await db.litellm_usertable.find_unique(where={"user_id": USER})
        assert user_row is not None and TEAM not in user_row.teams, (
            "delete_team's locked sweep must reap the reference member_add wrote just before losing the lock"
        )

        membership_row = await db.litellm_teammembership.find_first(where={"team_id": TEAM, "user_id": USER})
        assert membership_row is None
