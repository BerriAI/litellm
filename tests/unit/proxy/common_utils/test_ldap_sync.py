import asyncio

from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.common_utils.ldap_directory import DirectoryError, DirectoryGroup
from litellm.proxy.common_utils.ldap_sync import LDAPSync, resolve_user_ids
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from tests.unit.proxy.common_utils.test_ldap_directory import ALICE, Connection, group, user
from tests.unit.proxy.management_helpers.test_team_roster_sync import (
    TEAM,
    _PausedDeleteRedisCache,  # pyright: ignore[reportPrivateUsage]  # event-controlled Redis boundary
    _FakePrisma,  # pyright: ignore[reportPrivateUsage]  # reuse the transactional database fake
    _member,  # pyright: ignore[reportPrivateUsage]  # shared roster fixtures
    _team,  # pyright: ignore[reportPrivateUsage]  # shared roster fixtures
    _user,  # pyright: ignore[reportPrivateUsage]  # shared roster fixtures
)
from litellm.repositories.user_repository import UserRepository
from litellm.types.proxy.ldap_sync import LDAPSyncSettings

SETTINGS: Final = LDAPSyncSettings(
    enabled=True,
    server_url="ldaps://directory.example.org",
    group_search_base="ou=groups,dc=example,dc=org",
    group_filter="(cn=project-*)",
    user_search_base="ou=people,dc=example,dc=org",
)


class Users:
    def __init__(self, rows: Sequence[LiteLLM_UserTable]) -> None:
        self.rows = rows
        self.db = self
        self.writer_db = self
        self.litellm_usertable = self

    async def find_many(
        self,
        *,
        where: Mapping[str, object] | None = None,
        take: int | None = None,
        skip: int | None = None,
        order: Mapping[str, str] | None = None,
    ) -> Sequence[LiteLLM_UserTable]:
        return self.rows


@pytest.mark.asyncio
async def test_existing_users_match_case_insensitively_without_creating_unknown_users():
    users = Users([LiteLLM_UserTable(user_id="alice", user_email="Alice@example.org")])
    result = await resolve_user_ids(
        UserRepository(users),
        DirectoryGroup("project-a", "project-a", ("alice@example.org", "unknown@example.org")),
        SETTINGS,
    )
    assert result == frozenset({"alice"})
    assert len(users.rows) == 1


@pytest.mark.asyncio
async def test_duplicate_email_accounts_abort_group_matching():
    users = Users(
        [
            LiteLLM_UserTable(user_id="alice", user_email="Alice@example.org"),
            LiteLLM_UserTable(user_id="other", user_email="alice@example.org"),
        ]
    )
    result = await resolve_user_ids(
        UserRepository(users), DirectoryGroup("project-a", "project-a", ("alice@example.org",)), SETTINGS
    )
    assert isinstance(result, DirectoryError)


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ["user_id", "sso_user_id"])
async def test_configured_identity_field_matches_existing_user(field: str):
    users = Users([LiteLLM_UserTable(user_id="alice", sso_user_id="z1234")])
    identity = "alice" if field == "user_id" else "z1234"
    result = await resolve_user_ids(
        UserRepository(users),
        DirectoryGroup("project-a", "project-a", (identity,)),
        SETTINGS.model_copy(update={"litellm_match_field": field}),
    )
    assert result == frozenset({"alice"})


class LeaseRedis:
    def __init__(self) -> None:
        self.values: dict[str, object] = {}  # mutable-ok: fake Redis state

    async def async_set_cache(self, key: str, value: object, *, nx: bool, ttl: int) -> bool:
        assert nx and ttl > 0
        if key in self.values:
            return False
        self.values[key] = value
        return True

    async def async_get_cache(self, key: str) -> object | None:
        return self.values.get(key)


def job_for(
    prisma: _FakePrisma,
    connection: Connection,
    settings: LDAPSyncSettings = SETTINGS,
    redis: LeaseRedis | None = None,
    cache: UserApiKeyCache | None = None,
) -> LDAPSync:
    lock: Final = PodLockManager(redis or LeaseRedis())  # pyright: ignore[reportArgumentType]  # in-process Redis boundary
    return LDAPSync(
        settings,
        prisma,  # pyright: ignore[reportArgumentType]  # transactional in-process Prisma boundary
        lock,
        cache or UserApiKeyCache(),
        None,
        directory_connection=connection,
    )


@pytest.mark.asyncio
async def test_job_updates_existing_roster_and_revokes_only_removed_members_team_keys():
    prisma: Final = _FakePrisma(
        users=[_user("alice"), _user("bob", TEAM)],
        teams=[_team(_member("bob"))],
        memberships=[(TEAM, "bob")],
        tokens=[{"token": "bob-team-key", "user_id": "bob", "team_id": TEAM}],
    )
    connection: Final = Connection([group(TEAM, member=[ALICE])], {ALICE: [user("alice")]})
    await job_for(prisma, connection, SETTINGS.model_copy(update={"litellm_match_field": "user_id"})).run()
    assert [member.user_id for member in prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles] == ["alice"]
    assert prisma.writer_db.litellm_usertable.rows["bob"].teams == []
    assert prisma.writer_db.litellm_usertable.rows["alice"].teams == [TEAM]
    assert prisma.writer_db.litellm_verificationtoken.rows == []
    assert set(prisma.writer_db.litellm_usertable.rows) == {"alice", "bob"}
    assert connection.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("team_exists", [False, True])
async def test_dry_run_leaves_users_teams_memberships_and_keys_untouched(team_exists: bool):
    prisma: Final = _FakePrisma(
        users=[_user("bob", TEAM)],
        teams=[_team(_member("bob"))] if team_exists else [],
        memberships=[(TEAM, "bob")],
        tokens=[{"token": "bob-team-key", "user_id": "bob", "team_id": TEAM}],
    )
    connection: Final = Connection([group(TEAM, member=[])])
    await job_for(prisma, connection, SETTINGS.model_copy(update={"dry_run": True})).run()
    assert prisma.transactions == 0
    assert set(prisma.writer_db.litellm_teamtable.rows) == ({TEAM} if team_exists else set())
    assert prisma.writer_db.litellm_usertable.rows["bob"].teams == [TEAM]
    assert prisma.writer_db.litellm_teammembership.rows == [{"team_id": TEAM, "user_id": "bob", "budget_id": None}]
    assert prisma.writer_db.litellm_verificationtoken.rows == [
        {"token": "bob-team-key", "user_id": "bob", "team_id": TEAM}
    ]


@pytest.mark.asyncio
async def test_second_worker_cannot_sync_while_first_workers_lease_is_held():
    prisma: Final = _FakePrisma(teams=[_team()])
    redis: Final = LeaseRedis()
    first: Final = Connection([group(TEAM, member=[])])
    second: Final = Connection([group(TEAM, member=[])])
    await job_for(prisma, first, redis=redis).run()
    transactions: Final = prisma.transactions
    await job_for(prisma, second, redis=redis).run()
    assert first.closed
    assert not second.closed
    assert prisma.transactions == transactions


@pytest.mark.asyncio
async def test_bad_group_does_not_prevent_later_healthy_group_from_syncing():
    prisma: Final = _FakePrisma(
        users=[_user("bob", TEAM)],
        teams=[_team(_member("bob"))],
        memberships=[(TEAM, "bob")],
    )
    connection: Final = Connection([group("broken", member=[ALICE]), group(TEAM, member=[])])
    await job_for(prisma, connection).run()
    assert set(prisma.writer_db.litellm_teamtable.rows) == {TEAM}
    assert prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles == []
    assert prisma.writer_db.litellm_teammembership.rows == []
    assert prisma.writer_db.litellm_usertable.rows["bob"].teams == []


@pytest.mark.asyncio
async def test_failed_snapshot_preserves_every_team_roster():
    prisma: Final = _FakePrisma(users=[_user("bob", TEAM)], teams=[_team(_member("bob"))])
    connection: Final = Connection([group(TEAM, member=[])], result_code=50)
    await job_for(prisma, connection).run()
    assert prisma.transactions == 0
    assert [member.user_id for member in prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles] == ["bob"]
    assert connection.closed


@pytest.mark.asyncio
async def test_missing_team_is_created_with_defaults_and_only_existing_users(monkeypatch: pytest.MonkeyPatch):
    import litellm
    import litellm.proxy.proxy_server as proxy_server

    prisma: Final = _FakePrisma(users=[_user("alice")])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    monkeypatch.setattr(litellm, "default_team_params", {"models": ["test-model"], "max_budget": 42})
    connection: Final = Connection([group(TEAM, member=[ALICE])], {ALICE: [user("alice")]})
    await job_for(prisma, connection, SETTINGS.model_copy(update={"litellm_match_field": "user_id"})).run()
    assert set(prisma.writer_db.litellm_teamtable.rows) == {TEAM}
    created: Final = prisma.writer_db.litellm_teamtable.rows[TEAM]
    assert created.models == ["test-model"]
    assert created.max_budget == 42
    assert [member.user_id for member in created.members_with_roles] == ["alice"]
    assert set(prisma.writer_db.litellm_usertable.rows) == {"alice"}
    assert prisma.writer_db.litellm_usertable.rows["alice"].teams == [TEAM]


@pytest.mark.asyncio
async def test_failed_team_creation_does_not_block_later_groups(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.proxy_server as proxy_server
    from litellm.proxy._types import UI_TEAM_ID

    prisma: Final = _FakePrisma(
        users=[_user("bob", TEAM)],
        teams=[_team(_member("bob"))],
        memberships=[(TEAM, "bob")],
    )
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    connection: Final = Connection([group(UI_TEAM_ID, member=[]), group(TEAM, member=[])])
    await job_for(prisma, connection).run()
    assert set(prisma.writer_db.litellm_teamtable.rows) == {TEAM}
    assert prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles == []
    assert prisma.writer_db.litellm_usertable.rows["bob"].teams == []
    assert prisma.writer_db.litellm_teammembership.rows == []


@pytest.mark.asyncio
async def test_cancelled_job_finishes_committed_revocation_and_stops_before_next_group(monkeypatch: pytest.MonkeyPatch):
    import litellm.proxy.common_utils.auth_cache_invalidation_pubsub as pubsub_module

    monkeypatch.setattr(pubsub_module, "_BULK_EVICTION_WAIT_SECONDS", None)
    prisma: Final = _FakePrisma(
        users=[_user("bob", TEAM)],
        teams=[_team(_member("bob"))],
        memberships=[(TEAM, "bob")],
        tokens=[{"token": "bob-key", "user_id": "bob", "team_id": TEAM}],
    )
    redis: Final = _PausedDeleteRedisCache({"bob-key": {"token": "bob-key"}})
    cache: Final = UserApiKeyCache()
    cache.attach_redis_cache(redis)  # pyright: ignore[reportArgumentType]  # dict-backed Redis boundary
    connection: Final = Connection([group(TEAM, member=[]), group("later", member=[])])
    job: Final = asyncio.create_task(job_for(prisma, connection, cache=cache).run())
    await redis.reached.wait()
    assert prisma.writer_db.litellm_verificationtoken.rows == []
    _ = job.cancel()
    asyncio.get_running_loop().call_soon(redis.release.set)
    with pytest.raises(asyncio.CancelledError):
        await job
    assert "bob-key" not in redis.entries
    assert set(prisma.writer_db.litellm_teamtable.rows) == {TEAM}
    assert prisma.transactions == 1
