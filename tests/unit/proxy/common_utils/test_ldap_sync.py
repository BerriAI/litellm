import asyncio

from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.common_utils.ldap_directory import DirectoryError, DirectoryGroup
from litellm.proxy.common_utils.ldap_sync import (
    LDAP_SYNC_JOB_ID,
    LDAPSync,
    initialize_ldap_sync,
    ldap_sync_settings,
    resolve_user_ids,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.db.db_transaction_queue.pod_lock_manager import PodLockManager
from tests.unit.proxy.common_utils.test_ldap_directory import ALICE, Connection, entry, group, user
from tests.unit.proxy.management_helpers.test_team_roster_sync import (
    TEAM,
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

    async def async_delete_cache(self, key: str) -> int:
        return 1 if self.values.pop(key, None) is not None else 0


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
async def test_second_worker_cannot_sync_while_first_workers_run_holds_the_lock(monkeypatch: pytest.MonkeyPatch):
    prisma: Final = _FakePrisma(teams=[_team()])
    redis: Final = LeaseRedis()
    first: Final = job_for(prisma, Connection([group(TEAM, member=[])]), redis=redis)
    second_connection: Final = Connection([group(TEAM, member=[])])
    entered: Final = asyncio.Event()
    release: Final = asyncio.Event()
    original: Final = first.sync_group

    async def held_sync_group(group: DirectoryGroup) -> None:
        entered.set()
        await release.wait()
        await original(group)

    monkeypatch.setattr(first, "sync_group", held_sync_group)
    running: Final = asyncio.create_task(first.run())
    await entered.wait()
    await job_for(prisma, second_connection, redis=redis).run()
    release.set()
    await running
    assert not second_connection.closed
    assert redis.values == {}


@pytest.mark.asyncio
async def test_finished_run_releases_the_lock_so_a_restarted_worker_syncs_immediately():
    prisma: Final = _FakePrisma(teams=[_team()])
    redis: Final = LeaseRedis()
    first: Final = Connection([group(TEAM, member=[])])
    restarted: Final = Connection([group(TEAM, member=[])])
    await job_for(prisma, first, redis=redis).run()
    assert redis.values == {}
    await job_for(prisma, restarted, redis=redis).run()
    assert first.closed
    assert restarted.closed


@pytest.mark.asyncio
async def test_failed_run_still_releases_the_lock(monkeypatch: pytest.MonkeyPatch):
    redis: Final = LeaseRedis()
    job: Final = job_for(_FakePrisma(teams=[_team()]), Connection([group(TEAM, member=[])]), redis=redis)

    async def failing_sync() -> None:
        raise RuntimeError("database unavailable")

    monkeypatch.setattr(job, "_sync", failing_sync)
    with pytest.raises(RuntimeError):
        await job.run()
    assert redis.values == {}


@pytest.mark.asyncio
async def test_bad_group_does_not_prevent_later_healthy_group_from_syncing():
    prisma: Final = _FakePrisma(
        users=[_user("bob", TEAM)],
        teams=[_team(_member("bob"))],
        memberships=[(TEAM, "bob")],
    )
    connection: Final = Connection([group("broken", member=[ALICE]), group(TEAM, member=[])])
    await job_for(prisma, connection, SETTINGS.model_copy(update={"allow_empty_groups": True})).run()
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
    await job_for(prisma, connection, SETTINGS.model_copy(update={"allow_empty_groups": True})).run()
    assert set(prisma.writer_db.litellm_teamtable.rows) == {TEAM}
    assert prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles == []
    assert prisma.writer_db.litellm_usertable.rows["bob"].teams == []
    assert prisma.writer_db.litellm_teammembership.rows == []


@pytest.mark.asyncio
async def test_time_budget_stops_between_teams_without_interrupting_a_write(monkeypatch: pytest.MonkeyPatch):
    prisma: Final = _FakePrisma(users=[_user("alice")], teams=[_team()])
    connection: Final = Connection(
        [group(TEAM, member=[ALICE]), group("later", member=[ALICE])], {ALICE: [user("alice")]}
    )
    job: Final = job_for(prisma, connection, SETTINGS.model_copy(update={"litellm_match_field": "user_id"}))
    job.time_budget_seconds = 0.5
    original: Final = job.sync_group
    started: list[str] = []  # mutable-ok: records which teams began syncing

    async def slow_sync_group(group: DirectoryGroup) -> None:
        started.append(group.team_id)
        await asyncio.sleep(1)
        await original(group)

    monkeypatch.setattr(job, "sync_group", slow_sync_group)
    await job.run()
    assert started == [TEAM]
    assert [member.user_id for member in prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles] == ["alice"]
    assert set(prisma.writer_db.litellm_teamtable.rows) == {TEAM}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("group_attributes", "settings_update"),
    [
        ({}, {}),
        ({"member": []}, {}),
        ({"member": [ALICE]}, {"litellm_match_field": "sso_user_id"}),
    ],
    ids=["member_attribute_hidden", "empty_member_list", "no_litellm_user_matches"],
)
async def test_group_resolving_to_no_users_keeps_a_populated_team_by_default(
    group_attributes: Mapping[str, object], settings_update: Mapping[str, object]
):
    prisma: Final = _FakePrisma(
        users=[_user("bob", TEAM)],
        teams=[_team(_member("bob"))],
        memberships=[(TEAM, "bob")],
        tokens=[{"token": "bob-team-key", "user_id": "bob", "team_id": TEAM}],
    )
    connection: Final = Connection([group(TEAM, **group_attributes)], {ALICE: [user("alice")]})
    await job_for(prisma, connection, SETTINGS.model_copy(update=dict(settings_update))).run()
    assert prisma.transactions == 0
    assert [member.user_id for member in prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles] == ["bob"]
    assert prisma.writer_db.litellm_verificationtoken.rows == [
        {"token": "bob-team-key", "user_id": "bob", "team_id": TEAM}
    ]


@pytest.mark.asyncio
async def test_allow_empty_groups_removes_every_member_of_an_emptied_group():
    prisma: Final = _FakePrisma(
        users=[_user("bob", TEAM)],
        teams=[_team(_member("bob"))],
        memberships=[(TEAM, "bob")],
        tokens=[{"token": "bob-team-key", "user_id": "bob", "team_id": TEAM}],
    )
    connection: Final = Connection([group(TEAM, member=[])])
    await job_for(prisma, connection, SETTINGS.model_copy(update={"allow_empty_groups": True})).run()
    assert prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles == []
    assert prisma.writer_db.litellm_verificationtoken.rows == []


@pytest.mark.asyncio
@pytest.mark.parametrize("object_class", ["groupOfMembers", "groupOfURLs", "groupOfEntries"])
async def test_nested_group_of_any_class_keeps_the_team_and_its_keys(object_class: str):
    nested: Final = "cn=eng,ou=groups,dc=example,dc=org"
    prisma: Final = _FakePrisma(
        users=[_user("alice", TEAM), _user("bob", TEAM)],
        teams=[_team(_member("alice"), _member("bob"))],
        memberships=[(TEAM, "alice"), (TEAM, "bob")],
        tokens=[{"token": "bob-team-key", "user_id": "bob", "team_id": TEAM}],
    )
    connection: Final = Connection(
        [group(TEAM, member=[ALICE, nested])],
        {ALICE: [user("alice")], nested: [entry(nested, mail=["eng"], objectClass=["top", object_class])]},
    )
    await job_for(prisma, connection, SETTINGS.model_copy(update={"litellm_match_field": "user_id"})).run()
    assert prisma.transactions == 0
    assert [member.user_id for member in prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles] == [
        "alice",
        "bob",
    ]
    assert prisma.writer_db.litellm_verificationtoken.rows == [
        {"token": "bob-team-key", "user_id": "bob", "team_id": TEAM}
    ]


@pytest.mark.parametrize(
    "general_settings",
    [{}, {"ldap_sync": {"enabled": False}}],
    ids=["absent", "disabled"],
)
def test_absent_or_disabled_ldap_sync_has_no_settings(general_settings: Mapping[str, object]):
    assert ldap_sync_settings(general_settings) is None


def test_enabled_ldap_sync_returns_validated_settings():
    assert ldap_sync_settings({"ldap_sync": SETTINGS.model_dump()}) == SETTINGS


class RecordingScheduler:
    def __init__(self) -> None:
        self.jobs: list[Mapping[str, object]] = []  # mutable-ok: records registered jobs

    def add_job(self, func: object, trigger: str, **options: object) -> object:
        self.jobs.append({"trigger": trigger, **options})
        return object()


def test_initialize_registers_one_interval_job():
    scheduler: Final = RecordingScheduler()
    initialize_ldap_sync(
        scheduler,
        SETTINGS,
        _FakePrisma(),  # pyright: ignore[reportArgumentType]  # transactional in-process Prisma boundary
        PodLockManager(LeaseRedis()),  # pyright: ignore[reportArgumentType]  # in-process Redis boundary
        UserApiKeyCache(),
        None,  # pyright: ignore[reportArgumentType]  # proxy logging is unused at registration
    )
    assert [(job["trigger"], job["id"], job["minutes"]) for job in scheduler.jobs] == [
        ("interval", LDAP_SYNC_JOB_ID, SETTINGS.interval_minutes)
    ]


def test_initialize_requires_shared_redis():
    with pytest.raises(ValueError, match="shared Redis"):
        initialize_ldap_sync(
            RecordingScheduler(),
            SETTINGS,
            _FakePrisma(),  # pyright: ignore[reportArgumentType]  # transactional in-process Prisma boundary
            PodLockManager(None),
            UserApiKeyCache(),
            None,  # pyright: ignore[reportArgumentType]  # proxy logging is unused at registration
        )


@pytest.mark.asyncio
async def test_configured_ldap_and_litellm_identity_fields_create_and_fill_the_team(monkeypatch: pytest.MonkeyPatch):
    from litellm.proxy import proxy_server
    from tests.unit.proxy.management_helpers.test_team_roster_sync import (
        _UserRow,  # pyright: ignore[reportPrivateUsage]  # fake user row with extra columns
    )

    alice: Final = _UserRow.model_validate(
        {"user_id": "alice", "user_email": "alice@example.com", "sso_user_id": "z-alice"}
    )
    prisma: Final = _FakePrisma(users=[alice, _user("bob")])
    monkeypatch.setattr(proxy_server, "prisma_client", prisma)
    settings: Final = SETTINGS.model_copy(
        update={
            "group_id_attribute": "entryUUID",
            "group_name_attribute": "description",
            "user_match_attribute": "uid",
            "litellm_match_field": "sso_user_id",
        }
    )
    connection: Final = Connection(
        [group(member=[ALICE], entryUUID=[TEAM], description=["Project A"])],
        {ALICE: [user("alice@example.org", uid=["z-alice"])]},
    )
    await job_for(prisma, connection, settings).run()
    created: Final = prisma.writer_db.litellm_teamtable.rows[TEAM]
    assert created.team_alias == "Project A"
    assert [member.user_id for member in created.members_with_roles] == ["alice"]
    assert prisma.writer_db.litellm_usertable.rows["bob"].teams == []
