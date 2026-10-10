import asyncio
import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Final, Literal
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter

import litellm.proxy.common_utils.auth_cache_invalidation_pubsub as pubsub_module
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, Member, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import jwt_key_mapping_cache_key
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    team_membership_auth_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.management_helpers.team_roster_sync import (
    MembersMissing,
    RosterDelta,
    RosterPlan,
    RosterSync,
    RosterTarget,
    TeamGone,
    sync_team_roster,
)

ADMIN: Final = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin")
TEAM: Final = "t1"
READS: Final = ("query_raw", "teamtable.find_unique", "usertable.find_many", "budgettable.find_unique")
_STRINGS: Final = TypeAdapter(list[str])
_FILTER: Final = TypeAdapter(dict[str, object])
_FILTERS: Final = TypeAdapter(list[dict[str, object]])
_MEMBERS: Final = TypeAdapter(list[Member])


class _UserRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str
    user_email: str | None = None
    teams: list[str] = []


class _Record(BaseModel):
    """Attribute access like a Prisma row, over whatever columns the test seeded."""

    model_config = ConfigDict(extra="allow")


class _Ledger:
    """Every statement the fake ran, shared by reference so a transaction snapshot never forks it."""

    def __init__(self) -> None:
        self.statements: list[str] = []
        self.locked_users: list[list[str]] = []

    def __deepcopy__(self, memo: dict[int, object]) -> "_Ledger":
        return self


def _clause_matches(row: Mapping[str, object], field: str, clause: object) -> bool:
    if field == "OR":
        return any(_matches(row, inner) for inner in _FILTERS.validate_python(clause))
    if field == "AND":
        return all(_matches(row, inner) for inner in _FILTERS.validate_python(clause))
    if field == "NOT":
        return not _matches(row, _FILTER.validate_python(clause))
    if not isinstance(clause, dict):
        return row.get(field) == clause
    operator = _FILTER.validate_python(clause)
    if "in" in operator:
        return row.get(field) in set(_STRINGS.validate_python(operator["in"]))
    if "has" in operator:
        return operator["has"] in _STRINGS.validate_python(row.get(field) or [])
    raise AssertionError(f"unsupported filter {field}={clause!r}")


def _matches(row: Mapping[str, object], where: Mapping[str, object]) -> bool:
    return all(_clause_matches(row, field, clause) for field, clause in where.items())


class _Rows:
    """A list-backed Prisma table over the filters the roster sync issues."""

    def __init__(self, name: str, ledger: _Ledger, rows: Sequence[Mapping[str, object]] = ()) -> None:
        self.name = name
        self.ledger = ledger
        self.rows: list[dict[str, object]] = [dict(r) for r in rows]

    def _ran(self, method: str) -> None:
        self.ledger.statements.append(f"{self.name}.{method}")

    async def find_many(self, where: Mapping[str, object]) -> list[_Record]:
        self._ran("find_many")
        return [_Record.model_validate(r) for r in self.rows if _matches(r, where)]

    async def find_unique(self, where: Mapping[str, object]) -> _Record | None:
        self._ran("find_unique")
        return next((_Record.model_validate(r) for r in self.rows if _matches(r, where)), None)

    async def delete_many(self, where: Mapping[str, object]) -> int:
        self._ran("delete_many")
        before = len(self.rows)
        self.rows = [r for r in self.rows if not _matches(r, where)]
        return before - len(self.rows)

    async def create_many(self, data: Sequence[Mapping[str, object]], skip_duplicates: bool = False) -> int:
        self._ran("create_many")
        present = {(r.get("team_id"), r.get("user_id")) for r in self.rows}
        fresh = [dict(r) for r in data if not (skip_duplicates and (r.get("team_id"), r.get("user_id")) in present)]
        self.rows.extend(fresh)
        return len(fresh)


class _UserTable:
    def __init__(self, ledger: _Ledger, users: Sequence[_UserRow]) -> None:
        self.ledger = ledger
        self.rows: dict[str, _UserRow] = {u.user_id: u for u in users}

    async def find_many(self, where: Mapping[str, object]) -> list[_UserRow]:
        self.ledger.statements.append("usertable.find_many")
        return [u for u in self.rows.values() if _matches(u.model_dump(), where)]

    async def update_many(self, where: Mapping[str, object], data: Mapping[str, Mapping[str, Sequence[str]]]) -> int:
        self.ledger.statements.append("usertable.update_many")
        hit = [u for u in self.rows.values() if _matches(u.model_dump(), where)]
        for user in hit:
            self.rows[user.user_id] = user.model_copy(update={"teams": [*user.teams, *data["teams"]["push"]]})
        return len(hit)

    def detach(self, team_id: str, user_ids: Sequence[str]) -> None:
        for user in (self.rows[user_id] for user_id in user_ids if user_id in self.rows):
            self.rows[user.user_id] = user.model_copy(update={"teams": [t for t in user.teams if t != team_id]})


class _TeamTable:
    def __init__(self, ledger: _Ledger, teams: Sequence[LiteLLM_TeamTable]) -> None:
        self.ledger = ledger
        self.rows: dict[str, LiteLLM_TeamTable] = {t.team_id: t for t in teams}

    async def find_unique(self, where: Mapping[str, str]) -> LiteLLM_TeamTable | None:
        self.ledger.statements.append("teamtable.find_unique")
        return self.rows.get(where["team_id"])

    async def update(self, where: Mapping[str, str], data: Mapping[str, str]) -> LiteLLM_TeamTable:
        self.ledger.statements.append("teamtable.update")
        roster = _MEMBERS.validate_json(data["members_with_roles"])
        updated = self.rows[where["team_id"]].model_copy(
            update={"members_with_roles": roster, "updated_at": datetime.now(UTC)}
        )
        self.rows[where["team_id"]] = updated
        return updated


class _Db:
    def __init__(
        self,
        ledger: _Ledger,
        users: Sequence[_UserRow],
        teams: Sequence[LiteLLM_TeamTable],
        memberships: Sequence[tuple[str, str]],
        tokens: Sequence[Mapping[str, object]],
        budgets: Sequence[Mapping[str, object]],
        jwt_mappings: Sequence[Mapping[str, object]],
    ) -> None:
        self.litellm_usertable = _UserTable(ledger, users)
        self.litellm_teamtable = _TeamTable(ledger, teams)
        self.litellm_teammembership = _Rows(
            "teammembership", ledger, [{"team_id": t, "user_id": u, "budget_id": None} for t, u in memberships]
        )
        self.litellm_verificationtoken = _Rows("verificationtoken", ledger, tokens)
        self.litellm_deletedverificationtoken = _Rows("deletedverificationtoken", ledger)
        self.litellm_budgettable = _Rows("budgettable", ledger, budgets)
        self.litellm_jwtkeymapping = _Rows("jwtkeymapping", ledger, jwt_mappings)


class _Tx:
    def __init__(self, db: _Db, ledger: _Ledger) -> None:
        self.litellm_usertable = db.litellm_usertable
        self.litellm_teamtable = db.litellm_teamtable
        self.litellm_teammembership = db.litellm_teammembership
        self.litellm_verificationtoken = db.litellm_verificationtoken
        self.litellm_deletedverificationtoken = db.litellm_deletedverificationtoken
        self.litellm_budgettable = db.litellm_budgettable
        self.ledger = ledger

    async def query_raw(self, sql: str, *args: object) -> list[dict[str, object]]:
        self.ledger.statements.append("query_raw")
        if "pg_advisory_xact_lock" in sql:
            assert args == (TEAM,)
            return [{"locked": False}]
        assert "ORDER BY user_id FOR UPDATE" in sql
        user_ids = _STRINGS.validate_python(args[0])
        self.ledger.locked_users.append(user_ids)
        return [{"user_id": user_id} for user_id in user_ids if user_id in self.litellm_usertable.rows]

    async def execute_raw(self, sql: str, *args: object) -> int:
        self.ledger.statements.append("execute_raw")
        assert "array_remove(teams, $1)" in sql
        user_ids = _STRINGS.validate_python(args[1])
        self.litellm_usertable.detach(str(args[0]), user_ids)
        return len(user_ids)


class _FakePrisma:
    def __init__(
        self,
        users: Sequence[_UserRow] = (),
        teams: Sequence[LiteLLM_TeamTable] = (),
        memberships: Sequence[tuple[str, str]] = (),
        tokens: Sequence[Mapping[str, object]] = (),
        budgets: Sequence[Mapping[str, object]] = (),
        jwt_mappings: Sequence[Mapping[str, object]] = (),
        fail_commit: bool = False,
        replica_teams: Sequence[LiteLLM_TeamTable] | None = None,
    ) -> None:
        """``db`` is the routed client, which a read replica may serve; ``writer_db`` is the writer."""
        self.ledger = _Ledger()
        self.writer_db = _Db(self.ledger, users, teams, memberships, tokens, budgets, jwt_mappings)
        self.db = (
            _Db(self.ledger, users, replica_teams, memberships, tokens, budgets, jwt_mappings)
            if replica_teams is not None
            else self.writer_db
        )
        self._fail_commit = fail_commit
        self.transactions = 0

    @asynccontextmanager
    async def tx(self, *, timeout: object = None):
        self.transactions += 1
        snapshot = copy.deepcopy(self.writer_db)
        try:
            yield _Tx(self.writer_db, self.ledger)
            if self._fail_commit:
                raise RuntimeError("connection reset")
        except BaseException:
            self.writer_db.__dict__.update(snapshot.__dict__)
            raise


def _member(user_id: str, role: Literal["admin", "user"] = "user") -> Member:
    return Member(user_id=user_id, user_email=f"{user_id}@example.com", role=role)


def _team(
    *members: Member,
    default_team_member_models: list[str] | None = None,
    metadata: dict[str, object] | None = None,
) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=TEAM,
        members_with_roles=list(members),
        default_team_member_models=default_team_member_models,
        metadata=metadata,
    )


def _user(user_id: str, *teams: str) -> _UserRow:
    return _UserRow(user_id=user_id, user_email=f"{user_id}@example.com", teams=list(teams))


def _roster(prisma: _FakePrisma) -> list[tuple[str | None, str]]:
    return [(m.user_id, m.role) for m in prisma.writer_db.litellm_teamtable.rows[TEAM].members_with_roles]


def _memberships(prisma: _FakePrisma) -> list[tuple[object, object]]:
    return sorted((r["team_id"], r["user_id"]) for r in prisma.writer_db.litellm_teammembership.rows)


def _membership_budget(prisma: _FakePrisma, user_id: str) -> object:
    return next(r["budget_id"] for r in prisma.writer_db.litellm_teammembership.rows if r["user_id"] == user_id)


def _teams_of(prisma: _FakePrisma, user_id: str) -> list[str]:
    return prisma.writer_db.litellm_usertable.rows[user_id].teams


def _tokens(prisma: _FakePrisma) -> set[object]:
    return {r["token"] for r in prisma.writer_db.litellm_verificationtoken.rows}


def _writes(prisma: _FakePrisma) -> list[str]:
    return [s for s in prisma.ledger.statements if s not in READS]


def _cache_with(*keys: str) -> UserApiKeyCache:
    cache = UserApiKeyCache()
    for key in keys:
        cache.set_cache(key=key, value={"cache_key": key})
    return cache


async def _sync(prisma: _FakePrisma, plan: RosterPlan, cache: UserApiKeyCache | None = None):
    return await sync_team_roster(
        prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
        team_id=TEAM,
        plan=plan,
        user_api_key_dict=ADMIN,
        litellm_proxy_admin_name="default_user_id",
        user_api_key_cache=cache or UserApiKeyCache(),
        proxy_logging_obj=None,
    )


@pytest.mark.asyncio
async def test_target_adds_the_missing_members_and_removes_the_extra_ones():
    prisma = _FakePrisma(
        users=[_user("alice", TEAM), _user("bob", TEAM, "t2"), _user("carol")],
        teams=[_team(_member("alice", role="admin"), _member("bob"))],
        memberships=[(TEAM, "alice"), (TEAM, "bob")],
    )
    cache = _cache_with("bob", "carol", f"team_id:{TEAM}")

    outcome = await _sync(prisma, RosterTarget(member_ids=frozenset({"alice", "carol"})), cache)

    assert isinstance(outcome, RosterSync)
    assert (outcome.added, outcome.removed) == (frozenset({"carol"}), frozenset({"bob"}))
    written = prisma.writer_db.litellm_teamtable.rows[TEAM]
    assert written.updated_at is not None and outcome.team.updated_at == written.updated_at
    assert [(m.user_id, m.user_email, m.role) for m in outcome.team.members_with_roles] == [
        ("alice", "alice@example.com", "admin"),
        ("carol", "carol@example.com", "user"),
    ]
    assert _roster(prisma) == [("alice", "admin"), ("carol", "user")]
    assert _memberships(prisma) == [(TEAM, "alice"), (TEAM, "carol")]
    assert (_teams_of(prisma, "bob"), _teams_of(prisma, "carol")) == (["t2"], [TEAM])
    assert (cache.get_cache("bob"), cache.get_cache("carol"), cache.get_cache(f"team_id:{TEAM}")) == (None, None, None)


@pytest.mark.asyncio
async def test_delta_touches_only_the_named_members_and_tolerates_drift():
    prisma = _FakePrisma(
        users=[_user("alice", TEAM), _user("bob", TEAM), _user("carol", TEAM), _user("dave")],
        teams=[_team(_member("alice"), _member("bob"))],
        memberships=[(TEAM, "alice"), (TEAM, "bob"), (TEAM, "carol")],
    )

    outcome = await _sync(prisma, RosterDelta(add=frozenset({"alice", "carol"}), remove=frozenset({"bob", "dave"})))

    assert isinstance(outcome, RosterSync)
    assert (outcome.added, outcome.removed) == (frozenset({"carol"}), frozenset({"bob"}))
    assert _roster(prisma) == [("alice", "user"), ("carol", "user")]
    assert _memberships(prisma) == [(TEAM, "alice"), (TEAM, "carol")]
    assert (_teams_of(prisma, "alice"), _teams_of(prisma, "carol"), _teams_of(prisma, "bob")) == (
        [TEAM],
        [TEAM],
        [],
    )


@pytest.mark.asyncio
async def test_a_plan_matching_the_roster_only_takes_the_lock_and_leaves_the_cached_team_in_place():
    prisma = _FakePrisma(
        users=[_user("alice", TEAM), _user("bob", TEAM)],
        teams=[_team(_member("alice"), _member("bob"))],
        memberships=[(TEAM, "alice"), (TEAM, "bob")],
    )
    cache = _cache_with(f"team_id:{TEAM}")

    outcome = await _sync(prisma, RosterTarget(member_ids=frozenset({"alice", "bob"})), cache)

    assert isinstance(outcome, RosterSync)
    assert (outcome.added, outcome.removed) == (frozenset(), frozenset())
    assert prisma.ledger.statements == ["query_raw", "teamtable.find_unique"]
    assert cache.get_cache(f"team_id:{TEAM}") is not None


@pytest.mark.asyncio
async def test_a_delta_naming_nobody_reads_the_team_without_the_lock_or_a_transaction():
    prisma = _FakePrisma(
        users=[_user("alice", TEAM), _user("bob", TEAM)],
        teams=[_team(_member("alice"), _member("bob"))],
        memberships=[(TEAM, "alice"), (TEAM, "bob")],
    )

    outcome = await _sync(prisma, RosterDelta(add=frozenset(), remove=frozenset()))

    assert isinstance(outcome, RosterSync)
    assert [m.user_id for m in outcome.team.members_with_roles] == ["alice", "bob"]
    assert (outcome.added, outcome.removed) == (frozenset(), frozenset())
    assert prisma.ledger.statements == ["teamtable.find_unique"]
    assert prisma.transactions == 0


@pytest.mark.asyncio
async def test_a_delta_naming_nobody_reads_the_team_from_the_writer_not_the_replica():
    prisma = _FakePrisma(
        teams=[_team(_member("alice")).model_copy(update={"team_alias": "renamed"})],
        replica_teams=[_team(_member("alice")).model_copy(update={"team_alias": "stale"})],
    )

    outcome = await _sync(prisma, RosterDelta(add=frozenset(), remove=frozenset()))

    assert isinstance(outcome, RosterSync)
    assert outcome.team.team_alias == "renamed"


@pytest.mark.asyncio
async def test_every_member_row_the_plan_touches_is_locked_in_id_order_before_the_first_write():
    prisma = _FakePrisma(
        users=[_user("carol", TEAM), _user("alice"), _user("bob")],
        teams=[_team(_member("carol"))],
        memberships=[(TEAM, "carol")],
    )

    outcome = await _sync(prisma, RosterTarget(member_ids=frozenset({"bob", "alice"})))

    assert isinstance(outcome, RosterSync)
    assert (outcome.added, outcome.removed) == (frozenset({"alice", "bob"}), frozenset({"carol"}))
    assert prisma.ledger.locked_users == [["alice", "bob", "carol"]]
    first_write = next(index for index, statement in enumerate(prisma.ledger.statements) if statement not in READS)
    assert prisma.ledger.statements.index("query_raw", 1) < first_write


@pytest.mark.asyncio
async def test_a_deleted_team_answers_team_gone_without_writing():
    prisma = _FakePrisma(users=[_user("alice")])

    outcome = await _sync(prisma, RosterTarget(member_ids=frozenset({"alice"})))

    assert outcome == TeamGone(team_id=TEAM)
    assert _writes(prisma) == []


@pytest.mark.asyncio
async def test_an_unknown_member_answers_members_missing_and_leaves_the_roster_alone():
    prisma = _FakePrisma(
        users=[_user("alice", TEAM)],
        teams=[_team(_member("alice"))],
        memberships=[(TEAM, "alice")],
    )

    outcome = await _sync(prisma, RosterTarget(member_ids=frozenset({"zed", "ghost"})))

    assert outcome == MembersMissing(team_id=TEAM, user_ids=("ghost", "zed"))
    assert _roster(prisma) == [("alice", "user")]
    assert _memberships(prisma) == [(TEAM, "alice")]
    assert _teams_of(prisma, "alice") == [TEAM]
    assert _writes(prisma) == []


@pytest.mark.asyncio
async def test_added_members_each_get_a_budget_when_the_team_scopes_member_models():
    prisma = _FakePrisma(
        users=[_user("alice", TEAM), _user("carol"), _user("dave")],
        teams=[_team(_member("alice"), default_team_member_models=["gpt-x"])],
        memberships=[(TEAM, "alice")],
    )

    await _sync(prisma, RosterDelta(add=frozenset({"carol", "dave"}), remove=frozenset()))

    budgets = prisma.writer_db.litellm_budgettable.rows
    assert [(b["allowed_models"], b["created_by"], b["updated_by"]) for b in budgets] == [
        (("gpt-x",), "admin", "admin")
    ] * 2
    assert {_membership_budget(prisma, "carol"), _membership_budget(prisma, "dave")} == {
        b["budget_id"] for b in budgets
    }
    assert len({b["budget_id"] for b in budgets}) == 2
    assert _membership_budget(prisma, "alice") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("budgets", "expected_budget_id"),
    [([{"budget_id": "b-team"}], "b-team"), ([], None)],
    ids=["budget-row-present", "budget-row-gone"],
)
async def test_added_members_link_the_team_member_budget_when_the_team_names_one(
    budgets: list[dict[str, object]], expected_budget_id: str | None
):
    prisma = _FakePrisma(
        users=[_user("carol")],
        teams=[_team(metadata={"team_member_budget_id": "b-team"})],
        budgets=budgets,
    )

    await _sync(prisma, RosterDelta(add=frozenset({"carol"}), remove=frozenset()))

    assert _membership_budget(prisma, "carol") == expected_budget_id
    assert len(prisma.writer_db.litellm_budgettable.rows) == len(budgets)


@pytest.mark.asyncio
async def test_removed_members_lose_their_team_keys_and_every_cache_entry_for_them():
    prisma = _FakePrisma(
        users=[_user("alice", TEAM), _user("bob", TEAM, "t2")],
        teams=[_team(_member("alice"), _member("bob"))],
        memberships=[(TEAM, "alice"), (TEAM, "bob")],
        tokens=[
            {"token": "k-bob", "user_id": "bob", "team_id": TEAM},
            {"token": "k-bob-personal", "user_id": "bob"},
            {"token": "k-alice", "user_id": "alice", "team_id": TEAM},
        ],
        jwt_mappings=[{"token": "k-bob", "jwt_claim_name": "sub", "jwt_claim_value": "bob-claim", "jwt_issuer": None}],
    )
    jwt_key = jwt_key_mapping_cache_key("sub", "bob-claim", None)
    cache = _cache_with("k-bob", "k-alice", jwt_key, "bob")

    outcome = await _sync(prisma, RosterTarget(member_ids=frozenset({"alice"})), cache)

    assert isinstance(outcome, RosterSync) and outcome.removed == frozenset({"bob"})
    assert _tokens(prisma) == {"k-bob-personal", "k-alice"}
    assert [(r["token"], r["deleted_by"]) for r in prisma.writer_db.litellm_deletedverificationtoken.rows] == [
        ("k-bob", "admin")
    ]
    assert _teams_of(prisma, "bob") == ["t2"]
    assert (cache.get_cache("k-bob"), cache.get_cache(jwt_key), cache.get_cache("bob")) == (None, None, None)
    assert cache.get_cache("k-alice") is not None


class _SlowPublishRedisClient:
    def __init__(self) -> None:
        self.published: list[str] = []

    async def publish(self, channel: str, message: str) -> int:
        await asyncio.sleep(0.001)
        self.published.append(message)
        return 1


class _PubSubRedisCache:
    namespace = None

    def __init__(self, client: _SlowPublishRedisClient) -> None:
        self._client = client

    def init_pubsub_client(self) -> _SlowPublishRedisClient:
        return self._client


@pytest.mark.asyncio
async def test_removing_hundreds_of_members_at_once_broadcasts_every_key_and_member_eviction(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pubsub_module, "_in_flight_publishes", asyncio.Semaphore(16))
    monkeypatch.setattr(pubsub_module, "_pending_publishes", set())
    client = _SlowPublishRedisClient()
    removed = [f"u{index:03d}" for index in range(500)]
    prisma = _FakePrisma(
        users=[_user(u, TEAM) for u in removed],
        teams=[_team(*(_member(u) for u in removed))],
        memberships=[(TEAM, u) for u in removed],
        tokens=[{"token": f"key-{u}-{k}", "user_id": u, "team_id": TEAM} for u in removed for k in range(3)],
    )

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_PubSubRedisCache(client),
    ):
        outcome = await _sync(prisma, RosterTarget(member_ids=frozenset()))
        await pubsub_module.await_publish_backlog()

    assert isinstance(outcome, RosterSync) and outcome.removed == frozenset(removed)
    broadcast = {json.loads(message)["cache_key"] for message in client.published}
    assert {f"key-{u}-{k}" for u in removed for k in range(3)} <= broadcast
    assert set(removed) <= broadcast


class _WedgedPublishRedisClient:
    def __init__(self) -> None:
        self.release = asyncio.Event()

    async def publish(self, channel: str, message: str) -> int:
        await self.release.wait()
        return 1


@pytest.mark.asyncio
async def test_removing_members_clears_this_worker_before_waiting_on_a_wedged_redis(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pubsub_module, "_in_flight_publishes", asyncio.Semaphore(16))
    monkeypatch.setattr(pubsub_module, "_pending_publishes", set())
    monkeypatch.setattr(pubsub_module, "_PUBLISH_BACKLOG_WAIT_SECONDS", 0.5)
    client = _WedgedPublishRedisClient()
    removed = [f"u{index:03d}" for index in range(300)]
    token_of = {u: hashlib.sha256(f"key-{u}".encode()).hexdigest() for u in removed}
    member_keys = [
        key
        for u in removed
        for key in (
            u,
            token_of[u],
            team_membership_auth_cache_key(team_id=TEAM, user_id=u),
            team_membership_reservation_cache_key(user_id=u, team_id=TEAM),
        )
    ]  # comprehension-ok: a test fixture listing each member's four cache keys
    cache = UserApiKeyCache(
        in_memory_cache=InMemoryCache(max_size_in_memory=5_000),
        key_object_in_memory_cache=InMemoryCache(max_size_in_memory=5_000),
    )
    for key in member_keys:
        cache.in_memory_cache_for(key).set_cache(key=key, value={"cache_key": key})
    seeded = [key for key in member_keys if cache.in_memory_cache_for(key).get_cache(key) is not None]
    prisma = _FakePrisma(
        users=[_user(u, TEAM) for u in removed],
        teams=[_team(*(_member(u) for u in removed))],
        memberships=[(TEAM, u) for u in removed],
        tokens=[{"token": token_of[u], "user_id": u, "team_id": TEAM} for u in removed],
    )

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_PubSubRedisCache(client),  # pyright: ignore[reportArgumentType]  # the wedged fake only publishes
    ):
        sync = asyncio.create_task(_sync(prisma, RosterTarget(member_ids=frozenset()), cache))
        await asyncio.sleep(0.1)
        still_cached = [key for key in seeded if cache.in_memory_cache_for(key).get_cache(key) is not None]
        client.release.set()
        outcome = await sync
        await pubsub_module.await_publish_backlog()

    assert isinstance(outcome, RosterSync) and outcome.removed == frozenset(removed)
    assert seeded == member_keys
    assert still_cached == []


class _AuthRedisCache:
    namespace = None

    def __init__(self, entries: dict[str, object]) -> None:
        self.entries = dict(entries)

    async def async_get_cache(self, key: str, **kwargs: object) -> object | None:
        return self.entries.get(key)

    def set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.entries[key] = value

    async def async_set_cache(self, key: str, value: object, **kwargs: object) -> None:
        self.entries[key] = value

    def delete_cache(self, key: str) -> None:
        self.entries.pop(key, None)

    async def async_delete_cache(self, key: str) -> None:
        self.entries.pop(key, None)

    async def delete_cache_keys(self, keys: Sequence[str]) -> None:
        for key in keys:
            self.entries.pop(key, None)


@pytest.mark.asyncio
async def test_removing_members_clears_the_redis_copies_before_waiting_on_a_wedged_redis(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(pubsub_module, "_in_flight_publishes", asyncio.Semaphore(16))
    monkeypatch.setattr(pubsub_module, "_pending_publishes", set())
    monkeypatch.setattr(pubsub_module, "_PUBLISH_BACKLOG_WAIT_SECONDS", 0.5)
    client = _WedgedPublishRedisClient()
    removed = [f"u{index:03d}" for index in range(300)]
    token_of = {u: hashlib.sha256(f"key-{u}".encode()).hexdigest() for u in removed}
    member_keys = [
        key
        for u in removed
        for key in (
            u,
            token_of[u],
            team_membership_auth_cache_key(team_id=TEAM, user_id=u),
            team_membership_reservation_cache_key(user_id=u, team_id=TEAM),
        )
    ]  # comprehension-ok: a test fixture listing each member's four cache keys
    redis = _AuthRedisCache({key: {"cache_key": key} for key in member_keys})
    cache = UserApiKeyCache(
        in_memory_cache=InMemoryCache(max_size_in_memory=5_000),
        key_object_in_memory_cache=InMemoryCache(max_size_in_memory=5_000),
    )
    cache.attach_redis_cache(redis)  # pyright: ignore[reportArgumentType]  # a dict-backed fake of the Redis layer
    for key in member_keys:
        cache.in_memory_cache_for(key).set_cache(key=key, value={"cache_key": key})
    prisma = _FakePrisma(
        users=[_user(u, TEAM) for u in removed],
        teams=[_team(*(_member(u) for u in removed))],
        memberships=[(TEAM, u) for u in removed],
        tokens=[{"token": token_of[u], "user_id": u, "team_id": TEAM} for u in removed],
    )

    with patch(
        "litellm.proxy.common_utils.auth_cache_invalidation_pubsub.coordination_redis_cache",
        return_value=_PubSubRedisCache(client),  # pyright: ignore[reportArgumentType]  # the wedged fake only publishes
    ):
        sync = asyncio.create_task(_sync(prisma, RosterTarget(member_ids=frozenset()), cache))
        await asyncio.sleep(0.1)
        refilled = [key for key in member_keys if await cache.async_get_cache(key=key) is not None]
        client.release.set()
        outcome = await sync
        await pubsub_module.await_publish_backlog()

    assert isinstance(outcome, RosterSync) and outcome.removed == frozenset(removed)
    assert refilled == []
    assert not any(key in redis.entries for key in member_keys)


@pytest.mark.asyncio
async def test_a_failed_commit_leaves_every_table_and_cache_entry_as_it_was():
    prisma = _FakePrisma(
        users=[_user("alice", TEAM), _user("bob", TEAM), _user("carol")],
        teams=[_team(_member("alice"), _member("bob"))],
        memberships=[(TEAM, "alice"), (TEAM, "bob")],
        tokens=[{"token": "k-bob", "user_id": "bob", "team_id": TEAM}],
        fail_commit=True,
    )
    cache = _cache_with("k-bob", "carol")

    with pytest.raises(RuntimeError, match="connection reset"):
        await _sync(prisma, RosterTarget(member_ids=frozenset({"alice", "carol"})), cache)

    assert _roster(prisma) == [("alice", "user"), ("bob", "user")]
    assert _memberships(prisma) == [(TEAM, "alice"), (TEAM, "bob")]
    assert (_teams_of(prisma, "bob"), _teams_of(prisma, "carol")) == ([TEAM], [])
    assert _tokens(prisma) == {"k-bob"}
    assert prisma.writer_db.litellm_deletedverificationtoken.rows == []
    assert cache.get_cache("k-bob") is not None and cache.get_cache("carol") is not None


def _replace_with(member_ids: Sequence[str]) -> RosterPlan:
    return RosterTarget(member_ids=frozenset(member_ids))


def _add(member_ids: Sequence[str]) -> RosterPlan:
    return RosterDelta(add=frozenset(member_ids), remove=frozenset())


def _large_push(size: int) -> tuple[_FakePrisma, list[str], list[str]]:
    old = [f"old-{index:03d}" for index in range(size)]
    new = [f"new-{index:03d}" for index in range(size)]
    prisma = _FakePrisma(
        users=[*(_user(u, TEAM) for u in old), *(_user(u) for u in new)],
        teams=[_team(*(_member(u) for u in old), default_team_member_models=["gpt-x"])],
        memberships=[(TEAM, u) for u in old],
        tokens=[{"token": f"key-{u}", "user_id": u, "team_id": TEAM} for u in old],
    )
    return prisma, old, new


@pytest.mark.asyncio
@pytest.mark.parametrize(("plan_for", "keeps_old"), [(_replace_with, False), (_add, True)], ids=["replace", "add"])
async def test_the_statement_count_does_not_grow_with_the_member_count(
    plan_for: Callable[[Sequence[str]], RosterPlan], keeps_old: bool
):
    """A 500-member push used to run thousands of statements, one batch per member."""
    counts: dict[int, int] = {}
    for size in (5, 500):
        prisma, old, new = _large_push(size)

        outcome = await _sync(prisma, plan_for(new))

        assert isinstance(outcome, RosterSync) and outcome.added == frozenset(new)
        assert {user_id for user_id, _ in _roster(prisma)} == set(new) | (set(old) if keeps_old else set())
        assert _memberships(prisma) == sorted((TEAM, u) for u in [*new, *(old if keeps_old else [])])
        assert all(_teams_of(prisma, u) == [TEAM] for u in new)
        assert _tokens(prisma) == ({f"key-{u}" for u in old} if keeps_old else set())
        counts[size] = len(prisma.ledger.statements)
    assert counts[500] == counts[5]
