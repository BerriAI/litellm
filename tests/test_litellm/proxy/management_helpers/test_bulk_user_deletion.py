import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, Member, MemberDeleteRequest, UserAPIKeyAuth
from litellm.proxy.management_helpers.bulk_user_deletion import bulk_delete_users, bulk_remove_team_members
from litellm.types.proxy.management_endpoints.internal_user_endpoints import BulkDeleteUserRequest
from litellm.types.proxy.management_endpoints.team_endpoints import BulkTeamMemberDeleteRequest

ADMIN: Final = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin")
INTERNAL: Final = UserAPIKeyAuth(user_id="someone", user_role=LitellmUserRoles.INTERNAL_USER)
ORG_ADMIN: Final = UserAPIKeyAuth(user_id="org-admin", user_role=LitellmUserRoles.ORG_ADMIN)


class _UserRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str
    user_email: str | None = None
    teams: list[str] = []


class _Record(BaseModel):
    """Attribute access like a Prisma row, over whatever columns the test seeded."""

    model_config = ConfigDict(extra="allow")


def _in(where: Mapping[str, object], field: str) -> set[str] | None:
    clause = where.get(field)
    if isinstance(clause, dict) and "in" in clause:
        return set(clause["in"])
    if isinstance(clause, str):
        return {clause}
    return None


def _matches(row: Mapping[str, object], where: Mapping[str, object]) -> bool:
    if "OR" in where:
        return any(_matches(row, clause) for clause in where["OR"])
    return all((wanted := _in(where, field)) is not None and row.get(field) in wanted for field in where)


class _Rows:
    """A list-backed Prisma table supporting the `in`/equality/OR filters the helper issues."""

    def __init__(self, rows: Sequence[Mapping[str, object]] = ()) -> None:
        self.rows: list[dict[str, object]] = [dict(r) for r in rows]

    async def find_many(self, where: Mapping[str, object]) -> list[_Record]:
        return [_Record.model_validate(r) for r in self.rows if _matches(r, where)]

    async def delete_many(self, where: Mapping[str, object]) -> int:
        before = len(self.rows)
        self.rows = [r for r in self.rows if not _matches(r, where)]
        return before - len(self.rows)

    async def create_many(self, data: Sequence[Mapping[str, object]]) -> int:
        self.rows.extend(dict(r) for r in data)
        return len(data)


class _UserTable:
    def __init__(self, users: Sequence[_UserRow]) -> None:
        self.rows: dict[str, _UserRow] = {u.user_id: u for u in users}

    async def find_many(self, where: Mapping[str, object]) -> list[_UserRow]:
        return [u for u in self.rows.values() if _matches(u.model_dump(), where)]

    async def update(self, where: Mapping[str, str], data: Mapping[str, Mapping[str, Sequence[str]]]) -> _UserRow:
        row = self.rows[where["user_id"]]
        updated = row.model_copy(update={"teams": list(data["teams"]["set"])})
        self.rows[row.user_id] = updated
        return updated

    async def delete_many(self, where: Mapping[str, object]) -> int:
        doomed = [uid for uid, u in self.rows.items() if _matches(u.model_dump(), where)]
        for uid in doomed:
            del self.rows[uid]
        return len(doomed)


class _TeamTable:
    def __init__(self, teams: Sequence[LiteLLM_TeamTable]) -> None:
        self.rows: dict[str, LiteLLM_TeamTable] = {t.team_id: t for t in teams}
        self.update_calls = 0

    async def find_unique(self, where: Mapping[str, str]) -> LiteLLM_TeamTable | None:
        return self.rows.get(where["team_id"])

    async def update(self, where: Mapping[str, str], data: Mapping[str, str]) -> LiteLLM_TeamTable:
        self.update_calls += 1
        team = self.rows[where["team_id"]]
        team.members_with_roles = [Member(**m) for m in json.loads(data["members_with_roles"])]
        return team


class _Db:
    def __init__(
        self,
        users: Sequence[_UserRow],
        teams: Sequence[LiteLLM_TeamTable],
        memberships: Sequence[tuple[str, str]] = (),
        tokens: Sequence[Mapping[str, object]] = (),
        invitations: Sequence[Mapping[str, object]] = (),
        org_memberships: Sequence[Mapping[str, object]] = (),
    ) -> None:
        self.litellm_usertable = _UserTable(users)
        self.litellm_teamtable = _TeamTable(teams)
        self.litellm_teammembership = _Rows([{"team_id": t, "user_id": u} for t, u in memberships])
        self.litellm_verificationtoken = _Rows(tokens)
        self.litellm_deletedverificationtoken = _Rows()
        self.litellm_invitationlink = _Rows(invitations)
        self.litellm_organizationmembership = _Rows(org_memberships)


class _Tx:
    def __init__(self, db: _Db, on_lock: Callable[[str], None], fail_locks: frozenset[str]) -> None:
        self.litellm_teamtable = db.litellm_teamtable
        self.litellm_usertable = db.litellm_usertable
        self.litellm_teammembership = db.litellm_teammembership
        self.litellm_verificationtoken = db.litellm_verificationtoken
        self.litellm_deletedverificationtoken = db.litellm_deletedverificationtoken
        self._on_lock = on_lock
        self._fail_locks = fail_locks
        self.locks: list[str] = []
        self.roster_reads: list[str] = []

    async def query_raw(self, sql: str, *args: object) -> list[dict[str, object]]:
        team_id = str(args[0])
        if "pg_advisory_xact_lock" in sql:
            if team_id in self._fail_locks:
                raise RuntimeError("lock timeout")
            self.locks.append(team_id)
            self._on_lock(team_id)
            return []
        assert self.locks == [team_id], "roster must be read under this team's advisory lock"
        self.roster_reads.append(team_id)
        team = self.litellm_teamtable.rows.get(team_id)
        if team is None:
            return []
        return [{"members_with_roles": json.dumps([m.model_dump() for m in team.members_with_roles])}]


class _FakePrisma:
    def __init__(
        self,
        users: Sequence[_UserRow] = (),
        teams: Sequence[LiteLLM_TeamTable] = (),
        memberships: Sequence[tuple[str, str]] = (),
        tokens: Sequence[Mapping[str, object]] = (),
        invitations: Sequence[Mapping[str, object]] = (),
        org_memberships: Sequence[Mapping[str, object]] = (),
        on_lock: Callable[[str], None] = lambda _: None,
        fail_locks: frozenset[str] = frozenset(),
    ) -> None:
        self.db = _Db(users, teams, memberships, tokens, invitations, org_memberships)
        self._on_lock = on_lock
        self._fail_locks = fail_locks
        self.locks: list[str] = []
        self.roster_reads: list[str] = []

    @asynccontextmanager
    async def tx(self):
        tx = _Tx(self.db, self._on_lock, self._fail_locks)
        yield tx
        self.locks.extend(tx.locks)
        self.roster_reads.extend(tx.roster_reads)


def _team(team_id: str, *members: str, org: str | None = None) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=team_id,
        organization_id=org,
        members_with_roles=[Member(user_id=m, user_email=f"{m}@example.com", role="user") for m in members],
    )


def _user(user_id: str, *teams: str) -> _UserRow:
    return _UserRow(user_id=user_id, user_email=f"{user_id}@example.com", teams=list(teams))


def _roster(prisma: _FakePrisma, team_id: str) -> list[str | None]:
    return [m.user_id for m in prisma.db.litellm_teamtable.rows[team_id].members_with_roles]


async def _delete(prisma: _FakePrisma, user_ids: Sequence[str], caller: UserAPIKeyAuth = ADMIN):
    return await bulk_delete_users(
        data=BulkDeleteUserRequest(user_ids=tuple(user_ids)),
        user_api_key_dict=caller,
        prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
        litellm_proxy_admin_name="default_user_id",
        litellm_changed_by=None,
    )


async def _remove(prisma: _FakePrisma, team_id: str, members: Sequence[Mapping[str, str]], caller=ADMIN):
    return await bulk_remove_team_members(
        data=BulkTeamMemberDeleteRequest(team_id=team_id, members=tuple(MemberDeleteRequest(**m) for m in members)),
        user_api_key_dict=caller,
        prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
    )


@pytest.mark.asyncio
async def test_bulk_delete_removes_users_from_every_team_and_store():
    prisma = _FakePrisma(
        users=[_user("u1", "t1", "t2"), _user("u2", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "u2", "keep"), _team("t2", "u1", "other")],
        memberships=[("t1", "u1"), ("t2", "u1"), ("t1", "u2"), ("t1", "keep")],
        tokens=[{"token": "k1", "user_id": "u1", "team_id": "t1"}, {"token": "k2", "user_id": "keep"}],
        invitations=[
            {"id": "i1", "user_id": "u2", "created_by": "admin", "updated_by": "admin"},
            {"id": "i2", "user_id": "keep", "created_by": "u1", "updated_by": "admin"},
            {"id": "i3", "user_id": "keep", "created_by": "admin", "updated_by": "admin"},
        ],
        org_memberships=[{"user_id": "u1", "organization_id": "o1", "user_role": "internal_user"}],
    )

    response = await _delete(prisma, ["u1", "u2"])

    assert (response.total_requested, response.successful_deletions, response.failed_deletions) == (2, 2, 0)
    assert [(r.user_id, r.user_email, r.success, r.teams_removed) for r in response.results] == [
        ("u1", "u1@example.com", True, ("t1", "t2")),
        ("u2", "u2@example.com", True, ("t1",)),
    ]
    assert _roster(prisma, "t1") == ["keep"] and _roster(prisma, "t2") == ["other"]
    assert set(prisma.db.litellm_usertable.rows) == {"keep"}
    assert prisma.db.litellm_teammembership.rows == [{"team_id": "t1", "user_id": "keep"}]
    assert [t["token"] for t in prisma.db.litellm_verificationtoken.rows] == ["k2"]
    assert [t["token"] for t in prisma.db.litellm_deletedverificationtoken.rows] == ["k1"]
    assert [i["id"] for i in prisma.db.litellm_invitationlink.rows] == ["i3"]
    assert prisma.db.litellm_organizationmembership.rows == []
    assert sorted(prisma.locks) == ["t1", "t2"] and sorted(prisma.roster_reads) == ["t1", "t2"]


@pytest.mark.asyncio
async def test_bulk_delete_finds_teams_through_membership_rows_when_user_teams_array_is_stale():
    prisma = _FakePrisma(
        users=[_user("u1")],
        teams=[_team("t1", "u1", "keep")],
        memberships=[("t1", "u1")],
    )

    response = await _delete(prisma, ["u1"])

    assert response.results[0].teams_removed == ("t1",)
    assert _roster(prisma, "t1") == ["keep"]
    assert prisma.db.litellm_teammembership.rows == []


@pytest.mark.asyncio
async def test_bulk_delete_reads_roster_under_lock_so_a_concurrent_add_survives():
    team = _team("t1", "u1")

    def concurrent_member_add(team_id: str) -> None:
        team.members_with_roles.append(Member(user_id="late", role="user"))

    prisma = _FakePrisma(users=[_user("u1", "t1")], teams=[team], on_lock=concurrent_member_add)

    response = await _delete(prisma, ["u1"])

    assert response.results[0].success is True
    assert _roster(prisma, "t1") == ["late"]


@pytest.mark.asyncio
async def test_bulk_delete_reports_missing_and_duplicate_ids_per_item_and_still_deletes_the_rest():
    prisma = _FakePrisma(users=[_user("u1")])

    response = await _delete(prisma, ["u1", "ghost", "u1"])

    assert (response.successful_deletions, response.failed_deletions) == (1, 2)
    assert [(r.user_id, r.success, r.error) for r in response.results] == [
        ("u1", True, None),
        ("ghost", False, "User id=ghost not found"),
        ("u1", False, "Duplicate user_id in request: u1"),
    ]
    assert prisma.db.litellm_usertable.rows == {}


@pytest.mark.asyncio
async def test_bulk_delete_keeps_user_when_a_team_rewrite_fails_and_deletes_the_others():
    prisma = _FakePrisma(
        users=[_user("u1", "bad", "good"), _user("u2", "good")],
        teams=[_team("bad", "u1"), _team("good", "u1", "u2")],
        fail_locks=frozenset({"bad"}),
    )

    response = await _delete(prisma, ["u1", "u2"])

    assert [(r.user_id, r.success, r.teams_removed) for r in response.results] == [
        ("u1", False, ("good",)),
        ("u2", True, ("good",)),
    ]
    assert response.results[0].error == "Failed to remove from team bad: lock timeout"
    assert set(prisma.db.litellm_usertable.rows) == {"u1"}
    assert _roster(prisma, "bad") == ["u1"] and _roster(prisma, "good") == []


@pytest.mark.asyncio
async def test_bulk_delete_rejects_non_admin_callers_before_touching_the_db():
    prisma = _FakePrisma(users=[_user("u1")])

    with pytest.raises(HTTPException) as exc:
        await _delete(prisma, ["u1"], caller=INTERNAL)

    assert exc.value.status_code == 403
    assert set(prisma.db.litellm_usertable.rows) == {"u1"}


@pytest.mark.asyncio
async def test_org_admin_deletes_only_users_fully_inside_their_orgs():
    prisma = _FakePrisma(
        users=[_user("inside"), _user("straddles"), _user("orgless")],
        org_memberships=[
            {"user_id": "org-admin", "organization_id": "o1", "user_role": LitellmUserRoles.ORG_ADMIN.value},
            {"user_id": "inside", "organization_id": "o1", "user_role": "internal_user"},
            {"user_id": "straddles", "organization_id": "o1", "user_role": "internal_user"},
            {"user_id": "straddles", "organization_id": "o2", "user_role": "internal_user"},
        ],
    )

    response = await _delete(prisma, ["inside", "straddles", "orgless"], caller=ORG_ADMIN)

    assert [r.success for r in response.results] == [True, False, False]
    assert all("not within your admin scope" in (r.error or "") for r in response.results[1:])
    assert set(prisma.db.litellm_usertable.rows) == {"straddles", "orgless"}
    assert {(m["user_id"], m["organization_id"]) for m in prisma.db.litellm_organizationmembership.rows} == {
        ("org-admin", "o1"),
        ("straddles", "o1"),
        ("straddles", "o2"),
    }


@pytest.mark.asyncio
async def test_bulk_member_delete_removes_by_id_and_email_and_keeps_the_rest():
    prisma = _FakePrisma(
        users=[_user("u1", "t1", "t2"), _user("u2", "t1"), _user("keep", "t1")],
        teams=[_team("t1", "u1", "u2", "keep")],
        memberships=[("t1", "u1"), ("t1", "u2"), ("t1", "keep")],
        tokens=[
            {"token": "team-key", "user_id": "u1", "team_id": "t1"},
            {"token": "other-team-key", "user_id": "u1", "team_id": "t2"},
            {"token": "keep-key", "user_id": "keep", "team_id": "t1"},
        ],
    )

    response = await _remove(prisma, "t1", [{"user_id": "u1"}, {"user_email": "u2@example.com"}])

    assert (response.team_id, response.successful_deletions, response.failed_deletions) == ("t1", 2, 0)
    assert [(r.user_id, r.user_email, r.success) for r in response.results] == [
        ("u1", None, True),
        (None, "u2@example.com", True),
    ]
    assert _roster(prisma, "t1") == ["keep"]
    users = prisma.db.litellm_usertable.rows
    assert users["u1"].teams == ["t2"] and users["u2"].teams == [] and users["keep"].teams == ["t1"]
    assert prisma.db.litellm_teammembership.rows == [{"team_id": "t1", "user_id": "keep"}]
    assert sorted(t["token"] for t in prisma.db.litellm_verificationtoken.rows) == ["keep-key", "other-team-key"]
    assert [t["token"] for t in prisma.db.litellm_deletedverificationtoken.rows] == ["team-key"]
    assert prisma.locks == ["t1"] and prisma.roster_reads == ["t1"]


@pytest.mark.asyncio
async def test_bulk_member_delete_reports_members_not_on_the_team_without_rewriting_the_roster():
    prisma = _FakePrisma(users=[_user("u1", "t1"), _user("elsewhere")], teams=[_team("t1", "u1")])

    response = await _remove(prisma, "t1", [{"user_id": "elsewhere"}, {"user_email": "nobody@example.com"}])

    assert [(r.success, r.error) for r in response.results] == [
        (False, "User not found in team"),
        (False, "User not found in team"),
    ]
    assert (response.successful_deletions, response.failed_deletions) == (0, 2)
    assert prisma.db.litellm_teamtable.update_calls == 0
    assert _roster(prisma, "t1") == ["u1"]


@pytest.mark.asyncio
async def test_bulk_member_delete_cleans_a_user_whose_teams_array_still_names_the_team():
    prisma = _FakePrisma(users=[_user("stale", "t1")], teams=[_team("t1", "other")], memberships=[("t1", "stale")])

    response = await _remove(prisma, "t1", [{"user_id": "stale"}])

    assert response.results[0].success is True
    assert prisma.db.litellm_usertable.rows["stale"].teams == []
    assert prisma.db.litellm_teammembership.rows == []
    assert _roster(prisma, "t1") == ["other"] and prisma.db.litellm_teamtable.update_calls == 0


@pytest.mark.asyncio
async def test_bulk_member_delete_rejects_unknown_team_and_unauthorized_callers():
    prisma = _FakePrisma(users=[_user("u1", "t1")], teams=[_team("t1", "u1")])

    with pytest.raises(HTTPException) as missing:
        await _remove(prisma, "nope", [{"user_id": "u1"}])
    with pytest.raises(HTTPException) as forbidden:
        await _remove(prisma, "t1", [{"user_id": "u1"}], caller=INTERNAL)

    assert missing.value.status_code == 400
    assert forbidden.value.status_code == 403
    assert _roster(prisma, "t1") == ["u1"] and prisma.locks == []


@pytest.mark.asyncio
async def test_team_admin_may_bulk_remove_members():
    team = _team("t1", "lead", "u1")
    team.members_with_roles[0].role = "admin"
    prisma = _FakePrisma(users=[_user("lead", "t1"), _user("u1", "t1")], teams=[team])

    response = await _remove(prisma, "t1", [{"user_id": "u1"}], caller=UserAPIKeyAuth(user_id="lead"))

    assert response.results[0].success is True
    assert _roster(prisma, "t1") == ["lead"]


def test_request_models_enforce_batch_bounds():
    with pytest.raises(ValidationError):
        BulkDeleteUserRequest(user_ids=())
    with pytest.raises(ValidationError):
        BulkDeleteUserRequest(user_ids=tuple(f"u{i}" for i in range(501)))
    with pytest.raises(ValidationError):
        BulkTeamMemberDeleteRequest(team_id="t1", members=())
    with pytest.raises(ValidationError):
        BulkTeamMemberDeleteRequest(
            team_id="t1", members=tuple(MemberDeleteRequest(user_id=f"u{i}") for i in range(501))
        )
    assert len(BulkDeleteUserRequest(user_ids=tuple(f"u{i}" for i in range(500))).user_ids) == 500
