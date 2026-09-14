import json
from contextlib import asynccontextmanager
from typing import Final

import httpx
import pytest
from fastapi import HTTPException
from prisma.errors import UniqueViolationError
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.caching.caching import DualCache
from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, Member, UserAPIKeyAuth
from litellm.proxy.management_helpers.bulk_user_creation import bulk_create_users
from litellm.types.proxy.management_endpoints.internal_user_endpoints import (
    BulkNewUserItem,
    BulkNewUserRequest,
)

ADMIN: Final = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN)
INTERNAL: Final = UserAPIKeyAuth(user_id="someone", user_role=LitellmUserRoles.INTERNAL_USER)


class _UserRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_id: str
    user_email: str | None = None
    user_role: str | None = None
    teams: list[str] = []
    max_budget: float | None = None


class _UserTable:
    """Enough of the Prisma user table for the bulk path: set lookups, one create_many and per-row fallbacks."""

    def __init__(
        self,
        fail_ids: frozenset[str] = frozenset(),
        commit_then_drop: bool = False,
        raced_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.rows: dict[str, _UserRow] = {}
        self.fail_ids = fail_ids
        self.commit_then_drop = commit_then_drop
        self.raced_ids = raced_ids
        self.create_many_calls = 0

    async def count(self, where: object = None) -> int:
        return 0 if where is not None else len(self.rows)

    async def find_many(self, where: dict[str, dict[str, object]]) -> list[_UserRow]:
        if "user_id" in where:
            wanted = where["user_id"]["in"]
            return [row for row in self.rows.values() if row.user_id in wanted]
        wanted_emails = {str(e).lower() for e in where["user_email"]["in"]}
        return [row for row in self.rows.values() if (row.user_email or "").lower() in wanted_emails]

    async def create(self, data: dict[str, object]) -> _UserRow:
        row = _UserRow.model_validate(data)
        if row.user_id in self.fail_ids or row.user_id in self.rows:
            raise RuntimeError(f"insert failed for {row.user_id}")
        self.rows[row.user_id] = row
        return row

    async def create_many(self, data: list[dict[str, object]]) -> int:
        self.create_many_calls += 1
        rows = [_UserRow.model_validate(d) for d in data]
        if any(row.user_id in self.fail_ids for row in rows):
            raise RuntimeError("batch insert failed")
        raced = [row.user_id for row in rows if row.user_id in self.raced_ids]
        if raced:
            for user_id in raced:
                self.rows[user_id] = _UserRow(user_id=user_id, user_email=f"{user_id}@other-request.example")
            raise UniqueViolationError({}, message="Unique constraint failed on the fields: (`user_id`)")
        for row in rows:
            self.rows[row.user_id] = row
        if self.commit_then_drop:
            raise httpx.ReadError("connection reset after commit")
        return len(rows)

    async def update(self, where: dict[str, str], data: dict[str, object]) -> _UserRow:
        row = self.rows[where["user_id"]]
        updated = _UserRow.model_validate({**row.model_dump(), **data})
        self.rows[row.user_id] = updated
        return updated


class _TeamTable:
    def __init__(self, teams: list[LiteLLM_TeamTable]) -> None:
        self.rows = {team.team_id: team for team in teams}
        self.update_calls = 0

    async def find_many(self, where: dict[str, dict[str, list[str]]]) -> list[LiteLLM_TeamTable]:
        return [self.rows[team_id] for team_id in where["team_id"]["in"] if team_id in self.rows]

    async def update(self, where: dict[str, str], data: dict[str, str]) -> LiteLLM_TeamTable:
        self.update_calls += 1
        team = self.rows[where["team_id"]]
        team.members_with_roles = [Member(**m) for m in json.loads(data["members_with_roles"])]
        return team


class _MembershipTable:
    def __init__(self) -> None:
        self.rows: list[dict[str, object]] = []

    async def create_many(self, data: list[dict[str, object]], skip_duplicates: bool = False) -> int:
        self.rows.extend(data)
        return len(data)


class _Tx:
    def __init__(self, db: "_Db") -> None:
        self.litellm_teamtable = db.litellm_teamtable
        self.litellm_teammembership = db.litellm_teammembership
        self.locks: list[str] = []

    async def query_raw(self, sql: str, *args: object) -> list[dict[str, object]]:
        if "pg_advisory_xact_lock" in sql:
            self.locks.append(str(args[0]))
            return []
        team = self.litellm_teamtable.rows.get(str(args[0]))
        if team is None:
            return []
        return [{"members_with_roles": [m.model_dump() for m in team.members_with_roles]}]


class _Db:
    def __init__(
        self,
        teams: list[LiteLLM_TeamTable],
        fail_ids: frozenset[str] = frozenset(),
        commit_then_drop: bool = False,
        raced_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.litellm_usertable = _UserTable(fail_ids, commit_then_drop, raced_ids)
        self.litellm_teamtable = _TeamTable(teams)
        self.litellm_teammembership = _MembershipTable()


class _FakePrisma:
    def __init__(
        self,
        teams: list[LiteLLM_TeamTable] | None = None,
        fail_ids: frozenset[str] = frozenset(),
        commit_then_drop: bool = False,
        raced_ids: frozenset[str] = frozenset(),
    ) -> None:
        self.db = _Db(teams or [], fail_ids, commit_then_drop, raced_ids)
        self.tx_count = 0
        self.locks: list[str] = []

    def jsonify_object(self, data: dict[str, object]) -> dict[str, object]:
        return data

    @asynccontextmanager
    async def tx(self):
        self.tx_count += 1
        tx = _Tx(self.db)
        yield tx
        self.locks.extend(tx.locks)


class _License:
    def __init__(self, max_users: int | None = None) -> None:
        self.max_users = max_users
        self.seen: list[int] = []

    def is_over_limit(self, total_users: int) -> bool:
        self.seen.append(total_users)
        return self.max_users is not None and total_users > self.max_users


def _team(team_id: str, members: list[Member] | None = None) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(team_id=team_id, members_with_roles=members or [])


async def _no_keys(**kwargs: object) -> dict[str, object]:
    raise AssertionError(f"key generation was not requested: {kwargs}")


async def _run(prisma, users, caller=ADMIN, license=None, generate_key=_no_keys):
    return await bulk_create_users(
        users=[BulkNewUserItem(**u) for u in users],
        user_api_key_dict=caller,
        prisma_client=prisma,
        license_check=license or _License(),
        litellm_proxy_admin_name="default_user_id",
        user_api_key_cache=DualCache(),
        generate_key=generate_key,
    )


@pytest.mark.asyncio
async def test_creates_users_and_team_membership_in_every_store():
    prisma = _FakePrisma(teams=[_team("t1", [Member(user_id="existing", role="admin")]), _team("t2")])
    response = await _run(
        prisma,
        [
            {"user_id": "u1", "user_email": "a@example.com", "teams": ["t1", "t2"], "max_budget": 50},
            {"user_id": "u2", "user_email": "b@example.com", "teams": ["t1"]},
            {"user_id": "u3", "user_email": "c@example.com"},
        ],
    )

    assert (response.total_requested, response.successful_creations, response.failed_creations) == (3, 3, 0)
    assert [r.user_id for r in response.results] == ["u1", "u2", "u3"]
    assert all(r.success and r.key is None and r.error is None for r in response.results)
    assert [r.teams for r in response.results] == [("t1", "t2"), ("t1",), ()]

    users = prisma.db.litellm_usertable.rows
    assert users["u1"].teams == ["t1", "t2"] and users["u1"].max_budget == 50
    assert users["u2"].teams == ["t1"] and users["u3"].teams == []
    assert [m.user_id for m in prisma.db.litellm_teamtable.rows["t1"].members_with_roles] == ["existing", "u1", "u2"]
    assert [m.user_id for m in prisma.db.litellm_teamtable.rows["t2"].members_with_roles] == ["u1"]
    assert sorted((m["team_id"], m["user_id"]) for m in prisma.db.litellm_teammembership.rows) == [
        ("t1", "u1"),
        ("t1", "u2"),
        ("t2", "u1"),
    ]


@pytest.mark.asyncio
async def test_one_insert_and_one_locked_write_per_team():
    prisma = _FakePrisma(teams=[_team("t1"), _team("t2")])
    await _run(
        prisma,
        [{"user_id": f"u{i}", "teams": ["t1"] if i % 2 else ["t1", "t2"]} for i in range(20)],
    )

    assert prisma.db.litellm_usertable.create_many_calls == 1
    assert prisma.tx_count == 2
    assert sorted(prisma.locks) == ["t1", "t2"]
    assert prisma.db.litellm_teamtable.update_calls == 2
    assert len(prisma.db.litellm_teamtable.rows["t1"].members_with_roles) == 20
    assert len(prisma.db.litellm_teamtable.rows["t2"].members_with_roles) == 10


@pytest.mark.asyncio
async def test_bad_rows_fail_alone_and_good_rows_still_land():
    prisma = _FakePrisma(teams=[_team("t1")])
    prisma.db.litellm_usertable.rows["taken"] = _UserRow(user_id="taken", user_email="Taken@Example.com")
    response = await _run(
        prisma,
        [
            {"user_id": "u1", "user_email": "a@example.com", "teams": ["t1"]},
            {"user_id": "u2", "user_email": "A@EXAMPLE.COM"},
            {"user_id": "u1", "user_email": "z@example.com"},
            {"user_id": "u3", "user_email": "taken@example.com"},
            {"user_id": "taken"},
            {"user_id": "u4", "teams": ["missing"]},
            {"user_id": "u5", "teams": ["t1", "missing"]},
            {"user_id": "u6", "budget_duration": "not-a-duration"},
            {"user_id": "u7", "user_email": "ok@example.com", "teams": ["t1"]},
        ],
    )

    assert [r.success for r in response.results] == [True, False, False, False, False, False, False, False, True]
    assert (response.successful_creations, response.failed_creations) == (2, 7)
    errors = [r.error for r in response.results]
    assert "Duplicate user_email" in errors[1]
    assert "Duplicate user_id" in errors[2]
    assert "already exists" in errors[3] and "already exists" in errors[4]
    assert "missing" in errors[5] and "does not exist" in errors[5]
    assert "missing" in errors[6]
    assert errors[7] is not None

    assert set(prisma.db.litellm_usertable.rows) == {"taken", "u1", "u7"}
    assert [m.user_id for m in prisma.db.litellm_teamtable.rows["t1"].members_with_roles] == ["u1", "u7"]


@pytest.mark.asyncio
async def test_insert_failure_falls_back_to_per_row_and_reports_only_that_row():
    prisma = _FakePrisma(teams=[_team("t1")], fail_ids=frozenset({"u2"}))
    response = await _run(
        prisma,
        [{"user_id": "u1", "teams": ["t1"]}, {"user_id": "u2", "teams": ["t1"]}, {"user_id": "u3"}],
    )

    assert [r.success for r in response.results] == [True, False, True]
    assert "insert failed for u2" in (response.results[1].error or "")
    assert set(prisma.db.litellm_usertable.rows) == {"u1", "u3"}
    assert [m.user_id for m in prisma.db.litellm_teamtable.rows["t1"].members_with_roles] == ["u1"]


@pytest.mark.asyncio
async def test_insert_that_committed_but_lost_its_response_still_counts_as_created():
    prisma = _FakePrisma(teams=[_team("t1")], commit_then_drop=True)
    response = await _run(prisma, [{"user_id": "u1", "teams": ["t1"]}, {"user_id": "u2"}])

    assert [r.success for r in response.results] == [True, True]
    assert [r.error for r in response.results] == [None, None]
    assert set(prisma.db.litellm_usertable.rows) == {"u1", "u2"}
    assert [m.user_id for m in prisma.db.litellm_teamtable.rows["t1"].members_with_roles] == ["u1"]


@pytest.mark.asyncio
async def test_user_id_taken_by_a_concurrent_request_is_not_claimed_by_this_batch():
    prisma = _FakePrisma(teams=[_team("t1")], raced_ids=frozenset({"u1"}))
    response = await _run(prisma, [{"user_id": "u1", "teams": ["t1"]}, {"user_id": "u2", "teams": ["t1"]}])

    assert [r.success for r in response.results] == [False, True]
    assert "User id=u1 already exists" in (response.results[0].error or "")
    assert prisma.db.litellm_usertable.rows["u1"].user_email == "u1@other-request.example"
    assert [m.user_id for m in prisma.db.litellm_teamtable.rows["t1"].members_with_roles] == ["u2"]


@pytest.mark.asyncio
async def test_team_write_failure_keeps_user_and_reports_it_on_the_row():
    prisma = _FakePrisma(teams=[_team("t1"), _team("t2")])

    async def explode(where, data):
        raise RuntimeError("roster write failed")

    prisma.db.litellm_teamtable.update = explode
    response = await _run(prisma, [{"user_id": "u1", "teams": ["t1", "t2"]}])

    result = response.results[0]
    assert result.success is True
    assert result.teams == ()
    assert "t1" in (result.error or "") and "roster write failed" in (result.error or "")
    assert prisma.db.litellm_usertable.rows["u1"].teams == []
    assert (response.successful_creations, response.failed_creations) == (1, 0)


@pytest.mark.asyncio
async def test_keys_are_opt_in_per_row():
    prisma = _FakePrisma()
    calls: list[dict[str, object]] = []

    async def generate_key(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return {"token": f"sk-{kwargs['user_id']}"}

    response = await _run(
        prisma,
        [
            {"user_id": "u1"},
            {
                "user_id": "u2",
                "auto_create_key": True,
                "models": ["gpt-4o"],
                "key_alias": "u2-key",
                "blocked": True,
                "permissions": {"get_spend_routes": True},
                "aliases": {"fast": "gpt-4o"},
                "config": {"tier": "gold"},
                "budget_fallbacks": {"gpt-4o": ["gpt-4o-mini"]},
            },
            {"user_id": "u3", "auto_create_key": False},
        ],
        generate_key=generate_key,
    )

    assert [r.key for r in response.results] == [None, "sk-u2", None]
    assert len(calls) == 1
    assert calls[0]["user_id"] == "u2" and calls[0]["table_name"] == "key"
    assert calls[0]["models"] == ("gpt-4o",) and calls[0]["key_alias"] == "u2-key"
    assert calls[0]["blocked"] is True
    assert calls[0]["permissions"] == {"get_spend_routes": True}
    assert calls[0]["aliases"] == {"fast": "gpt-4o"}
    assert calls[0]["config"] == {"tier": "gold"}
    assert calls[0]["budget_fallbacks"] == {"gpt-4o": ("gpt-4o-mini",)}
    assert set(prisma.db.litellm_usertable.rows) == {"u1", "u2", "u3"}


@pytest.mark.asyncio
async def test_non_admin_cannot_create_admin_users_but_other_rows_proceed():
    prisma = _FakePrisma()
    response = await _run(
        prisma,
        [{"user_id": "u1", "user_role": "proxy_admin"}, {"user_id": "u2", "user_role": "internal_user"}],
        caller=INTERNAL,
    )

    assert [r.success for r in response.results] == [False, True]
    assert "Only proxy admins" in (response.results[0].error or "")
    assert set(prisma.db.litellm_usertable.rows) == {"u2"}


@pytest.mark.asyncio
async def test_license_is_checked_once_against_the_whole_batch():
    prisma = _FakePrisma()
    prisma.db.litellm_usertable.rows["existing"] = _UserRow(user_id="existing")
    license = _License(max_users=3)

    with pytest.raises(HTTPException) as exc:
        await _run(prisma, [{"user_id": f"u{i}"} for i in range(3)], license=license)

    assert exc.value.status_code == 403
    assert license.seen == [4]
    assert set(prisma.db.litellm_usertable.rows) == {"existing"}

    ok = await _run(prisma, [{"user_id": f"u{i}"} for i in range(2)], license=license)
    assert ok.successful_creations == 2

    resend = await _run(prisma, [{"user_id": f"u{i}"} for i in range(2)], license=license)
    assert [r.success for r in resend.results] == [False, False]
    assert all("already exists" in (r.error or "") for r in resend.results)
    assert license.seen == [4, 3]
    assert set(prisma.db.litellm_usertable.rows) == {"existing", "u0", "u1"}


def test_request_rejects_empty_oversized_and_invite_rows():
    with pytest.raises(ValidationError):
        BulkNewUserRequest(users=[])
    with pytest.raises(ValidationError):
        BulkNewUserRequest(users=[{"user_email": f"{i}@example.com"} for i in range(501)])
    with pytest.raises(ValidationError, match="send_invite_email"):
        BulkNewUserItem(user_email="a@example.com", send_invite_email=True)
    assert len(BulkNewUserRequest(users=[{"user_email": f"{i}@example.com"} for i in range(500)]).users) == 500
    assert BulkNewUserItem(user_email="a@example.com").auto_create_key is False
