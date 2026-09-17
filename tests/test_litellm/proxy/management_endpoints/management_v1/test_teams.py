"""`POST /management/v1/teams/{team_id}/members/bulk_update`: the per-member limit writes and the
HTTP contract around them.

The in-memory Prisma here follows the one in
`tests/test_litellm/proxy/management_helpers/test_bulk_user_deletion.py`, extended with the budget
table and the membership/budget relation the bulk budget writer needs.
"""

import copy
from collections.abc import Mapping, Sequence
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Final

import pytest
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict, Field

from litellm.proxy._types import LiteLLM_TeamTable, LitellmUserRoles, Member, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    team_membership_auth_cache_key,
    team_membership_reservation_cache_key,
)
from litellm.proxy.list_api.common import ManagementProblem, problem_response, request_validation_problem
from litellm.proxy.management_endpoints.management_v1 import router
from litellm.proxy.management_endpoints.management_v1.common import MANAGEMENT_V1_PREFIX
from litellm.proxy.management_helpers.bulk_team_member_budgets import bulk_update_team_member_budgets
from litellm.types.proxy.management_endpoints.team_endpoints import (
    MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES,
    BulkTeamMemberBudgetUpdateRequest,
    TeamMemberBudgetUpdateResult,
)

ADMIN: Final = UserAPIKeyAuth(user_id="admin", user_role=LitellmUserRoles.PROXY_ADMIN, api_key="sk-admin")
OUTSIDER: Final = UserAPIKeyAuth(user_id="outsider", user_role=LitellmUserRoles.INTERNAL_USER)
TEAM_ID: Final = "t1"


class _BudgetRow(BaseModel):
    """A `LiteLLM_BudgetTable` row, carrying every column the merge patch reads or writes."""

    model_config = ConfigDict(extra="allow")

    budget_id: str
    max_budget: float | None = None
    soft_budget: float | None = None
    max_parallel_requests: int | None = None
    tpm_limit: int | None = None
    rpm_limit: int | None = None
    model_max_budget: Mapping[str, object] | None = None
    budget_duration: str | None = None
    budget_reset_at: datetime | None = None
    allowed_models: list[str] = Field(default_factory=list)
    created_by: str | None = None
    updated_by: str | None = None


class _MembershipRow(BaseModel):
    """A `LiteLLM_TeamMembership` row; `litellm_budget_table` is only filled on an `include` read."""

    model_config = ConfigDict(extra="allow")

    user_id: str
    team_id: str
    budget_id: str | None = None
    litellm_budget_table: _BudgetRow | None = None


def _wanted(where: Mapping[str, object], field: str) -> set[str] | None:
    clause: Final = where.get(field)
    if isinstance(clause, dict) and "in" in clause:
        return set(clause["in"])
    if isinstance(clause, str):
        return {clause}
    return None


def _matches(row: Mapping[str, object], where: Mapping[str, object]) -> bool:
    return all((wanted := _wanted(where, field)) is not None and row.get(field) in wanted for field in where)


class _BudgetTable:
    def __init__(self, budgets: Sequence[_BudgetRow]) -> None:
        self.rows: dict[str, _BudgetRow] = {b.budget_id: b for b in budgets}

    async def find_unique(self, where: Mapping[str, str]) -> _BudgetRow | None:
        return self.rows.get(where["budget_id"])

    async def update(self, where: Mapping[str, str], data: Mapping[str, object]) -> _BudgetRow:
        row: Final = self.rows[where["budget_id"]]
        updated: Final = row.model_copy(update=dict(data))
        self.rows[row.budget_id] = updated
        return updated

    async def create(self, data: Mapping[str, object], include: Mapping[str, bool] | None = None) -> _BudgetRow:
        budget_id: Final = f"new-budget-{len(self.rows) + 1}"
        row: Final = _BudgetRow.model_validate({**data, "budget_id": budget_id})
        self.rows[budget_id] = row
        return row


class _MembershipTable:
    def __init__(self, budgets: _BudgetTable, memberships: Sequence[_MembershipRow]) -> None:
        self._budgets = budgets
        self.rows: list[_MembershipRow] = list(memberships)

    def _index_of(self, user_id: str, team_id: str) -> int | None:
        return next(
            (i for i, r in enumerate(self.rows) if r.user_id == user_id and r.team_id == team_id),
            None,
        )

    async def find_many(
        self, where: Mapping[str, object], include: Mapping[str, bool] | None = None
    ) -> list[_MembershipRow]:
        matched: Final = [r for r in self.rows if _matches(r.model_dump(), where)]
        if not include:
            return matched
        return [
            r.model_copy(update={"litellm_budget_table": self._budgets.rows.get(r.budget_id or "")}) for r in matched
        ]

    async def update(self, where: Mapping[str, Mapping[str, str]], data: Mapping[str, object]) -> _MembershipRow:
        key: Final = where["user_id_team_id"]
        index: Final = self._index_of(key["user_id"], key["team_id"])
        assert index is not None, f"no membership row for {key}"
        relation: Final = data.get("litellm_budget_table")
        if isinstance(relation, dict) and relation.get("disconnect"):
            self.rows[index] = self.rows[index].model_copy(update={"budget_id": None})
        return self.rows[index]

    async def upsert(self, where: Mapping[str, Mapping[str, str]], data: Mapping[str, object]) -> _MembershipRow:
        key: Final = where["user_id_team_id"]
        budget_id: Final = data["update"]["litellm_budget_table"]["connect"]["budget_id"]
        index: Final = self._index_of(key["user_id"], key["team_id"])
        if index is None:
            self.rows.append(_MembershipRow(user_id=key["user_id"], team_id=key["team_id"], budget_id=budget_id))
            return self.rows[-1]
        self.rows[index] = self.rows[index].model_copy(update={"budget_id": budget_id})
        return self.rows[index]


class _TeamTable:
    def __init__(self, teams: Sequence[LiteLLM_TeamTable]) -> None:
        self.rows: dict[str, LiteLLM_TeamTable] = {t.team_id: t for t in teams}

    async def find_unique(self, where: Mapping[str, str]) -> LiteLLM_TeamTable | None:
        return self.rows.get(where["team_id"])


class _Db:
    def __init__(
        self,
        teams: Sequence[LiteLLM_TeamTable],
        memberships: Sequence[_MembershipRow],
        budgets: Sequence[_BudgetRow],
    ) -> None:
        self.litellm_teamtable = _TeamTable(teams)
        self.litellm_budgettable = _BudgetTable(budgets)
        self.litellm_teammembership = _MembershipTable(self.litellm_budgettable, memberships)


class _FakePrisma:
    def __init__(
        self,
        teams: Sequence[LiteLLM_TeamTable] = (),
        memberships: Sequence[_MembershipRow] = (),
        budgets: Sequence[_BudgetRow] = (),
    ) -> None:
        self.db = _Db(teams, memberships, budgets)

    @asynccontextmanager
    async def tx(self, *, timeout: object = None):
        snapshot: Final = copy.deepcopy(self.db)
        try:
            yield self.db
        except BaseException:
            self.db = snapshot
            raise


def _team(
    *members: str,
    team_id: str = TEAM_ID,
    default_budget_id: str | None = None,
    admins: Sequence[str] = (),
) -> LiteLLM_TeamTable:
    return LiteLLM_TeamTable(
        team_id=team_id,
        metadata={"team_member_budget_id": default_budget_id} if default_budget_id else {},
        members_with_roles=[
            Member(user_id=m, user_email=f"{m}@example.com", role="admin" if m in admins else "user") for m in members
        ],
    )


def _membership(user_id: str, budget_id: str | None = None, team_id: str = TEAM_ID) -> _MembershipRow:
    return _MembershipRow(user_id=user_id, team_id=team_id, budget_id=budget_id)


def _budget(
    budget_id: str,
    *,
    max_budget: float | None = None,
    tpm_limit: int | None = None,
    rpm_limit: int | None = None,
    budget_duration: str | None = None,
) -> _BudgetRow:
    return _BudgetRow(
        budget_id=budget_id,
        max_budget=max_budget,
        tpm_limit=tpm_limit,
        rpm_limit=rpm_limit,
        budget_duration=budget_duration,
    )


async def _bulk_update(
    prisma: _FakePrisma,
    members: Sequence[Mapping[str, object]],
    team_id: str = TEAM_ID,
    caller: UserAPIKeyAuth = ADMIN,
    cache: UserApiKeyCache | None = None,
) -> tuple[TeamMemberBudgetUpdateResult, ...]:
    return await bulk_update_team_member_budgets(
        team_id=team_id,
        data=BulkTeamMemberBudgetUpdateRequest.model_validate({"members": list(members)}),
        user_api_key_dict=caller,
        prisma_client=prisma,  # pyright: ignore[reportArgumentType]  # fake stands in for PrismaClient
        user_api_key_cache=cache or UserApiKeyCache(),
    )


def _budget_id_of(prisma: _FakePrisma, user_id: str, team_id: str = TEAM_ID) -> str | None:
    row: Final = next(r for r in prisma.db.litellm_teammembership.rows if r.user_id == user_id and r.team_id == team_id)
    return row.budget_id


def _budget_of(prisma: _FakePrisma, user_id: str, team_id: str = TEAM_ID) -> _BudgetRow:
    budget_id: Final = _budget_id_of(prisma, user_id, team_id)
    assert budget_id is not None, f"{user_id} has no budget"
    return prisma.db.litellm_budgettable.rows[budget_id]


def _seeded_cache(*user_ids: str, team_id: str = TEAM_ID) -> UserApiKeyCache:
    cache: Final = UserApiKeyCache()
    for user_id in user_ids:
        cache.set_cache(key=team_membership_auth_cache_key(team_id=team_id, user_id=user_id), value={"cap": "old"})
        cache.set_cache(
            key=team_membership_reservation_cache_key(user_id=user_id, team_id=team_id), value={"cap": "old"}
        )
    return cache


def _cached_keys(cache: UserApiKeyCache, user_id: str, team_id: str = TEAM_ID) -> tuple[object, object]:
    return (
        cache.get_cache(key=team_membership_auth_cache_key(team_id=team_id, user_id=user_id)),
        cache.get_cache(key=team_membership_reservation_cache_key(user_id=user_id, team_id=team_id)),
    )


@pytest.mark.asyncio
async def test_patching_one_member_of_a_shared_budget_row_forks_it_and_leaves_the_other_member_untouched():
    prisma = _FakePrisma(
        teams=[_team("m1", "m2")],
        memberships=[_membership("m1", "shared-b"), _membership("m2", "shared-b")],
        budgets=[_budget("shared-b", max_budget=100.0, tpm_limit=900)],
    )

    results = await _bulk_update(prisma, [{"user_id": "m1", "max_budget_in_team": 50}])

    assert [(r.user_id, r.success, r.max_budget) for r in results] == [("m1", True, 50.0)]
    assert _budget_id_of(prisma, "m1") not in (None, "shared-b")
    assert (_budget_of(prisma, "m1").max_budget, _budget_of(prisma, "m1").tpm_limit) == (50.0, 900)
    assert _budget_id_of(prisma, "m2") == "shared-b"
    assert prisma.db.litellm_budgettable.rows["shared-b"].max_budget == 100.0
    assert results[0].budget_id == _budget_id_of(prisma, "m1")


@pytest.mark.asyncio
async def test_patching_members_of_the_team_default_budget_gives_each_their_own_row_and_leaves_the_default_alone():
    prisma = _FakePrisma(
        teams=[_team("m1", "m2", "m3", default_budget_id="team-default")],
        memberships=[
            _membership("m1", "team-default"),
            _membership("m2", "team-default"),
            _membership("m3", "team-default"),
        ],
        budgets=[_budget("team-default", max_budget=25.0, tpm_limit=1000)],
    )

    results = await _bulk_update(
        prisma,
        [{"user_id": "m1", "max_budget_in_team": 5}, {"user_id": "m2", "max_budget_in_team": 7}],
    )

    assert [r.success for r in results] == [True, True]
    default = prisma.db.litellm_budgettable.rows["team-default"]
    assert (default.max_budget, default.tpm_limit) == (25.0, 1000)
    assert _budget_id_of(prisma, "m3") == "team-default"
    patched = (_budget_id_of(prisma, "m1"), _budget_id_of(prisma, "m2"))
    assert len(set(patched)) == 2 and "team-default" not in patched
    assert (_budget_of(prisma, "m1").max_budget, _budget_of(prisma, "m1").tpm_limit) == (5.0, 1000)
    assert (_budget_of(prisma, "m2").max_budget, _budget_of(prisma, "m2").tpm_limit) == (7.0, 1000)


@pytest.mark.asyncio
async def test_the_team_default_row_is_forked_even_when_only_one_membership_points_at_it():
    prisma = _FakePrisma(
        teams=[_team("m1", "m2", default_budget_id="team-default")],
        memberships=[_membership("m1", "team-default")],
        budgets=[_budget("team-default", max_budget=25.0, tpm_limit=1000)],
    )

    results = await _bulk_update(prisma, [{"user_id": "m1", "max_budget_in_team": 5}])

    assert [(r.success, r.max_budget, r.tpm_limit) for r in results] == [(True, 5.0, 1000)]
    default = prisma.db.litellm_budgettable.rows["team-default"]
    assert (default.max_budget, default.tpm_limit) == (25.0, 1000)
    assert _budget_id_of(prisma, "m1") not in (None, "team-default")


@pytest.mark.asyncio
async def test_a_budget_row_only_one_member_points_at_is_updated_in_place():
    prisma = _FakePrisma(
        teams=[_team("m1", "m2", default_budget_id="team-default")],
        memberships=[_membership("m1", "priv-m1"), _membership("m2", "team-default")],
        budgets=[_budget("team-default", max_budget=25.0), _budget("priv-m1", max_budget=10.0, tpm_limit=5)],
    )

    results = await _bulk_update(prisma, [{"user_id": "m1", "max_budget_in_team": 20}])

    assert [(r.success, r.budget_id, r.max_budget) for r in results] == [(True, "priv-m1", 20.0)]
    assert set(prisma.db.litellm_budgettable.rows) == {"team-default", "priv-m1"}
    assert _budget_id_of(prisma, "m1") == "priv-m1"
    assert (_budget_of(prisma, "m1").max_budget, _budget_of(prisma, "m1").tpm_limit) == (20.0, 5)


@pytest.mark.asyncio
async def test_an_omitted_field_is_kept_an_explicit_null_clears_it_and_clearing_the_last_limit_disconnects():
    prisma = _FakePrisma(
        teams=[_team("m1")],
        memberships=[_membership("m1", "priv-m1")],
        budgets=[_budget("priv-m1", max_budget=10.0, tpm_limit=5, rpm_limit=7)],
    )

    kept = await _bulk_update(prisma, [{"user_id": "m1", "rpm_limit": 9}])

    assert (kept[0].max_budget, kept[0].tpm_limit, kept[0].rpm_limit) == (10.0, 5, 9)

    cleared = await _bulk_update(prisma, [{"user_id": "m1", "tpm_limit": None}])

    assert (cleared[0].max_budget, cleared[0].tpm_limit, cleared[0].rpm_limit) == (10.0, None, 9)
    assert _budget_id_of(prisma, "m1") == "priv-m1"

    emptied = await _bulk_update(prisma, [{"user_id": "m1", "max_budget_in_team": None, "rpm_limit": None}])

    assert (emptied[0].success, emptied[0].budget_id, emptied[0].max_budget) == (True, None, None)
    assert _budget_id_of(prisma, "m1") is None


@pytest.mark.asyncio
async def test_budget_duration_seeds_a_reset_time_derived_from_the_duration_and_clearing_it_clears_the_reset():
    prisma = _FakePrisma(
        teams=[_team("m1", "m2")],
        memberships=[_membership("m1", "priv-m1"), _membership("m2", "priv-m2")],
        budgets=[_budget("priv-m1", max_budget=10.0), _budget("priv-m2", max_budget=10.0)],
    )
    before = datetime.now(timezone.utc)

    await _bulk_update(
        prisma,
        [{"user_id": "m1", "budget_duration": "2d"}, {"user_id": "m2", "budget_duration": "5d"}],
    )

    two_day = _budget_of(prisma, "m1").budget_reset_at
    five_day = _budget_of(prisma, "m2").budget_reset_at
    assert two_day is not None and five_day is not None
    assert before < two_day <= before + timedelta(days=2)
    assert before + timedelta(days=4) - timedelta(seconds=1) < five_day <= before + timedelta(days=5)
    assert five_day - two_day == timedelta(days=3)

    await _bulk_update(prisma, [{"user_id": "m1", "budget_duration": None}])

    assert _budget_of(prisma, "m1").budget_reset_at is None
    assert _budget_of(prisma, "m1").budget_duration is None
    assert _budget_of(prisma, "m1").max_budget == 10.0


@pytest.mark.asyncio
async def test_a_member_named_twice_is_written_once_and_the_later_rows_report_the_duplicate():
    prisma = _FakePrisma(
        teams=[_team("m1")],
        memberships=[_membership("m1", "priv-m1")],
        budgets=[_budget("priv-m1", max_budget=1.0)],
    )

    results = await _bulk_update(
        prisma,
        [
            {"user_id": "m1", "max_budget_in_team": 10},
            {"user_id": "m1", "max_budget_in_team": 20},
            {"user_email": "m1@example.com", "max_budget_in_team": 30},
        ],
    )

    assert [(r.success, r.error) for r in results] == [
        (True, None),
        (False, "Duplicate member in request"),
        (False, "Duplicate member in request"),
    ]
    assert _budget_of(prisma, "m1").max_budget == 10.0


@pytest.mark.asyncio
async def test_a_row_naming_somebody_off_the_team_fails_without_writing_while_the_rest_of_the_batch_lands():
    prisma = _FakePrisma(
        teams=[_team("m1")],
        memberships=[_membership("m1", "priv-m1"), _membership("elsewhere", "priv-other")],
        budgets=[_budget("priv-m1", max_budget=1.0), _budget("priv-other", max_budget=2.0)],
    )

    results = await _bulk_update(
        prisma,
        [
            {"user_id": "elsewhere", "max_budget_in_team": 99},
            {"user_email": "nobody@example.com", "max_budget_in_team": 99},
            {"user_id": "m1", "max_budget_in_team": 10},
        ],
    )

    assert [(r.success, r.error) for r in results] == [
        (False, "User not found in team"),
        (False, "User not found in team"),
        (True, None),
    ]
    assert prisma.db.litellm_budgettable.rows["priv-other"].max_budget == 2.0
    assert _budget_of(prisma, "m1").max_budget == 10.0
    assert set(prisma.db.litellm_budgettable.rows) == {"priv-m1", "priv-other"}


@pytest.mark.asyncio
async def test_each_result_carries_the_limits_read_back_after_the_write_in_request_order():
    prisma = _FakePrisma(
        teams=[_team("m1", "m2")],
        memberships=[_membership("m1", "priv-m1"), _membership("m2", "priv-m2")],
        budgets=[
            _budget("priv-m1", tpm_limit=100, budget_duration="7d"),
            _budget("priv-m2", rpm_limit=3),
        ],
    )

    results = await _bulk_update(
        prisma,
        [{"user_id": "m2", "rpm_limit": 8}, {"user_id": "m1", "max_budget_in_team": 42}],
    )

    assert [r.user_id for r in results] == ["m2", "m1"]
    assert (results[1].max_budget, results[1].tpm_limit, results[1].budget_duration) == (42.0, 100, "7d")
    assert (results[0].rpm_limit, results[0].max_budget) == (8, None)


@pytest.mark.asyncio
async def test_every_written_member_is_evicted_from_both_team_membership_cache_keys():
    prisma = _FakePrisma(
        teams=[_team("m1", "m2", "m3")],
        memberships=[_membership("m1", "priv-m1"), _membership("m2", "priv-m2"), _membership("m3", "priv-m3")],
        budgets=[_budget("priv-m1", max_budget=1.0), _budget("priv-m2", max_budget=2.0), _budget("priv-m3")],
    )
    cache = _seeded_cache("m1", "m2", "m3")

    await _bulk_update(
        prisma,
        [{"user_id": "m1", "max_budget_in_team": 10}, {"user_id": "m2", "max_budget_in_team": 20}],
        cache=cache,
    )

    assert _cached_keys(cache, "m1") == (None, None)
    assert _cached_keys(cache, "m2") == (None, None)
    assert _cached_keys(cache, "m3") == ({"cap": "old"}, {"cap": "old"})


@pytest.mark.asyncio
async def test_a_member_with_no_cap_of_their_own_reports_the_team_default_cap_but_only_their_own_rate_limits():
    prisma = _FakePrisma(
        teams=[_team("m1", default_budget_id="team-default")],
        memberships=[],
        budgets=[_budget("team-default", max_budget=25.0, tpm_limit=1000)],
    )

    results = await _bulk_update(prisma, [{"user_id": "m1", "tpm_limit": 7}])

    assert [(r.success, r.max_budget, r.max_budget_source, r.tpm_limit) for r in results] == [
        (True, 25.0, "team_default", 7)
    ]
    assert _budget_of(prisma, "m1").max_budget is None
    default = prisma.db.litellm_budgettable.rows["team-default"]
    assert (default.max_budget, default.tpm_limit) == (25.0, 1000)


@pytest.mark.asyncio
async def test_an_explicit_cap_reports_as_the_members_own_while_clearing_one_falls_back_to_the_team_default():
    prisma = _FakePrisma(
        teams=[_team("m1", "m2", default_budget_id="team-default")],
        memberships=[_membership("m1", "priv-m1"), _membership("m2", "priv-m2")],
        budgets=[
            _budget("team-default", max_budget=25.0),
            _budget("priv-m1", max_budget=5.0),
            _budget("priv-m2", max_budget=9.0),
        ],
    )

    results = await _bulk_update(
        prisma,
        [{"user_id": "m1", "max_budget_in_team": 50}, {"user_id": "m2", "max_budget_in_team": None}],
    )

    assert [(r.user_id, r.max_budget, r.max_budget_source) for r in results] == [
        ("m1", 50.0, "member"),
        ("m2", 25.0, "team_default"),
    ]
    assert results[1].budget_id is None
    assert _budget_id_of(prisma, "m2") is None
    assert prisma.db.litellm_budgettable.rows["team-default"].max_budget == 25.0


@pytest.mark.asyncio
async def test_a_team_with_no_default_budget_reports_no_effective_cap_for_a_member_without_one():
    prisma = _FakePrisma(
        teams=[_team("m1")],
        memberships=[_membership("m1", "priv-m1")],
        budgets=[_budget("priv-m1", tpm_limit=5)],
    )

    results = await _bulk_update(prisma, [{"user_id": "m1", "rpm_limit": 3}])

    assert [(r.success, r.max_budget, r.max_budget_source) for r in results] == [(True, None, None)]
    assert (results[0].tpm_limit, results[0].rpm_limit) == (5, 3)


@pytest.mark.asyncio
async def test_a_row_that_names_nobody_on_the_team_reports_no_cap_and_no_source():
    prisma = _FakePrisma(
        teams=[_team("m1", default_budget_id="team-default")],
        memberships=[_membership("m1", "priv-m1")],
        budgets=[_budget("team-default", max_budget=25.0), _budget("priv-m1", max_budget=5.0)],
    )

    results = await _bulk_update(
        prisma,
        [{"user_id": "ghost", "max_budget_in_team": 1}, {"user_id": "m1", "max_budget_in_team": 6}],
    )

    assert [(r.success, r.max_budget, r.max_budget_source) for r in results] == [
        (False, None, None),
        (True, 6.0, "member"),
    ]


app = FastAPI()


@app.exception_handler(ManagementProblem)
async def management_problem_exception_handler(request: Request, exc: ManagementProblem):
    return problem_response(exc.problem)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return problem_response(request_validation_problem(exc.errors()))


app.include_router(router)
client = TestClient(app)

BULK_UPDATE_PATH: Final = f"{MANAGEMENT_V1_PREFIX}/teams/{TEAM_ID}/members/bulk_update"


@pytest.fixture
def as_proxy_admin():
    app.dependency_overrides[user_api_key_auth] = lambda: ADMIN
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def as_outsider():
    app.dependency_overrides[user_api_key_auth] = lambda: OUTSIDER
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def prisma(monkeypatch):
    fake = _FakePrisma(
        teams=[_team("m1", "m2")],
        memberships=[_membership("m1", "priv-m1")],
        budgets=[_budget("priv-m1", max_budget=1.0)],
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.prisma_client", fake)
    return fake


def _post(body: object, path: str = BULK_UPDATE_PATH):
    return client.post(path, json=body, headers={"Authorization": "Bearer sk-1234"})


def test_unknown_fields_empty_and_oversized_batches_are_422_problem_documents(prisma, as_proxy_admin):
    bodies = (
        {"members": [{"user_id": "m1", "max_budget": 10}]},
        {"members": [{"user_id": "m1"}], "team_id": TEAM_ID},
        {"members": []},
        {"members": [{"user_id": f"u{i}"} for i in range(MAX_BULK_TEAM_MEMBER_BUDGET_UPDATES + 1)]},
    )

    for body in bodies:
        response = _post(body)

        assert response.status_code == 422, body
        assert response.headers["content-type"] == "application/problem+json"
        assert response.json()["type"] == "urn:litellm:error:invalid-request-body"
    assert prisma.db.litellm_budgettable.rows["priv-m1"].max_budget == 1.0


def test_an_unknown_team_is_a_404_problem_document(prisma, as_proxy_admin):
    response = _post(
        {"members": [{"user_id": "m1", "max_budget_in_team": 10}]},
        path=f"{MANAGEMENT_V1_PREFIX}/teams/nope/members/bulk_update",
    )

    assert response.status_code == 404
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:litellm:error:team-not-found"
    assert prisma.db.litellm_budgettable.rows["priv-m1"].max_budget == 1.0


def test_a_caller_who_administers_neither_the_team_nor_its_org_is_a_403_problem_document(prisma, as_outsider):
    response = _post({"members": [{"user_id": "m1", "max_budget_in_team": 10}]})

    assert response.status_code == 403
    assert response.headers["content-type"] == "application/problem+json"
    assert response.json()["type"] == "urn:litellm:error:forbidden"
    assert prisma.db.litellm_budgettable.rows["priv-m1"].max_budget == 1.0


def test_a_team_admin_may_bulk_update_their_own_teams_members(prisma, monkeypatch):
    prisma.db.litellm_teamtable.rows[TEAM_ID] = _team("lead", "m1", admins=("lead",))
    app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="lead", user_role=LitellmUserRoles.INTERNAL_USER
    )
    try:
        response = _post({"members": [{"user_id": "m1", "max_budget_in_team": 10}]})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert [(r["user_id"], r["success"], r["max_budget"]) for r in response.json()["data"]] == [("m1", True, 10.0)]
