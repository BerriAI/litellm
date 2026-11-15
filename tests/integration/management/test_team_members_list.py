"""`GET /management/v1/teams/{team_id}/members`, `/team/info?include_members=false` and a
`/team/member_update` that no longer reads the roster, on a team with thousands of members.

`/team/info` loads, validates and serializes every member of a team on every call, and
`/team/member_update` called it only to find one member. On a roster in the tens of thousands that
is seconds of event-loop time per call, so readers that need one page, one member or no member at
all get a route that reads only that.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import (
    Gateway,
    Scenario,
    held,
    list_value,
    object_value,
    string_value,
)
from tests.integration._support.database import read_rows, write_rows

ROSTER: Final = 5000
WITH_MEMBERSHIP: Final = 4000
CUSTOM_BUDGETS: Final = 10
OWN_BUDGET_MEMBER: Final = 1232
DEFAULT_MEMBER_BUDGET: Final = 25.0
MAX_PAGE_SIZE: Final = 100
CONCURRENT_NUDGES: Final = 16
DEFAULT_PAGE_SIZE: Final = 50

SEED_ROSTER_SQL: Final = """
UPDATE "LiteLLM_TeamTable"
SET members_with_roles = (
    SELECT jsonb_agg(
        jsonb_strip_nulls(jsonb_build_object(
            'user_id', %s || '-' || lpad(n::text, 5, '0'),
            'user_email', CASE WHEN n %% 2 = 0 THEN %s || '.' || lpad((n * 37 %% 5003)::text, 5, '0') || '@roster.example' END,
            'role', CASE WHEN n %% 1000 = 0 THEN 'admin' ELSE 'user' END
        )) ORDER BY n
    ) || jsonb_build_array(jsonb_build_object('user_email', %s || '.nobody@roster.example', 'role', 'user'))
    FROM generate_series(1, %s::int) AS n
)
WHERE team_id = %s
"""
SEED_USERS_SQL: Final = """
INSERT INTO "LiteLLM_UserTable" (user_id, user_role, user_email, user_alias, teams, models)
SELECT %s || '-' || lpad(n::text, 5, '0'), 'internal_user',
       %s || '.' || lpad((n * 37 %% 5003)::text, 5, '0') || '@users.example',
       %s || ' alias ' || lpad((%s::int + 1 - n)::text, 5, '0'), ARRAY[%s]::text[], '{}'::text[]
FROM generate_series(1, %s::int) AS n
"""
SEED_BUDGETS_SQL: Final = """
INSERT INTO "LiteLLM_BudgetTable" (budget_id, max_budget, budget_duration, budget_reset_at, created_by, updated_by)
SELECT %s || '-budget-' || n::text, 100 + n, '30d', TIMESTAMP '2030-01-01' + (n || ' days')::interval, 'seed', 'seed'
FROM generate_series(0, %s::int - 1) AS n
UNION ALL
SELECT %s || '-budget-own', 100 + %s::int %% %s::int, '30d',
       TIMESTAMP '2030-01-01' + ((%s::int %% %s::int) || ' days')::interval, 'seed', 'seed'
"""
SEED_MEMBERSHIPS_SQL: Final = """
INSERT INTO "LiteLLM_TeamMembership" (user_id, team_id, spend, total_spend, budget_id)
SELECT %s || '-' || lpad(n::text, 5, '0'), %s, n * 0.5, CASE WHEN n %% 3 = 0 THEN 0 ELSE n * 0.5 + 1 END,
       CASE WHEN n = %s::int THEN %s || '-budget-own'
            WHEN n %% 4 = 0 THEN %s || '-budget-' || (n %% %s::int)::text
            WHEN n %% 4 = 1 THEN %s END
FROM generate_series(1, %s::int) AS n
"""
ROWS_READ_SQL: Final = """
SELECT coalesce(sum(seq_tup_read + coalesce(idx_tup_fetch, 0)), 0)::bigint AS rows_read
FROM pg_stat_user_tables WHERE relname = %s
"""


def _email_rank(n: int) -> int:
    return n * 37 % 5003


@dataclass(frozen=True, slots=True)
class LargeTeam:
    team_id: str
    prefix: str
    default_budget_id: str

    def user_id(self, n: int) -> str:
        return f"{self.prefix}-{n:05d}"

    def email(self, n: int) -> str:
        return f"{self.prefix}.{_email_rank(n):05d}@{'roster' if n % 2 == 0 else 'users'}.example"

    @property
    def path(self) -> str:
        return f"/management/v1/teams/{self.team_id}/members"


def _delete_seeded_rows(prefix: str) -> None:
    write_rows('DELETE FROM "LiteLLM_TeamMembership" WHERE user_id LIKE %s', (f"{prefix}-%",))
    write_rows('DELETE FROM "LiteLLM_UserTable" WHERE user_id LIKE %s', (f"{prefix}-%",))
    write_rows('DELETE FROM "LiteLLM_BudgetTable" WHERE budget_id LIKE %s', (f"{prefix}-budget-%",))


@pytest.fixture(scope="module")
def large_team(session_gateway: Gateway) -> Iterator[LargeTeam]:
    prefix: Final = f"roster{uuid.uuid4().hex[:12]}"
    with session_gateway.scenario() as scenario:
        team_id: Final = scenario.team(team_alias=f"{prefix} team", team_member_budget=DEFAULT_MEMBER_BUDGET)
        metadata: Final = object_value(
            object_value(session_gateway.get("/team/info", {"team_id": team_id})["team_info"])["metadata"]
        )
        default_budget_id: Final = string_value(metadata["team_member_budget_id"])
        scenario.cleanups.callback(_delete_seeded_rows, prefix)
        write_rows(SEED_USERS_SQL, (prefix, prefix, prefix, str(ROSTER), team_id, str(ROSTER)))
        own: Final = (str(OWN_BUDGET_MEMBER), str(CUSTOM_BUDGETS))
        write_rows(SEED_BUDGETS_SQL, (prefix, str(CUSTOM_BUDGETS), prefix, *own, *own))
        write_rows(
            SEED_MEMBERSHIPS_SQL,
            (
                prefix,
                team_id,
                str(OWN_BUDGET_MEMBER),
                prefix,
                prefix,
                str(CUSTOM_BUDGETS),
                default_budget_id,
                str(WITH_MEMBERSHIP),
            ),
        )
        write_rows(SEED_ROSTER_SQL, (prefix, prefix, prefix, str(ROSTER), team_id))
        write_rows('ANALYZE "LiteLLM_TeamMembership"', ())
        write_rows('ANALYZE "LiteLLM_UserTable"', ())
        yield LargeTeam(team_id=team_id, prefix=prefix, default_budget_id=default_budget_id)


def _page(
    gateway: Gateway, team: LargeTeam, params: Mapping[str, str], *, key: str | None = None
) -> Mapping[str, JsonValue]:
    response: Final = gateway.request("GET", team.path, params=params, key=key)
    assert response.status_code == 200, f"GET {team.path} {dict(params)}: {response.status_code} {response.text}"
    return object_value(response.json())


def _rows(page: Mapping[str, JsonValue]) -> tuple[Mapping[str, JsonValue], ...]:
    return tuple(object_value(row) for row in list_value(page["data"]))


def _user_ids(page: Mapping[str, JsonValue]) -> list[JsonValue]:
    return [row["user_id"] for row in _rows(page)]


def _every_page(gateway: Gateway, team: LargeTeam, params: Mapping[str, str]) -> tuple[Mapping[str, JsonValue], ...]:
    first: Final = _page(gateway, team, {**params, "page_size": str(MAX_PAGE_SIZE)})
    total_pages: Final = object_value(first["meta"])["total_pages"]
    assert isinstance(total_pages, int)
    rest: Final = tuple(
        _page(gateway, team, {**params, "page_size": str(MAX_PAGE_SIZE), "page": str(number)})
        for number in range(2, total_pages + 1)
    )
    return tuple(chain.from_iterable(_rows(page) for page in (first, *rest)))


def _team_info_rows(team_info: Mapping[str, JsonValue], team: LargeTeam) -> list[Mapping[str, JsonValue]]:
    memberships: Final = {
        object_value(row)["user_id"]: object_value(row) for row in list_value(team_info["team_memberships"])
    }
    without_membership: Final[Mapping[str, JsonValue]] = {
        "team_id": team.team_id,
        "spend": 0.0,
        "total_spend": 0.0,
        "budget_id": None,
        "litellm_budget_table": None,
        "budget_source": "team_default",
    }
    return [
        {**object_value(member), **memberships.get(object_value(member)["user_id"], without_membership)}
        for member in list_value(object_value(team_info["team_info"])["members_with_roles"])
    ]


def test_first_page_is_small_and_counts_the_whole_roster(gateway: Gateway, large_team: LargeTeam) -> None:
    response: Final = gateway.request("GET", large_team.path)
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    assert object_value(body["meta"]) == {
        "total_count": ROSTER + 1,
        "page": 1,
        "page_size": DEFAULT_PAGE_SIZE,
        "total_pages": -(-(ROSTER + 1) // DEFAULT_PAGE_SIZE),
    }
    assert _user_ids(body) == [large_team.user_id(n) for n in range(1, DEFAULT_PAGE_SIZE + 1)]
    whole_roster: Final = gateway.request("GET", "/team/info", params={"team_id": large_team.team_id})
    assert whole_roster.status_code == 200, whole_roster.text
    assert len(response.content) < 50_000, f"one page is {len(response.content)} bytes"
    assert len(response.content) * 20 < len(whole_roster.content), (
        f"one page is {len(response.content)} bytes, /team/info is {len(whole_roster.content)}"
    )


def test_page_size_is_capped(gateway: Gateway, large_team: LargeTeam) -> None:
    page: Final = _page(gateway, large_team, {"page_size": str(ROSTER)})
    assert len(_rows(page)) == MAX_PAGE_SIZE
    assert object_value(page["meta"])["page_size"] == MAX_PAGE_SIZE


def test_every_member_matches_team_info(gateway: Gateway, large_team: LargeTeam) -> None:
    team_info: Final = gateway.get("/team/info", {"team_id": large_team.team_id})
    expected: Final = _team_info_rows(team_info, large_team)
    assert len(expected) == ROSTER + 1
    assert [row["budget_source"] for row in expected].count("custom") == WITH_MEMBERSHIP // 4
    assert list(_every_page(gateway, large_team, {})) == expected


def test_search_reaches_members_past_the_first_page(gateway: Gateway, large_team: LargeTeam) -> None:
    by_id: Final = _page(gateway, large_team, {"q": f"{large_team.prefix}-0499".upper()})
    assert _user_ids(by_id) == [large_team.user_id(n) for n in range(4990, 5000)]
    by_roster_email: Final = _page(gateway, large_team, {"q": large_team.email(4998).upper()})
    assert _user_ids(by_roster_email) == [large_team.user_id(4998)]
    by_user_row_email: Final = _page(gateway, large_team, {"q": large_team.email(4999)})
    assert _user_ids(by_user_row_email) == [large_team.user_id(4999)]
    shadowed_by_roster_email: Final = _page(
        gateway, large_team, {"q": large_team.email(4998).replace("@roster", "@users")}
    )
    assert _user_ids(shadowed_by_roster_email) == []
    by_alias: Final = _page(gateway, large_team, {"q": f"{large_team.prefix} ALIAS 00002"})
    assert _user_ids(by_alias) == [large_team.user_id(ROSTER - 1)]
    without_user_id: Final = _page(gateway, large_team, {"q": f"{large_team.prefix}.nobody"})
    assert [(row["user_id"], row["user_email"]) for row in _rows(without_user_id)] == [
        (None, f"{large_team.prefix}.nobody@roster.example")
    ]
    literal_wildcard: Final = _page(gateway, large_team, {"q": f"{large_team.prefix}_0499"})
    assert object_value(literal_wildcard["meta"])["total_count"] == 0


def test_filters_reach_members_past_the_first_page(gateway: Gateway, large_team: LargeTeam) -> None:
    admins: Final = _page(gateway, large_team, {"filter[role][eq]": "admin"})
    assert _user_ids(admins) == [large_team.user_id(n) for n in range(1000, ROSTER + 1, 1000)]
    assert object_value(admins["meta"])["total_count"] == ROSTER // 1000
    either_role: Final = _page(gateway, large_team, {"filter[role][in]": "admin,user", "page_size": "1"})
    assert object_value(either_role["meta"])["total_count"] == ROSTER + 1
    named: Final = _page(
        gateway,
        large_team,
        {"filter[user_id][in]": f"{large_team.user_id(4321)},{large_team.user_id(7)},not-on-this-team"},
    )
    assert _user_ids(named) == [large_team.user_id(7), large_team.user_id(4321)]
    named_admin: Final = _page(
        gateway,
        large_team,
        {"filter[user_id][in]": f"{large_team.user_id(3000)},{large_team.user_id(7)}", "filter[role][eq]": "admin"},
    )
    assert _user_ids(named_admin) == [large_team.user_id(3000)]


def test_a_member_whose_user_id_holds_a_comma_is_found_by_the_eq_filter(gateway: Gateway, scenario: Scenario) -> None:
    comma_id: Final = f"cn=jane,ou=eng-{uuid.uuid4().hex[:8]}"
    scenario.user(user_id=comma_id, user_role="internal_user")
    team_id: Final = scenario.team(members_with_roles=[{"user_id": comma_id, "role": "user"}])
    path: Final = f"/management/v1/teams/{team_id}/members"

    by_eq: Final = gateway.request("GET", path, params={"filter[user_id][eq]": comma_id})
    assert by_eq.status_code == 200, by_eq.text
    assert [row["user_id"] for row in _rows(object_value(by_eq.json()))] == [comma_id]

    by_in: Final = gateway.request("GET", path, params={"filter[user_id][in]": comma_id})
    assert by_in.status_code == 200, by_in.text
    assert _rows(object_value(by_in.json())) == ()


def _effective_max_budget(n: int) -> float:
    return 100.0 + n % CUSTOM_BUDGETS if n <= WITH_MEMBERSHIP and n % 4 == 0 else DEFAULT_MEMBER_BUDGET


def _spend(n: int) -> float:
    return n * 0.5 if n <= WITH_MEMBERSHIP else 0.0


SORTED_MEMBERS: Final[Mapping[str, Sequence[int]]] = {
    "user_id": range(1, ROSTER + 1),
    "-user_id": range(ROSTER, 0, -1),
    "user_email": sorted(range(1, ROSTER + 1), key=_email_rank),
    "-user_email": (0, *sorted(range(1, ROSTER + 1), key=_email_rank, reverse=True)),
    "user_alias": range(ROSTER, 0, -1),
    "-user_alias": range(1, ROSTER + 1),
    "-role": (*(n for n in range(1, ROSTER + 1) if n % 1000), *range(1000, ROSTER + 1, 1000)),
    "role": (*range(1000, ROSTER + 1, 1000), *(n for n in range(1, ROSTER + 1) if n % 1000)),
    "-spend": (*range(WITH_MEMBERSHIP, 0, -1), *range(WITH_MEMBERSHIP + 1, ROSTER + 1)),
    "-total_spend": sorted(
        range(1, ROSTER + 1),
        key=lambda n: (-(_spend(n) if n % 3 == 0 else _spend(n) + 1) if n <= WITH_MEMBERSHIP else 0, n),
    ),
    "-max_budget": sorted(range(1, ROSTER + 1), key=lambda n: (-_effective_max_budget(n), n)),
    "max_budget": sorted(range(1, ROSTER + 1), key=lambda n: (_effective_max_budget(n), n)),
    "-budget_reset_at": sorted(
        (n for n in range(1, WITH_MEMBERSHIP + 1) if n % 4 == 0), key=lambda n: (-(n % CUSTOM_BUDGETS), n)
    ),
}


@pytest.mark.parametrize("sort", tuple(SORTED_MEMBERS))
def test_sort_orders_the_whole_roster_before_paging(sort: str, gateway: Gateway, large_team: LargeTeam) -> None:
    expected: Final = SORTED_MEMBERS[sort]
    first: Final = _page(gateway, large_team, {"sort": sort, "page_size": "7"})
    assert _user_ids(first) == [large_team.user_id(n) if n else None for n in expected[:7]]
    deep: Final = _page(gateway, large_team, {"sort": sort, "page_size": "7", "page": "43"})
    assert _user_ids(deep) == [large_team.user_id(n) if n else None for n in expected[42 * 7 : 43 * 7]]


def test_only_readers_of_team_info_can_list_members(
    gateway: Gateway, scenario: Scenario, large_team: LargeTeam
) -> None:
    outsider: Final = scenario.key(user_id=scenario.user(user_role="internal_user"))
    member: Final = scenario.key(user_id=large_team.user_id(4242))
    team_key: Final = scenario.key(team_id=large_team.team_id)
    other_team_key: Final = scenario.key(team_id=scenario.team())
    statuses: Final = {
        name: (
            gateway.request("GET", "/team/info", params={"team_id": large_team.team_id}, key=key).status_code,
            gateway.request("GET", large_team.path, params={"page_size": "1"}, key=key).status_code,
        )
        for name, key in (
            ("outsider", outsider),
            ("member", member),
            ("team key", team_key),
            ("other team's key", other_team_key),
        )
    }
    assert statuses == {
        "outsider": (403, 403),
        "member": (200, 200),
        "team key": (200, 200),
        "other team's key": (403, 403),
    }
    refused: Final = gateway.request("GET", large_team.path, key=outsider)
    assert refused.headers["content-type"] == "application/problem+json"
    assert object_value(refused.json())["status"] == 403
    assert "data" not in object_value(refused.json())


def test_unknown_team_is_a_404_problem(gateway: Gateway, large_team: LargeTeam) -> None:
    response: Final = gateway.request("GET", f"/management/v1/teams/{large_team.team_id}-missing/members")
    assert response.status_code == 404, response.text
    assert response.headers["content-type"] == "application/problem+json"
    assert object_value(response.json())["status"] == 404


def test_another_teams_members_never_appear(gateway: Gateway, scenario: Scenario, large_team: LargeTeam) -> None:
    small_team: Final = scenario.team()
    newcomer: Final = scenario.member(small_team)
    page: Final = gateway.request("GET", f"/management/v1/teams/{small_team}/members", params={"sort": "user_id"})
    assert page.status_code == 200, page.text
    listed: Final = [object_value(row)["user_id"] for row in list_value(object_value(page.json())["data"])]
    assert newcomer in listed and len(listed) == 2, f"/team/new adds its creator, so two members: {listed}"
    assert not [user_id for user_id in listed if str(user_id).startswith(large_team.prefix)]
    assert object_value(object_value(page.json())["meta"])["total_count"] == 2


def test_team_info_without_members_keeps_every_other_field(gateway: Gateway, large_team: LargeTeam) -> None:
    full: Final = gateway.get("/team/info", {"team_id": large_team.team_id})
    trimmed_response: Final = gateway.request(
        "GET", "/team/info", params={"team_id": large_team.team_id, "include_members": "false"}
    )
    assert trimmed_response.status_code == 200, trimmed_response.text
    trimmed: Final = object_value(trimmed_response.json())
    assert len(list_value(object_value(full["team_info"])["members_with_roles"])) == ROSTER + 1
    assert trimmed == {
        **full,
        "team_memberships": [],
        "team_info": {**object_value(full["team_info"]), "members_with_roles": []},
    }
    assert len(trimmed_response.content) < 20_000, f"{len(trimmed_response.content)} bytes without the roster"
    explicit: Final = gateway.get("/team/info", {"team_id": large_team.team_id, "include_members": "true"})
    assert explicit == full


def _rows_read(table: str) -> int:
    value: Final = read_rows(ROWS_READ_SQL, (table,))[0]["rows_read"]
    assert isinstance(value, int)
    return value


def test_member_update_reads_a_handful_of_rows_not_the_roster(gateway: Gateway, large_team: LargeTeam) -> None:
    target: Final = large_team.user_id(OWN_BUDGET_MEMBER)
    tables: Final = ("LiteLLM_TeamMembership", "LiteLLM_UserTable")

    def rows_read() -> tuple[int, ...]:
        # A backend flushes its counters when it next goes idle, so nudge every pooled connection before reading.
        with ThreadPoolExecutor(max_workers=CONCURRENT_NUDGES) as pool:
            tuple(
                pool.map(lambda _: gateway.get("/management/v1/budgets", {"page_size": "1"}), range(CONCURRENT_NUDGES))
            )
        return tuple(_rows_read(table) for table in tables)

    _, before = held(lambda: (rows_read(), rows_read()), lambda reads: reads[0] == reads[1], holding=2, seconds=60)
    updated: Final = gateway.post(
        "/team/member_update", {"team_id": large_team.team_id, "user_id": target, "max_budget_in_team": 77.0}
    )
    assert updated["user_id"] == target and updated["max_budget_in_team"] == 77.0

    def rows_read_since() -> tuple[int, ...]:
        return tuple(now - earlier for now, earlier in zip(rows_read(), before, strict=True))

    read: Final = held(rows_read_since, lambda counts: counts[0] > 0, holding=3, seconds=30)
    assert max(read) < ROSTER // 10, f"one member update read {dict(zip(tables, read, strict=True))} rows"
    row: Final = _rows(_page(gateway, large_team, {"filter[user_id][in]": target}))[0]
    assert object_value(row["litellm_budget_table"])["max_budget"] == 77.0
    assert row["budget_source"] == "custom"


def test_member_update_by_email_held_only_on_the_user_row(gateway: Gateway, large_team: LargeTeam) -> None:
    target: Final = large_team.user_id(4321)
    email: Final = large_team.email(4321)
    assert read_rows('SELECT user_email FROM "LiteLLM_UserTable" WHERE user_id = %s', (target,)) == [
        {"user_email": email}
    ]
    updated: Final = gateway.post(
        "/team/member_update", {"team_id": large_team.team_id, "user_email": email, "max_budget_in_team": 55.0}
    )
    assert (updated["user_id"], updated["user_email"]) == (target, email)
    budgets: Final = read_rows(
        'SELECT m.user_id, b.max_budget FROM "LiteLLM_TeamMembership" m '
        'JOIN "LiteLLM_BudgetTable" b ON b.budget_id = m.budget_id '
        "WHERE m.team_id = %s AND b.max_budget = 55.0",
        (large_team.team_id,),
    )
    assert budgets == [{"user_id": target, "max_budget": 55.0}]


def test_member_update_by_roster_email(gateway: Gateway, large_team: LargeTeam) -> None:
    target: Final = large_team.user_id(4320)
    updated: Final = gateway.post(
        "/team/member_update",
        {
            "team_id": large_team.team_id,
            "user_email": large_team.email(4320),
            "max_budget_in_team": 44.0,
        },
    )
    assert updated["user_id"] == target
    row: Final = _rows(_page(gateway, large_team, {"filter[user_id][in]": target}))[0]
    assert object_value(row["litellm_budget_table"])["max_budget"] == 44.0


def test_member_update_refuses_an_email_of_a_user_outside_the_team(
    gateway: Gateway, scenario: Scenario, large_team: LargeTeam
) -> None:
    outsider_email: Final = f"{large_team.prefix}-outsider@users.example"
    scenario.user(user_role="internal_user", user_email=outsider_email)
    for email in (outsider_email, f"{large_team.prefix}-unknown@users.example"):
        response = gateway.request(
            "POST",
            "/team/member_update",
            {"team_id": large_team.team_id, "user_email": email, "max_budget_in_team": 1.0},
        )
        assert response.status_code == 400, f"{email}: {response.status_code} {response.text}"
        assert "User id doesn't exist in team table" in response.text
    assert (
        read_rows(
            'SELECT m.user_id FROM "LiteLLM_TeamMembership" m JOIN "LiteLLM_BudgetTable" b ON b.budget_id = m.budget_id '
            "WHERE m.team_id = %s AND b.max_budget = 1.0",
            (large_team.team_id,),
        )
        == []
    )
