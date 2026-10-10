"""SCIM group pushes write the roster in bulk, whatever the member count.

POST /scim/v2/Groups, PUT and PATCH /scim/v2/Groups/{id} used to run /team/member_add once per member, so a
500-member push was thousands of statements and outlasted the edge in front of the proxy. One locked
transaction now resolves every member, writes the roster and the membership rows and tags the users, so a
push of 500 members runs as many statements as a push of 5.
"""

import os
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from integration._support.database import read_rows, write_rows
from integration._support.database_relay import StatementCountingRelay, statement_counting_relay
from integration._support.process import owned_proxy_process
from pydantic import JsonValue

GROUP_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:Group"
PATCH_OP_SCHEMA: Final = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
SMALL_PUSH: Final = 5
LARGE_PUSH: Final = 500
# Background work on an idle proxy (the cron-job leader poll, spend flushes) lands a few statements inside a
# measured window; the per-member loop this guards against added about six statements per member.
NOISE_ALLOWANCE: Final = 100
UNFIXED_PUSH_SECONDS: Final = 120
SWAP_GROUP: Final = 200
SWAP_ROUNDS: Final = 10
SEED_USERS_SQL: Final = """
INSERT INTO "LiteLLM_UserTable" (user_id, user_role, teams, models)
SELECT %s || '-' || lpad(n::text, 3, '0'), 'internal_user', '{}'::text[], '{}'::text[]
FROM generate_series(1, %s::int) AS n
"""


@dataclass(frozen=True, slots=True)
class Measured:
    response: httpx.Response
    statements: int
    seconds: float


def _measured(relay: StatementCountingRelay, send: Callable[[], httpx.Response]) -> Measured:
    before: Final = relay.statements
    started: Final = time.perf_counter()
    response: Final = send()
    return Measured(response=response, statements=relay.statements - before, seconds=time.perf_counter() - started)


def _seed_users(scenario: Scenario, count: int) -> tuple[str, ...]:
    prefix: Final = f"integration-scim-{uuid.uuid4().hex}"
    write_rows(SEED_USERS_SQL, (prefix, str(count)))
    scenario.cleanups.callback(write_rows, 'DELETE FROM "LiteLLM_UserTable" WHERE user_id LIKE %s', (f"{prefix}-%",))
    return tuple(f"{prefix}-{index:03d}" for index in range(1, count + 1))


def _members(users: Sequence[str]) -> list[JsonValue]:
    return [{"value": user} for user in users]


def _create_group(candidate: Gateway, users: Sequence[str]) -> httpx.Response:
    return candidate.request(
        "POST",
        "/scim/v2/Groups",
        {"schemas": [GROUP_SCHEMA], "displayName": f"integration-{uuid.uuid4().hex}", "members": _members(users)},
    )


def _replace_group(candidate: Gateway, team: str, users: Sequence[str]) -> httpx.Response:
    return candidate.request(
        "PUT",
        f"/scim/v2/Groups/{team}",
        {
            "schemas": [GROUP_SCHEMA],
            "id": team,
            "displayName": f"integration-{uuid.uuid4().hex}",
            "members": _members(users),
        },
    )


def _add_to_group(candidate: Gateway, team: str, users: Sequence[str]) -> httpx.Response:
    return candidate.request(
        "PATCH",
        f"/scim/v2/Groups/{team}",
        {
            "schemas": [PATCH_OP_SCHEMA],
            "Operations": [{"op": "add", "path": "members", "value": _members(users)}],
        },
    )


def _body(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def _member_ids(body: Mapping[str, JsonValue]) -> frozenset[str]:
    members: Final = body.get("members") or []
    assert isinstance(members, list), members
    return frozenset(string_value(object_value(member)["value"]) for member in members)


def _created_team(scenario: Scenario, created: httpx.Response) -> str:
    assert created.status_code == 201, created.text
    team: Final = string_value(_body(created)["id"])
    scenario.cleanups.callback(scenario.delete_team, team)
    return team


def _membership_user_ids(team: str) -> frozenset[str]:
    rows: Final = read_rows('SELECT user_id FROM "LiteLLM_TeamMembership" WHERE team_id = %s', (team,))
    return frozenset(string_value(row["user_id"]) for row in rows)


_UTC_MICROSECONDS: Final = "YYYY-MM-DD HH24:MI:SS.US"


def _updated_at(user_id: str) -> str:
    rows: Final = read_rows(
        f"SELECT to_char(updated_at, '{_UTC_MICROSECONDS}') AS updated_at FROM \"LiteLLM_UserTable\" WHERE user_id = %s",
        (user_id,),
    )
    return string_value(rows[0]["updated_at"])


def _database_utc_now() -> str:
    rows: Final = read_rows(f"SELECT to_char(NOW() AT TIME ZONE 'UTC', '{_UTC_MICROSECONDS}') AS utc_now", ())
    return string_value(rows[0]["utc_now"])


def _users_referencing(team: str) -> frozenset[str]:
    rows: Final = read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE %s = ANY(teams)', (team,))
    return frozenset(string_value(row["user_id"]) for row in rows)


def _assert_landed(candidate: Gateway, team: str, users: Sequence[str]) -> None:
    assert _member_ids(candidate.get(f"/scim/v2/Groups/{team}")) == frozenset(users)
    assert _membership_user_ids(team) == frozenset(users)
    assert _users_referencing(team) == frozenset(users)


@pytest.mark.timeout(300)  # owned proxy boot plus three 500-member pushes
def test_group_pushes_of_500_members_run_as_many_statements_as_pushes_of_5(
    gateway: Gateway, tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    with (
        gateway.scenario() as scenario,
        statement_counting_relay(os.environ["DATABASE_URL"]) as (relay, relayed_url),
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": relayed_url},
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
            client_timeout=UNFIXED_PUSH_SECONDS,
        ) as owned,
    ):
        candidate: Final = owned.gateway
        small_a: Final = _seed_users(scenario, SMALL_PUSH)
        small_b: Final = _seed_users(scenario, SMALL_PUSH)
        large: Final = _seed_users(scenario, LARGE_PUSH)

        created_small: Final = _measured(relay, lambda: _create_group(candidate, small_a))
        small_team: Final = _created_team(scenario, created_small.response)
        created_large: Final = _measured(relay, lambda: _create_group(candidate, large))
        large_team: Final = _created_team(scenario, created_large.response)
        assert _member_ids(_body(created_large.response)) == frozenset(large)
        _assert_landed(candidate, small_team, small_a)
        _assert_landed(candidate, large_team, large)
        detach_window_open: Final = _database_utc_now()

        replaced_small: Final = _measured(relay, lambda: _replace_group(candidate, small_team, small_b))
        replaced_large: Final = _measured(relay, lambda: _replace_group(candidate, large_team, small_a))
        assert replaced_small.response.status_code == 200, replaced_small.response.text
        assert replaced_large.response.status_code == 200, replaced_large.response.text
        assert _member_ids(_body(replaced_small.response)) == frozenset(small_b)
        _assert_landed(candidate, small_team, small_b)
        _assert_landed(candidate, large_team, small_a)
        assert detach_window_open <= _updated_at(large[0]) <= _database_utc_now()

        patched_small: Final = _measured(relay, lambda: _add_to_group(candidate, small_team, small_a))
        patched_large: Final = _measured(relay, lambda: _add_to_group(candidate, large_team, large))
        assert patched_small.response.status_code == 200, patched_small.response.text
        assert patched_large.response.status_code == 200, patched_large.response.text
        _assert_landed(candidate, small_team, [*small_a, *small_b])
        _assert_landed(candidate, large_team, [*small_a, *large])

        pushes: Final = (
            ("POST", created_small, created_large),
            ("PUT", replaced_small, replaced_large),
            ("PATCH", patched_small, patched_large),
        )
        for verb, small, big in pushes:
            record_property(
                f"{verb.lower()}_statements",
                f"{SMALL_PUSH} members: {small.statements} statements in {small.seconds:.2f}s; "
                f"{LARGE_PUSH} members: {big.statements} statements in {big.seconds:.2f}s",
            )
        for verb, small, big in pushes:
            assert min(small.statements, big.statements) > 0, (verb, small.statements, big.statements)
            assert big.statements <= small.statements + NOISE_ALLOWANCE, (verb, small.statements, big.statements)


def _trade_rosters(candidate: Gateway, teams: tuple[str, str], rosters: tuple[Sequence[str], Sequence[str]]) -> None:
    with ThreadPoolExecutor(max_workers=2) as pool:
        left: Final = pool.submit(_replace_group, candidate, teams[0], rosters[1])
        right: Final = pool.submit(_replace_group, candidate, teams[1], rosters[0])
        for response in (left.result(), right.result()):
            assert response.status_code == 200, response.text


@pytest.mark.timeout(300)
def test_two_groups_trading_rosters_in_parallel_both_land(gateway: Gateway, tmp_path: Path) -> None:
    with (
        gateway.scenario() as scenario,
        owned_proxy_process(
            gateway,
            tmp_path,
            {},
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
            client_timeout=UNFIXED_PUSH_SECONDS,
        ) as owned,
    ):
        candidate: Final = owned.gateway
        left_users: Final = _seed_users(scenario, SWAP_GROUP)
        right_users: Final = _seed_users(scenario, SWAP_GROUP)
        left: Final = _created_team(scenario, _create_group(candidate, left_users))
        right: Final = _created_team(scenario, _create_group(candidate, right_users))
        for round_index in range(SWAP_ROUNDS):
            held: Final = (left_users, right_users) if round_index % 2 == 0 else (right_users, left_users)
            _trade_rosters(candidate, (left, right), held)
        _assert_landed(candidate, left, left_users)
        _assert_landed(candidate, right, right_users)
