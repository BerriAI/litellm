"""SCIM group pushes write the roster in bulk, whatever the member count.

POST /scim/v2/Groups, PUT and PATCH /scim/v2/Groups/{id} used to run /team/member_add once per member, so a
500-member push was thousands of statements and outlasted the edge in front of the proxy. One locked
transaction now resolves every member, writes the roster and the membership rows and tags the users, so a
push of 500 members runs as many statements as a push of 5. DELETE /scim/v2/Groups/{id} used to detach the
members one at a time and leave their membership rows and the team's keys behind; it now goes through
/team/delete's bulk path, so deleting 500 members costs as many statements as deleting 5 and leaves nothing.
"""

import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, string_value
from integration._support.database import read_rows
from integration._support.database_relay import StatementCountingRelay, statement_counting_relay
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.scim import (
    add_to_group,
    assert_gone,
    assert_landed,
    body,
    create_group,
    created_team,
    delete_group,
    member_ids,
    replace_group,
    seed_users,
    team_key,
)

SMALL_PUSH: Final = 5
LARGE_PUSH: Final = 500
# Background work on an idle proxy (the cron-job leader poll, spend flushes) lands a few statements inside a
# measured window; the per-member loop this guards against added about six statements per member.
NOISE_ALLOWANCE: Final = 100
UNFIXED_PUSH_SECONDS: Final = 120
SWAP_GROUP: Final = 200
SWAP_ROUNDS: Final = 10
OWNED_PROXY_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 2 * UNFIXED_PUSH_SECONDS)
_UTC_MICROSECONDS: Final = "YYYY-MM-DD HH24:MI:SS.US"


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


def _updated_at(user_id: str) -> str:
    rows: Final = read_rows(
        f"SELECT to_char(updated_at, '{_UTC_MICROSECONDS}') AS updated_at FROM \"LiteLLM_UserTable\" WHERE user_id = %s",
        (user_id,),
    )
    return string_value(rows[0]["updated_at"])


def _database_utc_now() -> str:
    rows: Final = read_rows(f"SELECT to_char(NOW() AT TIME ZONE 'UTC', '{_UTC_MICROSECONDS}') AS utc_now", ())
    return string_value(rows[0]["utc_now"])


@pytest.mark.timeout(OWNED_PROXY_TIMEOUT)
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
        small_a: Final = seed_users(scenario, SMALL_PUSH)
        small_b: Final = seed_users(scenario, SMALL_PUSH)
        large: Final = seed_users(scenario, LARGE_PUSH)

        created_small: Final = _measured(relay, lambda: create_group(candidate, small_a))
        small_team: Final = created_team(scenario, created_small.response)
        created_large: Final = _measured(relay, lambda: create_group(candidate, large))
        large_team: Final = created_team(scenario, created_large.response)
        assert member_ids(body(created_large.response)) == frozenset(large)
        assert_landed(candidate, small_team, small_a)
        assert_landed(candidate, large_team, large)
        detach_window_open: Final = _database_utc_now()

        replaced_small: Final = _measured(relay, lambda: replace_group(candidate, small_team, small_b))
        replaced_large: Final = _measured(relay, lambda: replace_group(candidate, large_team, small_a))
        assert replaced_small.response.status_code == 200, replaced_small.response.text
        assert replaced_large.response.status_code == 200, replaced_large.response.text
        assert member_ids(body(replaced_small.response)) == frozenset(small_b)
        assert_landed(candidate, small_team, small_b)
        assert_landed(candidate, large_team, small_a)
        assert detach_window_open <= _updated_at(large[0]) <= _database_utc_now()

        patched_small: Final = _measured(relay, lambda: add_to_group(candidate, small_team, small_a))
        patched_large: Final = _measured(relay, lambda: add_to_group(candidate, large_team, large))
        assert patched_small.response.status_code == 200, patched_small.response.text
        assert patched_large.response.status_code == 200, patched_large.response.text
        assert_landed(candidate, small_team, [*small_a, *small_b])
        assert_landed(candidate, large_team, [*small_a, *large])

        small_key: Final = team_key(candidate, small_team)
        large_key: Final = team_key(candidate, large_team)
        deleted_small: Final = _measured(relay, lambda: delete_group(candidate, small_team))
        deleted_large: Final = _measured(relay, lambda: delete_group(candidate, large_team))
        assert deleted_small.response.status_code == 204, deleted_small.response.text
        assert deleted_large.response.status_code == 204, deleted_large.response.text
        assert_gone(candidate, small_team, [*small_a, *small_b], small_key)
        assert_gone(candidate, large_team, [*small_a, *large], large_key)

        pushes: Final = (
            ("POST", created_small, created_large),
            ("PUT", replaced_small, replaced_large),
            ("PATCH", patched_small, patched_large),
            ("DELETE", deleted_small, deleted_large),
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
        left: Final = pool.submit(replace_group, candidate, teams[0], rosters[1])
        right: Final = pool.submit(replace_group, candidate, teams[1], rosters[0])
        for response in (left.result(), right.result()):
            assert response.status_code == 200, response.text


@pytest.mark.timeout(OWNED_PROXY_TIMEOUT)
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
        left_users: Final = seed_users(scenario, SWAP_GROUP)
        right_users: Final = seed_users(scenario, SWAP_GROUP)
        left: Final = created_team(scenario, create_group(candidate, left_users))
        right: Final = created_team(scenario, create_group(candidate, right_users))
        rounds: Final = tuple(
            (left_users, right_users) if round_index % 2 == 0 else (right_users, left_users)
            for round_index in range(SWAP_ROUNDS)
        )
        for held in rounds:
            _trade_rosters(candidate, (left, right), held)
        assert_landed(candidate, left, left_users)
        assert_landed(candidate, right, right_users)
