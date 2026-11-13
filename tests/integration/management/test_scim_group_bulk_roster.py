"""SCIM group pushes write the roster in bulk, whatever the member count.

POST /scim/v2/Groups, PUT and PATCH /scim/v2/Groups/{id} used to run /team/member_add once per member, so a
500-member push was thousands of statements and outlasted the edge in front of the proxy. One locked
transaction now resolves every member, writes the roster and the membership rows and tags the users, so a
push of 500 members runs as many statements as a push of 5. DELETE /scim/v2/Groups/{id} used to detach the
members one at a time and leave their membership rows and keys behind; it now detaches them in one locked pass,
so deleting 500 members who each hold a key costs as many statements as deleting 5 and leaves no membership rows
or member keys.
"""

import os
import time
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Final, LiteralString

import httpx
import psycopg
import pytest
from integration._support.budget import case_timeout_seconds
from integration._support.client import (
    Gateway,
    Scenario,
    delete_key_if_present,
    eventually,
    object_value,
    string_value,
)
from integration._support.database import read_rows, write_rows
from integration._support.database_relay import StatementCountingRelay, statement_counting_relay
from integration._support.process import graceful_stop_seconds, owned_proxy_process
from integration._support.scim import (
    GROUP_SCHEMA,
    PATCH_OP_SCHEMA,
    add_to_group,
    assert_gone,
    assert_landed,
    body,
    create_group,
    created_team,
    delete_group,
    member_ids,
    members,
    membership_user_ids,
    replace_group,
    seed_users,
    team_key,
    users_referencing,
)
from psycopg import sql
from pydantic import JsonValue

pytestmark: Final = pytest.mark.shares_worker_state

SMALL_PUSH: Final = 5
LARGE_PUSH: Final = 500
# Background work on an idle proxy (the cron-job leader poll, spend flushes) lands a few statements inside a
# measured window; the per-member loop this guards against added about six statements per member.
NOISE_ALLOWANCE: Final = 100
UNFIXED_PUSH_SECONDS: Final = 120
SWAP_GROUP: Final = 200
SWAP_ROUNDS: Final = 10
OWNED_PROXY_TIMEOUT: Final = int(2 * graceful_stop_seconds() + 2 * UNFIXED_PUSH_SECONDS)
CHAOS_GROUP: Final = 100
KEY_WRITERS: Final = 16
ROLLBACK_SECONDS: Final = 90
TEAM_ROW_SQL: Final = 'SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s'
LOCK_USER_ROW_SQL: Final = 'SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s FOR UPDATE'
LOCK_KEY_ROW_SQL: Final = 'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s FOR UPDATE'
SESSIONS_BLOCKED_BY_SQL: Final = (
    "SELECT count(*)::int AS waiting FROM pg_stat_activity WHERE %s::int = ANY(pg_blocking_pids(pid))"
)
BACKENDS_BLOCKED_BY_SQL: Final = "SELECT pid FROM pg_stat_activity WHERE %s::int = ANY(pg_blocking_pids(pid))"
TRANSACTIONS_OPEN_SINCE_SQL: Final = """
SELECT count(*)::int AS open FROM pg_stat_activity
WHERE datname = current_database() AND backend_type = 'client backend' AND pid <> pg_backend_pid()
  AND xact_start < %s::timestamptz
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

        with ThreadPoolExecutor(max_workers=KEY_WRITERS) as pool:
            small_keys: Final = tuple(pool.map(partial(team_key, candidate, small_team), (*small_a, *small_b)))
            large_keys: Final = tuple(pool.map(partial(team_key, candidate, large_team), (*small_a, *large)))
        deleted_small: Final = _measured(relay, lambda: delete_group(candidate, small_team))
        deleted_large: Final = _measured(relay, lambda: delete_group(candidate, large_team))
        assert deleted_small.response.status_code == 204, deleted_small.response.text
        assert deleted_large.response.status_code == 204, deleted_large.response.text
        assert_gone(candidate, small_team, [*small_a, *small_b], small_keys[0])
        assert_gone(candidate, large_team, [*small_a, *large], large_keys[-1])

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


def _int_value(rows: Sequence[Mapping[str, JsonValue]], column: str) -> int:
    value: Final = rows[0][column]
    assert isinstance(value, int), rows
    return value


@contextmanager
def _holding_row_lock(lock_sql: LiteralString, row_id: str) -> Iterator[int]:
    with psycopg.connect(os.environ["DATABASE_URL"]) as holder:
        _ = holder.execute(lock_sql, (row_id,))
        yield holder.info.backend_pid


def _sessions_blocked_by(backend_pid: int) -> int:
    return _int_value(read_rows(SESSIONS_BLOCKED_BY_SQL, (str(backend_pid),)), "waiting")


def _backend_blocked_by(backend_pid: int) -> int:
    return _int_value(read_rows(BACKENDS_BLOCKED_BY_SQL, (str(backend_pid),)), "pid")


def _database_now() -> str:
    return string_value(read_rows("SELECT now()::text AS now", ())[0]["now"])


def _transactions_open_since(moment: str) -> int:
    return _int_value(read_rows(TRANSACTIONS_OPEN_SINCE_SQL, (moment,)), "open")


def _roster_ids(team: str) -> tuple[str, ...]:
    rows: Final = read_rows(TEAM_ROW_SQL, (team,))
    assert len(rows) == 1, rows
    roster: Final = rows[0]["members_with_roles"]
    assert isinstance(roster, list), roster
    return tuple(string_value(object_value(member)["user_id"]) for member in roster)


def test_a_group_deleted_while_its_push_waits_on_a_member_row_waits_for_the_push_then_removes_every_trace(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario, ThreadPoolExecutor(max_workers=2) as pool:
        users: Final = seed_users(scenario, SMALL_PUSH)
        team: Final = created_team(scenario, create_group(gateway, ()))
        with _holding_row_lock(LOCK_USER_ROW_SQL, min(users)) as holder_pid:
            push: Final = pool.submit(replace_group, gateway, team, users)
            _ = eventually(lambda: _sessions_blocked_by(holder_pid), lambda waiting: waiting >= 1, seconds=30)
            push_pid: Final = _backend_blocked_by(holder_pid)
            deleted: Final = pool.submit(delete_group, gateway, team)
            _ = eventually(lambda: _sessions_blocked_by(push_pid), lambda waiting: waiting >= 1, seconds=30)
            assert not push.done(), push.result().text
            assert not deleted.done(), deleted.result().text
        assert push.result().status_code == 200, push.result().text
        assert deleted.result().status_code == 204, deleted.result().text
        assert read_rows(TEAM_ROW_SQL, (team,)) == []
        assert membership_user_ids(team) == frozenset()
        assert users_referencing(team) == frozenset()


@contextmanager
def _refusing_memberships_for(user_id: str) -> Iterator[None]:
    name: Final = sql.Identifier(f"integration_refuse_membership_{uuid.uuid4().hex}")
    with psycopg.connect(os.environ["DATABASE_URL"], autocommit=True) as connection:
        _ = connection.execute(
            sql.SQL(
                "CREATE FUNCTION {}() RETURNS trigger LANGUAGE plpgsql AS "
                "$$ BEGIN RAISE EXCEPTION 'integration refused this membership write'; END $$"
            ).format(name)
        )
        _ = connection.execute(
            sql.SQL(
                'CREATE TRIGGER {} BEFORE INSERT ON "LiteLLM_TeamMembership" FOR EACH ROW '
                "WHEN (NEW.user_id = {}) EXECUTE FUNCTION {}()"
            ).format(name, sql.Literal(user_id), name)
        )
        try:
            yield
        finally:
            _ = connection.execute(sql.SQL('DROP TRIGGER {} ON "LiteLLM_TeamMembership"').format(name))
            _ = connection.execute(sql.SQL("DROP FUNCTION {}()").format(name))


def test_a_push_whose_membership_write_fails_answers_5xx_keeps_the_old_roster_and_a_retry_lands(
    gateway: Gateway,
) -> None:
    with gateway.scenario() as scenario:
        kept: Final = seed_users(scenario, SMALL_PUSH)
        arriving: Final = seed_users(scenario, SMALL_PUSH)
        team: Final = created_team(scenario, create_group(gateway, kept))
        replacement: Final = (*kept[:2], *arriving)
        with _refusing_memberships_for(max(arriving)):
            refused: Final = replace_group(gateway, team, replacement)
            assert 500 <= refused.status_code < 600, refused.text
            assert_landed(gateway, team, kept)
        retried: Final = replace_group(gateway, team, replacement)
        assert retried.status_code == 200, retried.text
        assert_landed(gateway, team, replacement)


def _five_kilobyte_member(scenario: Scenario) -> str:
    user_id: Final = f"integration-scim-{uuid.uuid4().hex}-".ljust(5 * 1024, "x")
    scenario.cleanups.callback(write_rows, 'DELETE FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    return user_id


ODD_MEMBER_VALUES: Final[Mapping[str, Callable[[Scenario, Sequence[str]], JsonValue]]] = {
    "int": lambda _scenario, _kept: 7,
    "list": lambda _scenario, kept: [kept[0]],
    "empty": lambda _scenario, _kept: "",
    "five-kilobytes": lambda scenario, _kept: _five_kilobyte_member(scenario),
    "duplicate": lambda _scenario, kept: kept[0],
}


@pytest.mark.parametrize("shape", tuple(ODD_MEMBER_VALUES))
def test_a_push_with_an_odd_member_value_never_answers_500_and_keeps_the_roster_consistent(
    gateway: Gateway, shape: str
) -> None:
    with gateway.scenario() as scenario:
        kept: Final = seed_users(scenario, SMALL_PUSH)
        team: Final = created_team(scenario, create_group(gateway, kept))
        odd: Final = ODD_MEMBER_VALUES[shape](scenario, kept)
        pushed: Final = gateway.request(
            "PUT",
            f"/scim/v2/Groups/{team}",
            {
                "schemas": [GROUP_SCHEMA],
                "id": team,
                "displayName": f"integration-{uuid.uuid4().hex}",
                "members": [*members(kept), {"value": odd}],
            },
        )
        assert pushed.status_code < 500, pushed.text
        roster: Final = _roster_ids(team)
        assert len(roster) == len(frozenset(roster)), roster
        assert frozenset(kept) <= frozenset(roster), roster
        assert_landed(gateway, team, roster)
        if pushed.status_code >= 400:
            assert frozenset(roster) == frozenset(kept), (pushed.text, roster)


@dataclass(frozen=True, slots=True)
class PatchShape:
    operations: Callable[[Sequence[str]], list[JsonValue]]
    expected: Callable[[Sequence[str]], tuple[str, ...]]


PATCH_SHAPES: Final[Mapping[str, PatchShape]] = {
    "okta-remove-by-filter": PatchShape(
        operations=lambda users: [{"op": "remove", "path": f'members[value eq "{users[1]}"]'}],
        expected=lambda users: (users[0], users[2]),
    ),
    "entra-remove-by-value-list": PatchShape(
        operations=lambda users: [{"op": "Remove", "path": "members", "value": [{"value": users[1]}]}],
        expected=lambda users: (users[0], users[2]),
    ),
    "replace-members": PatchShape(
        operations=lambda users: [
            {"op": "replace", "path": "members", "value": [{"value": users[0]}, {"value": users[3]}]}
        ],
        expected=lambda users: (users[0], users[3]),
    ),
}


@pytest.mark.parametrize("shape", tuple(PATCH_SHAPES))
def test_patch_member_operation_shapes_land_the_expected_roster(gateway: Gateway, shape: str) -> None:
    with gateway.scenario() as scenario:
        users: Final = seed_users(scenario, 4)
        team: Final = created_team(scenario, create_group(gateway, users[:3]))
        patched: Final = gateway.request(
            "PATCH",
            f"/scim/v2/Groups/{team}",
            {"schemas": [PATCH_OP_SCHEMA], "Operations": PATCH_SHAPES[shape].operations(users)},
        )
        assert patched.status_code == 200, patched.text
        assert_landed(gateway, team, PATCH_SHAPES[shape].expected(users))


def _team_key_for(scenario: Scenario, team: str, user_id: str) -> str:
    key: Final = string_value(scenario.gateway.post("/key/generate", {"team_id": team, "user_id": user_id})["key"])
    scenario.cleanups.callback(delete_key_if_present, scenario.gateway, key)
    return key


def _push_outcome(candidate: Gateway, team: str, users: Sequence[str]) -> str:
    try:
        return str(replace_group(candidate, team, users).status_code)
    except httpx.TransportError as error:
        return type(error).__name__


@pytest.mark.timeout(case_timeout_seconds() + 2 * graceful_stop_seconds() + UNFIXED_PUSH_SECONDS + ROLLBACK_SECONDS)
def test_a_worker_killed_mid_push_leaves_the_old_roster_and_a_retry_lands_the_new_one(
    gateway: Gateway, tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
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
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        candidate: Final = owned.gateway
        leaving: Final = seed_users(scenario, CHAOS_GROUP)
        arriving: Final = seed_users(scenario, CHAOS_GROUP)
        team: Final = created_team(scenario, create_group(candidate, leaving))
        pinned_key: Final = _team_key_for(scenario, team, max(leaving))
        with _holding_row_lock(LOCK_KEY_ROW_SQL, sha256(pinned_key.encode()).hexdigest()) as holder_pid:
            push: Final = pool.submit(_push_outcome, candidate, team, arriving)
            _ = eventually(
                lambda: _sessions_blocked_by(holder_pid),
                lambda waiting: waiting > 0,
                seconds=UNFIXED_PUSH_SECONDS,
            )
            killed_at: Final = _database_now()
            victims: Final = tuple(sorted(owned.worker_pids()))
            assert victims, "expected at least one serving worker"
            for pid in victims:
                _ = owned.kill_worker(pid)
        record_property("interrupted_push", push.result())
        _ = eventually(lambda: _transactions_open_since(killed_at), lambda open: open == 0, seconds=ROLLBACK_SECONDS)
        settled: Final = frozenset(_roster_ids(team))
        record_property(
            "roster_after_kill",
            {"leaving": len(settled & frozenset(leaving)), "arriving": len(settled & frozenset(arriving))},
        )
        assert settled in (frozenset(leaving), frozenset(arriving)), sorted(settled)
        assert_landed(candidate, team, sorted(settled))

        retried: Final = replace_group(candidate, team, arriving)
        assert retried.status_code == 200, retried.text
        assert_landed(candidate, team, arriving)
        assert (
            read_rows(
                'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s',
                (sha256(pinned_key.encode()).hexdigest(),),
            )
            == []
        )
