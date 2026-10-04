"""`/team/delete` as one locked transaction, whatever the roster size.

The delete removes the team row, its membership rows, every member's `teams` reference and every
team key in one pass, writes one tombstone per team, evicts the cached team object and takes the
team's advisory lock (the one `/team/member_add` takes) before it writes. A roster larger than the
Prisma pool used to fail with P2028 because each member got its own transaction.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import psycopg
import pytest
import yaml
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import (
    Gateway,
    Scenario,
    delete_key_if_present,
    eventually,
    object_value,
    string_value,
)
from tests.integration._support.database import read_rows, write_rows
from tests.integration._support.process import owned_proxy_process

LARGE_ROSTER: Final = 250
POOL_LIMIT: Final = 5
# One statement seeds the whole roster: 250 individual /user/new calls would dominate the runtime.
SEED_USERS_SQL: Final = """
INSERT INTO "LiteLLM_UserTable" (user_id, user_role, teams, models)
SELECT %s || '-' || lpad(n::text, 3, '0'), 'internal_user', '{}'::text[], '{}'::text[]
FROM generate_series(1, %s::int) AS n
"""
# The master key's user id. `/team/new` appends the creator to the roster as an admin, so every team
# created here carries this member alongside the ones the test adds.
PROXY_ADMIN: Final = "default_user_id"
TAKE_TEAM_LOCK_SQL: Final = "SELECT pg_advisory_xact_lock(hashtext(%s))"
# Sessions blocked on the advisory lock the given backend holds, and nothing else on the shared rig.
WAITERS_ON_HELD_LOCK_SQL: Final = """
SELECT count(*)::int AS waiting
FROM pg_locks waiter
JOIN pg_stat_activity session ON session.pid = waiter.pid
WHERE waiter.locktype = 'advisory'
  AND NOT waiter.granted
  AND session.wait_event_type = 'Lock'
  AND session.query ILIKE %s
  AND (waiter.classid, waiter.objid, waiter.objsubid) IN (
      SELECT held.classid, held.objid, held.objsubid
      FROM pg_locks held
      WHERE held.locktype = 'advisory' AND held.granted AND held.pid = %s::int
  )
"""


def _hashed(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _team_rows(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))


def _membership_user_ids(team_id: str) -> list[JsonValue]:
    rows: Final = read_rows(
        'SELECT user_id FROM "LiteLLM_TeamMembership" WHERE team_id = %s ORDER BY user_id', (team_id,)
    )
    return [row["user_id"] for row in rows]


def _user_teams(user_id: str) -> JsonValue:
    rows: Final = read_rows('SELECT teams FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,))
    assert len(rows) == 1, f"user row for {user_id}: {rows}"
    return rows[0]["teams"]


def _users_referencing(team_id: str) -> list[JsonValue]:
    rows: Final = read_rows(
        'SELECT user_id FROM "LiteLLM_UserTable" WHERE %s = ANY(teams) ORDER BY user_id', (team_id,)
    )
    return [row["user_id"] for row in rows]


def _user_ids_with_prefix(prefix: str) -> list[JsonValue]:
    rows: Final = read_rows(
        'SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id LIKE %s ORDER BY user_id', (f"{prefix}-%",)
    )
    return [row["user_id"] for row in rows]


def _live_token(hashed: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT token, team_id FROM "LiteLLM_VerificationToken" WHERE token = %s', (hashed,))


def _deleted_token(hashed: str) -> list[dict[str, JsonValue]]:
    return read_rows('SELECT token, team_id FROM "LiteLLM_DeletedVerificationToken" WHERE token = %s', (hashed,))


def _tombstones(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT team_id, members_with_roles FROM "LiteLLM_DeletedTeamTable" WHERE team_id = %s', (team_id,)
    )


def _roster_user_ids(roster: JsonValue) -> list[str]:
    assert isinstance(roster, list), f"roster is not a list: {roster!r}"
    return sorted(string_value(object_value(member)["user_id"]) for member in roster)


def _waiters_on_lock_held_by(backend_pid: int) -> int:
    rows: Final = read_rows(WAITERS_ON_HELD_LOCK_SQL, ("%pg_advisory_xact_lock%", str(backend_pid)))
    waiting: Final = rows[0]["waiting"]
    assert isinstance(waiting, int)
    return waiting


def _remove_team_by_sql(team_id: str) -> None:
    """Cleanup for a team the test expects to have deleted itself. Whatever a failed delete left behind
    (row, memberships, `teams` references) goes by SQL so the shared rig stays clean without sending
    another request through the proxy under test."""
    if not _team_rows(team_id):
        return
    write_rows('DELETE FROM "LiteLLM_TeamMembership" WHERE team_id = %s', (team_id,))
    write_rows(
        'UPDATE "LiteLLM_UserTable" SET teams = array_remove(teams, %s) WHERE %s = ANY(teams)', (team_id, team_id)
    )
    write_rows('DELETE FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))
    assert _team_rows(team_id) == []


def _remove_team_if_present(gateway: Gateway, team_id: str) -> None:
    """Cleanup for teams the test deletes itself: the API delete first, SQL for anything it leaves."""
    if not _team_rows(team_id):
        return
    gateway.request("POST", "/team/delete", {"team_ids": [team_id]})
    _remove_team_by_sql(team_id)


def _create_team(scenario: Scenario, **fields: JsonValue) -> str:
    """A team the test deletes itself; cleanup only removes it if the test left it behind."""
    created: Final = scenario.gateway.post("/team/new", {"team_alias": f"integration-{uuid.uuid4().hex}", **fields})
    team_id: Final = string_value(created["team_id"])
    scenario.cleanups.callback(_remove_team_if_present, scenario.gateway, team_id)
    return team_id


def _generate_key(scenario: Scenario, **fields: JsonValue) -> str:
    """A key the team delete is expected to remove; cleanup only deletes it if it is still live."""
    key: Final = string_value(scenario.gateway.post("/key/generate", fields)["key"])
    scenario.cleanups.callback(delete_key_if_present, scenario.gateway, key)
    return key


def _delete_seeded_users(prefix: str) -> None:
    write_rows('DELETE FROM "LiteLLM_UserTable" WHERE user_id LIKE %s', (f"{prefix}-%",))
    assert _user_ids_with_prefix(prefix) == []


def _seed_users(scenario: Scenario, prefix: str, count: int) -> tuple[str, ...]:
    """Insert `count` user rows in one statement; ids are `<prefix>-001` … `<prefix>-<count>`."""
    users: Final = tuple(f"{prefix}-{index:03d}" for index in range(1, count + 1))
    write_rows(SEED_USERS_SQL, (prefix, str(count)))
    scenario.cleanups.callback(_delete_seeded_users, prefix)
    assert _user_ids_with_prefix(prefix) == list(users)
    return users


def _bulk_member_add(gateway: Gateway, team_id: str, users: Sequence[str]) -> None:
    gateway.post(
        "/team/member_add",
        {"team_id": team_id, "member": [{"role": "user", "user_id": user_id} for user_id in users]},
    )


def _delete_teams(gateway: Gateway, team_ids: Sequence[str]) -> httpx.Response:
    return gateway.request("POST", "/team/delete", {"team_ids": list(team_ids)})


def _team_info(gateway: Gateway, team_id: str) -> httpx.Response:
    return gateway.request("GET", "/team/info", params={"team_id": team_id})


def _team_not_found_body(team_id: str) -> dict[str, JsonValue]:
    """The proxy's exception handler wraps the 404 detail as an `error` object with the detail stringified."""
    return {
        "error": {
            "message": f"{{'message': 'Team not found, passed team id: {team_id}.'}}",
            "type": "auth_error",
            "param": "None",
            "code": "404",
        }
    }


def _chat(gateway: Gateway, model: str, key: str) -> httpx.Response:
    return gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"team delete {uuid.uuid4().hex}"}]},
        key=key,
    )


def _post_with_timeout(gateway: Gateway, path: str, body: Mapping[str, JsonValue], timeout: float) -> httpx.Response:
    """Like `Gateway.request` with a per-call timeout longer than the client's default 15 s."""
    return gateway.client.request(
        "POST", path, json=body, headers={"Authorization": f"Bearer {gateway.key}"}, timeout=timeout
    )


def _post_in_background(
    pool: ThreadPoolExecutor, gateway: Gateway, path: str, body: Mapping[str, JsonValue]
) -> Future[httpx.Response]:
    return pool.submit(_post_with_timeout, gateway, path, body, 60)


def _config_with_pool_limit(tmp_path: Path, pool_limit: int) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"]["database_connection_pool_limit"] = pool_limit
    config["general_settings"]["database_connection_pool_timeout"] = 60
    path: Final = tmp_path / f"pool-{pool_limit}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _redis() -> Redis:
    return Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"]))


def test_delete_small_team_removes_rows_keys_tombstone_and_cache(
    gateway: Gateway, record_property: Callable[[str, object], None]
) -> None:
    with gateway.scenario() as scenario, _redis() as cache:
        model: Final = scenario.model()
        users: Final = sorted(scenario.user() for _ in range(3))
        team: Final = _create_team(scenario)
        _bulk_member_add(gateway, team, users)
        keys: Final = tuple(_generate_key(scenario, team_id=team) for _ in range(2))
        hashed: Final = tuple(_hashed(key) for key in keys)
        roster: Final = sorted([PROXY_ADMIN, *users])
        assert _membership_user_ids(team) == roster
        assert _users_referencing(team) == roster
        assert all(_user_teams(user) == [team] for user in users), [_user_teams(user) for user in users]
        assert all(len(_live_token(digest)) == 1 for digest in hashed), hashed

        warm: Final = _chat(gateway, model, keys[0])
        assert warm.status_code == 200, warm.text
        team_cache_key: Final = f"team_id:{team}"
        eventually(lambda: cache.exists(team_cache_key), lambda present: present == 1, seconds=10)
        record_property("redis_keys_before_delete", sorted(entry.decode() for entry in cache.keys(f"*{team}*")))

        response: Final = _delete_teams(gateway, [team])
        assert response.status_code == 200, response.text
        assert response.json() == {"deleted_teams": [team]}

        assert _team_rows(team) == []
        assert _membership_user_ids(team) == []
        assert _users_referencing(team) == []
        assert [_user_teams(user) for user in users] == [[], [], []]
        assert [_live_token(digest) for digest in hashed] == [[], []]
        assert [_deleted_token(digest) for digest in hashed] == [
            [{"token": hashed[0], "team_id": team}],
            [{"token": hashed[1], "team_id": team}],
        ]
        tombstones: Final = _tombstones(team)
        assert len(tombstones) == 1, tombstones
        assert tombstones[0]["team_id"] == team
        assert _roster_user_ids(tombstones[0]["members_with_roles"]) == roster

        info: Final = _team_info(gateway, team)
        assert info.status_code == 404, info.text
        assert info.json() == _team_not_found_body(team)

        assert cache.exists(team_cache_key) == 0
        record_property("redis_keys_after_delete", sorted(entry.decode() for entry in cache.keys(f"*{team}*")))


@pytest.mark.timeout(240)  # owned two-worker proxy boot plus a 250-member roster
def test_delete_250_member_team_succeeds_with_pool_limit_five_on_two_workers(
    gateway: Gateway, tmp_path: Path, record_property: Callable[[str, object], None]
) -> None:
    prefix: Final = f"integration-roster-{uuid.uuid4().hex}"
    # The scenario is bound to the shared gateway and its cleanups are SQL, so a failed delete on the
    # owned proxy (and whatever it does to that proxy's workers) cannot mask the assertion below with a
    # second failure during cleanup. The owned proxy is stopped before the cleanups run.
    with (
        gateway.scenario() as scenario,
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": os.environ["DATABASE_URL"]},
            config=_config_with_pool_limit(tmp_path, POOL_LIMIT),
            remove_environment=("DATABASE_URL_READ_REPLICA",),
            workers=2,
        ) as owned,
    ):
        team: Final = string_value(
            owned.gateway.post("/team/new", {"team_alias": f"integration-{uuid.uuid4().hex}"})["team_id"]
        )
        scenario.cleanups.callback(_remove_team_by_sql, team)
        users: Final = _seed_users(scenario, prefix, LARGE_ROSTER)
        added: Final = _post_with_timeout(
            owned.gateway,
            "/team/member_add",
            {"team_id": team, "member": [{"role": "user", "user_id": user_id} for user_id in users]},
            timeout=120,
        )
        assert added.status_code == 200, added.text
        roster: Final = sorted([PROXY_ADMIN, *users])
        assert _membership_user_ids(team) == roster
        assert _users_referencing(team) == roster

        response: Final = _post_with_timeout(owned.gateway, "/team/delete", {"team_ids": [team]}, timeout=120)
        record_property("h2_delete_response", f"{response.status_code} {response.text[:300]}")
        assert response.status_code == 200, (
            f"/team/delete of a {LARGE_ROSTER}-member team with database_connection_pool_limit={POOL_LIMIT}: "
            f"{response.status_code} {response.text}"
        )
        assert response.json() == {"deleted_teams": [team]}
        assert _team_rows(team) == []
        assert _membership_user_ids(team) == []
        assert _users_referencing(team) == []
        assert _user_ids_with_prefix(prefix) == list(users)
        tombstones: Final = _tombstones(team)
        assert len(tombstones) == 1, tombstones
        assert _roster_user_ids(tombstones[0]["members_with_roles"]) == roster


def test_delete_waits_for_the_team_advisory_lock_and_completes_after_release(
    gateway: Gateway, record_property: Callable[[str, object], None]
) -> None:
    with gateway.scenario() as scenario:
        user: Final = scenario.user()
        team: Final = _create_team(scenario)
        _bulk_member_add(gateway, team, [user])
        hashed: Final = _hashed(_generate_key(scenario, team_id=team))
        with ThreadPoolExecutor(max_workers=1) as pool:
            with psycopg.connect(os.environ["DATABASE_URL"]) as holder:
                holder.execute(TAKE_TEAM_LOCK_SQL, (team,))
                pending: Final = _post_in_background(pool, gateway, "/team/delete", {"team_ids": [team]})
                eventually(
                    lambda: _waiters_on_lock_held_by(holder.info.backend_pid),
                    lambda waiting: waiting >= 1,
                    seconds=20,
                )
                assert not pending.done(), "delete returned while the team lock was still held"
                assert _team_rows(team) == [{"team_id": team}], "team row deleted while the team lock was held"
                # Recorded before the count assertion so both legs document what the delete had already
                # written by the time it reached the lock.
                record_property(
                    "state_while_blocked",
                    json.dumps(
                        {
                            "membership_user_ids": _membership_user_ids(team),
                            "user_teams": _user_teams(user),
                            "live_token_rows": len(_live_token(hashed)),
                            "tombstones": len(_tombstones(team)),
                        }
                    ),
                )
                # One transaction per delete: a per-member fan-out would queue one waiter per roster entry.
                eventually(
                    lambda: _waiters_on_lock_held_by(holder.info.backend_pid),
                    lambda waiting: waiting == 1,
                    seconds=10,
                )
            # leaving the holder block commits its transaction, which releases the advisory lock
            response: Final = pending.result(timeout=60)
        assert response.status_code == 200, response.text
        assert response.json() == {"deleted_teams": [team]}
        assert _team_rows(team) == []
        assert _membership_user_ids(team) == []
        assert _user_teams(user) == []
        assert _live_token(hashed) == []
        assert len(_tombstones(team)) == 1, _tombstones(team)


def test_deleting_two_teams_sharing_a_member_in_one_call_clears_both_from_the_member(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user: Final = scenario.user()
        first: Final = _create_team(scenario)
        second: Final = _create_team(scenario)
        _bulk_member_add(gateway, first, [user])
        _bulk_member_add(gateway, second, [user])
        assert _user_teams(user) == [first, second]

        response: Final = _delete_teams(gateway, [first, second])
        assert response.status_code == 200, response.text
        assert response.json() == {"deleted_teams": [first, second]}

        assert _team_rows(first) == []
        assert _team_rows(second) == []
        assert _membership_user_ids(first) == []
        assert _membership_user_ids(second) == []
        assert _user_teams(user) == []
        assert [row["team_id"] for row in _tombstones(first)] == [first]
        assert [row["team_id"] for row in _tombstones(second)] == [second]


def test_deleting_one_team_leaves_the_members_other_team_and_key_intact(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user()
        deleted: Final = _create_team(scenario)
        kept: Final = scenario.team()
        _bulk_member_add(gateway, deleted, [user])
        _bulk_member_add(gateway, kept, [user])
        kept_key: Final = scenario.key(team_id=kept, user_id=user)
        before: Final = _chat(gateway, model, kept_key)
        assert before.status_code == 200, before.text

        response: Final = _delete_teams(gateway, [deleted])
        assert response.status_code == 200, response.text
        assert response.json() == {"deleted_teams": [deleted]}

        assert _team_rows(deleted) == []
        assert _team_rows(kept) == [{"team_id": kept}]
        assert _membership_user_ids(deleted) == []
        assert _membership_user_ids(kept) == [PROXY_ADMIN, user]
        assert _user_teams(user) == [kept]
        assert _live_token(_hashed(kept_key)) == [{"token": _hashed(kept_key), "team_id": kept}]
        after: Final = _chat(gateway, model, kept_key)
        assert after.status_code == 200, after.text


@pytest.mark.timeout(240)  # owned proxy boot
def test_delete_writes_one_audit_row_for_the_team_and_one_per_key(gateway: Gateway, tmp_path: Path) -> None:
    with (
        owned_proxy_process(
            gateway,
            tmp_path,
            {"DATABASE_URL": os.environ["DATABASE_URL"], "LITELLM_STORE_AUDIT_LOGS": "true"},
            remove_environment=("DATABASE_URL_READ_REPLICA",),
        ) as owned,
        owned.gateway.scenario() as scenario,
    ):
        user: Final = scenario.user()
        team: Final = _create_team(scenario)
        _bulk_member_add(owned.gateway, team, [user])
        hashed: Final = _hashed(_generate_key(scenario, team_id=team))

        response: Final = _delete_teams(owned.gateway, [team])
        assert response.status_code == 200, response.text
        assert response.json() == {"deleted_teams": [team]}
        assert _team_rows(team) == []

        def deleted_audit_rows() -> list[dict[str, JsonValue]]:
            return read_rows(
                'SELECT table_name, action, object_id FROM "LiteLLM_AuditLog" '
                "WHERE object_id IN (%s, %s) AND action = 'deleted' ORDER BY table_name",
                (team, hashed),
            )

        rows: Final = eventually(deleted_audit_rows, lambda found: len(found) >= 2, seconds=30)
        assert rows == [
            {"table_name": "LiteLLM_TeamTable", "action": "deleted", "object_id": team},
            {"table_name": "LiteLLM_VerificationToken", "action": "deleted", "object_id": hashed},
        ]


def test_second_delete_of_the_same_team_is_404_with_one_tombstone(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        user: Final = scenario.user()
        team: Final = _create_team(scenario)
        _bulk_member_add(gateway, team, [user])

        first: Final = _delete_teams(gateway, [team])
        assert first.status_code == 200, first.text
        assert first.json() == {"deleted_teams": [team]}

        second: Final = _delete_teams(gateway, [team])
        assert second.status_code == 404, second.text
        assert second.json() == {"detail": {"error": f"Team not found, passed team_id={team}"}}

        assert _team_rows(team) == []
        assert [row["team_id"] for row in _tombstones(team)] == [team]
        assert _user_teams(user) == []


def test_delete_empty_team_writes_tombstone_and_team_info_is_404(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = _create_team(scenario)
        present: Final = _team_info(gateway, team)
        assert present.status_code == 200, present.text

        response: Final = _delete_teams(gateway, [team])
        assert response.status_code == 200, response.text
        assert response.json() == {"deleted_teams": [team]}

        assert _team_rows(team) == []
        assert _membership_user_ids(team) == []
        assert _tombstones(team) == [
            {"team_id": team, "members_with_roles": [{"role": "admin", "user_id": PROXY_ADMIN, "user_email": None}]}
        ]
        info: Final = _team_info(gateway, team)
        assert info.status_code == 404, info.text
        assert info.json() == _team_not_found_body(team)


def test_delete_keys_only_team_removes_keys_and_revokes_them(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team: Final = _create_team(scenario)
        keys: Final = tuple(_generate_key(scenario, team_id=team) for _ in range(2))
        hashed: Final = tuple(_hashed(key) for key in keys)
        assert _membership_user_ids(team) == [PROXY_ADMIN]
        for key in keys:
            warm = _chat(gateway, model, key)
            assert warm.status_code == 200, warm.text

        response: Final = _delete_teams(gateway, [team])
        assert response.status_code == 200, response.text
        assert response.json() == {"deleted_teams": [team]}

        assert _team_rows(team) == []
        assert [_live_token(digest) for digest in hashed] == [[], []]
        assert [_deleted_token(digest) for digest in hashed] == [
            [{"token": hashed[0], "team_id": team}],
            [{"token": hashed[1], "team_id": team}],
        ]
        for key in keys:
            revoked = _chat(gateway, model, key)
            assert revoked.status_code == 401, f"{revoked.status_code} {revoked.text}"


def test_delete_three_teams_in_one_call_lists_all_and_tombstones_each_once(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        teams: Final = tuple(_create_team(scenario) for _ in range(3))
        for team in teams:
            _bulk_member_add(gateway, team, [scenario.user()])

        response: Final = _delete_teams(gateway, teams)
        assert response.status_code == 200, response.text
        assert response.json() == {"deleted_teams": list(teams)}

        for team in teams:
            assert _team_rows(team) == []
            assert _membership_user_ids(team) == []
            assert _users_referencing(team) == []
            assert [row["team_id"] for row in _tombstones(team)] == [team]


def test_recreating_the_same_team_id_after_delete_serves_the_fresh_team(gateway: Gateway) -> None:
    with gateway.scenario() as scenario, _redis() as cache:
        model: Final = scenario.model()
        original_member: Final = scenario.user()
        replacement_member: Final = scenario.user()
        team: Final = _create_team(scenario)
        _bulk_member_add(gateway, team, [original_member])
        original_key: Final = _generate_key(scenario, team_id=team)
        warm: Final = _chat(gateway, model, original_key)
        assert warm.status_code == 200, warm.text
        team_cache_key: Final = f"team_id:{team}"
        eventually(lambda: cache.exists(team_cache_key), lambda present: present == 1, seconds=10)

        response: Final = _delete_teams(gateway, [team])
        assert response.status_code == 200, response.text
        assert _team_rows(team) == []
        assert cache.exists(team_cache_key) == 0

        fresh_alias: Final = f"integration-recreated-{uuid.uuid4().hex}"
        recreated: Final = gateway.request(
            "POST",
            "/team/new",
            {
                "team_id": team,
                "team_alias": fresh_alias,
                "members_with_roles": [{"role": "user", "user_id": replacement_member}],
            },
        )
        assert recreated.status_code == 200, recreated.text
        assert recreated.json()["team_id"] == team

        info: Final = _team_info(gateway, team)
        assert info.status_code == 200, info.text
        team_info: Final = object_value(info.json()["team_info"])
        assert team_info["team_alias"] == fresh_alias
        assert _roster_user_ids(team_info["members_with_roles"]) == [PROXY_ADMIN, replacement_member]
        assert _membership_user_ids(team) == [PROXY_ADMIN, replacement_member]
        assert _user_teams(replacement_member) == [team]
        assert _user_teams(original_member) == []

        fresh_key: Final = scenario.key(team_id=team)
        served: Final = _chat(gateway, model, fresh_key)
        assert served.status_code == 200, served.text
        cached: Final = eventually(lambda: cache.get(team_cache_key), lambda value: value is not None, seconds=10)
        assert isinstance(cached, bytes), cached
        assert json.loads(cached)["team_alias"] == fresh_alias, cached
        revoked: Final = _chat(gateway, model, original_key)
        assert revoked.status_code == 401, f"{revoked.status_code} {revoked.text}"


def test_member_add_and_delete_released_together_leave_no_team_reference(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        newcomer: Final = scenario.user()
        team: Final = _create_team(scenario)
        with ThreadPoolExecutor(max_workers=2) as pool:
            with psycopg.connect(os.environ["DATABASE_URL"]) as holder:
                holder.execute(TAKE_TEAM_LOCK_SQL, (team,))
                pending_delete: Final = _post_in_background(pool, gateway, "/team/delete", {"team_ids": [team]})
                pending_add: Final = _post_in_background(
                    pool,
                    gateway,
                    "/team/member_add",
                    {"team_id": team, "member": {"role": "user", "user_id": newcomer}},
                )
                eventually(
                    lambda: _waiters_on_lock_held_by(holder.info.backend_pid),
                    lambda waiting: waiting == 2,
                    seconds=20,
                )
                assert not pending_delete.done() and not pending_add.done()
            # leaving the holder block commits its transaction, which releases the advisory lock
            deleted: Final = pending_delete.result(timeout=60)
            added: Final = pending_add.result(timeout=60)
        assert deleted.status_code == 200, deleted.text
        assert deleted.json() == {"deleted_teams": [team]}
        assert added.status_code in (200, 404), f"{added.status_code} {added.text}"
        assert _team_rows(team) == []
        assert _membership_user_ids(team) == []
        assert _user_teams(newcomer) == []
        assert _users_referencing(team) == []
        assert [row["team_id"] for row in _tombstones(team)] == [team]
