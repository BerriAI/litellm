"""SCIM v2 group helpers for the management cells: seeded users, group pushes, the team lock, and what a group
leaves behind in Postgres."""

import os
import uuid
from collections.abc import Generator, Mapping, Sequence
from contextlib import contextmanager
from typing import Final

import httpx
import psycopg
from integration._support.client import JSON_OBJECT, Gateway, Scenario, object_value, string_value
from integration._support.database import read_rows, write_rows
from pydantic import JsonValue

GROUP_SCHEMA: Final = "urn:ietf:params:scim:schemas:core:2.0:Group"
PATCH_OP_SCHEMA: Final = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
SEED_USERS_SQL: Final = """
INSERT INTO "LiteLLM_UserTable" (user_id, user_role, teams, models)
SELECT %s || '-' || lpad(n::text, 3, '0'), 'internal_user', '{}'::text[], '{}'::text[]
FROM generate_series(1, %s::int) AS n
"""
TAKE_TEAM_LOCK_SQL: Final = "SELECT pg_advisory_xact_lock(hashtext(%s))"
WAITERS_ON_HELD_LOCK_SQL: Final = """
SELECT count(*)::int AS waiting
FROM pg_locks waiter
JOIN pg_stat_activity session ON session.pid = waiter.pid
WHERE waiter.locktype = 'advisory'
  AND NOT waiter.granted
  AND session.wait_event_type = 'Lock'
  AND (waiter.classid, waiter.objid, waiter.objsubid) IN (
      SELECT held.classid, held.objid, held.objsubid
      FROM pg_locks held
      WHERE held.locktype = 'advisory' AND held.granted AND held.pid = %s::int
  )
"""


def seed_users(scenario: Scenario, count: int) -> tuple[str, ...]:
    prefix: Final = f"integration-scim-{uuid.uuid4().hex}"
    write_rows(SEED_USERS_SQL, (prefix, str(count)))
    scenario.cleanups.callback(write_rows, 'DELETE FROM "LiteLLM_UserTable" WHERE user_id LIKE %s', (f"{prefix}-%",))
    return tuple(f"{prefix}-{index:03d}" for index in range(1, count + 1))


def members(users: Sequence[str]) -> list[JsonValue]:
    return [{"value": user} for user in users]


def group_name() -> str:
    return f"integration-{uuid.uuid4().hex}"


def create_group(candidate: Gateway, users: Sequence[str], *, display_name: str | None = None) -> httpx.Response:
    return candidate.request(
        "POST",
        "/scim/v2/Groups",
        {"schemas": [GROUP_SCHEMA], "displayName": display_name or group_name(), "members": members(users)},
    )


def replace_group(
    candidate: Gateway, team: str, users: Sequence[str], *, display_name: str | None = None
) -> httpx.Response:
    return candidate.request(
        "PUT",
        f"/scim/v2/Groups/{team}",
        {"schemas": [GROUP_SCHEMA], "id": team, "displayName": display_name or group_name(), "members": members(users)},
    )


def add_to_group(candidate: Gateway, team: str, users: Sequence[str]) -> httpx.Response:
    return candidate.request(
        "PATCH",
        f"/scim/v2/Groups/{team}",
        {"schemas": [PATCH_OP_SCHEMA], "Operations": [{"op": "add", "path": "members", "value": members(users)}]},
    )


def body(response: httpx.Response) -> dict[str, JsonValue]:
    return JSON_OBJECT.validate_json(response.content)


def member_ids(group: Mapping[str, JsonValue]) -> frozenset[str]:
    listed: Final = group.get("members") or []
    assert isinstance(listed, list), listed
    return frozenset(string_value(object_value(member)["value"]) for member in listed)


def delete_group(candidate: Gateway, team: str, *, key: str | None = None) -> httpx.Response:
    return candidate.request("DELETE", f"/scim/v2/Groups/{team}", key=key)


def delete_team_if_present(scenario: Scenario, team: str) -> None:
    if read_rows('SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,)):
        scenario.delete_team(team)


def created_team(scenario: Scenario, created: httpx.Response) -> str:
    assert created.status_code == 201, created.text
    team: Final = string_value(body(created)["id"])
    scenario.cleanups.callback(delete_team_if_present, scenario, team)
    return team


def team_key(candidate: Gateway, team: str) -> str:
    created: Final = candidate.post("/key/generate", {"team_id": team, "key_alias": f"integration-{uuid.uuid4().hex}"})
    return string_value(created["key"])


def membership_user_ids(team: str) -> frozenset[str]:
    rows: Final = read_rows('SELECT user_id FROM "LiteLLM_TeamMembership" WHERE team_id = %s', (team,))
    return frozenset(string_value(row["user_id"]) for row in rows)


def keys_of(team: str) -> frozenset[str]:
    rows: Final = read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE team_id = %s', (team,))
    return frozenset(string_value(row["token"]) for row in rows)


def users_referencing(team: str) -> frozenset[str]:
    rows: Final = read_rows('SELECT user_id FROM "LiteLLM_UserTable" WHERE %s = ANY(teams)', (team,))
    return frozenset(string_value(row["user_id"]) for row in rows)


def user_role(candidate: Gateway, user: str) -> str:
    return string_value(object_value(candidate.get("/user/info", {"user_id": user})["user_info"])["user_role"])


def groups_of(candidate: Gateway, user: str) -> frozenset[str]:
    groups: Final = candidate.get(f"/scim/v2/Users/{user}").get("groups") or []
    assert isinstance(groups, list), groups
    return frozenset(string_value(object_value(group)["value"]) for group in groups)


def assert_landed(candidate: Gateway, team: str, users: Sequence[str]) -> None:
    assert member_ids(candidate.get(f"/scim/v2/Groups/{team}")) == frozenset(users)
    assert membership_user_ids(team) == frozenset(users)
    assert users_referencing(team) == frozenset(users)


def assert_gone(candidate: Gateway, team: str, users: Sequence[str], key: str) -> None:
    assert candidate.request("GET", f"/scim/v2/Groups/{team}").status_code == 404
    assert team not in groups_of(candidate, users[0])
    assert membership_user_ids(team) == frozenset()
    assert users_referencing(team) == frozenset()
    assert keys_of(team) == frozenset()
    assert candidate.request("GET", "/v1/models", key=key).status_code == 401


@contextmanager
def held_team_lock(team: str) -> Generator[int]:
    """Hold the team's advisory lock from a session of our own and yield its backend pid; leaving the block
    commits, which releases the lock to whoever is queued on it."""
    with psycopg.connect(os.environ["DATABASE_URL"]) as holder:
        holder.execute(TAKE_TEAM_LOCK_SQL, (team,))
        yield holder.info.backend_pid


def waiters_on_lock_held_by(backend_pid: int) -> int:
    rows: Final = read_rows(WAITERS_ON_HELD_LOCK_SQL, (str(backend_pid),))
    return int(string_value(rows[0]["waiting"]))
