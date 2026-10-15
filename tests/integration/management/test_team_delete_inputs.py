"""Sad inputs for /team/delete: malformed ids, callers without access, and rosters the API can no longer produce.

Legacy roster shapes (email-only entries, entries with neither id nor email) are seeded straight into
``members_with_roles`` because ``/team/member_add`` backfills ``user_id`` and will not write them any more.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Mapping
from hashlib import sha256
from typing import Final

import httpx
import pytest
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

RecordProperty = Callable[[str, object], None]

TEAM_SQL: Final = 'SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = %s'
ROSTER_READ_SQL: Final = 'SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s'
ROSTER_SQL: Final = 'UPDATE "LiteLLM_TeamTable" SET members_with_roles = %s::jsonb WHERE team_id = %s'
MEMBERSHIP_SQL: Final = 'SELECT user_id FROM "LiteLLM_TeamMembership" WHERE team_id = %s'
USER_SQL: Final = 'SELECT user_id FROM "LiteLLM_UserTable" WHERE user_id = %s'
USER_EMAIL_SQL: Final = 'UPDATE "LiteLLM_UserTable" SET user_email = %s WHERE user_id = %s'
TOKEN_SQL: Final = 'SELECT token FROM "LiteLLM_VerificationToken" WHERE token = %s'
TOMBSTONE_SQL: Final = 'SELECT id FROM "LiteLLM_DeletedTeamTable" WHERE team_id = %s'
AUDIT_SQL: Final = 'SELECT id, table_name, action FROM "LiteLLM_AuditLog" WHERE object_id = %s'

NOT_FOUND: Final = "Team not found, passed team_id="
# /team/delete sits on management_routes but on no internal-user route list, so the route gate in
# RouteChecks.non_proxy_admin_allowed_routes_check answers 401 before _verify_team_access ever runs
# (pinned by tests/integration/authorization/test_team_admin_gate.py as team_admin=401, others=401).
ROUTE_GATE_MESSAGE: Final = "Only proxy admin can be used"
UNKNOWN_TEAM: Final = f"integration-missing-{uuid.uuid4().hex}"
FIVE_KB_TEAM: Final = "t" * 5120


def _team_rows(team_id: str) -> list[dict[str, JsonValue]]:
    return read_rows(TEAM_SQL, (team_id,))


def _delete_team_if_present(gateway: Gateway, team_id: str) -> None:
    """Cleanup for a team the test deletes itself: a no-op once the row is gone."""
    if _team_rows(team_id):
        gateway.post("/team/delete", {"team_ids": [team_id]})
    assert _team_rows(team_id) == []


def _reset_roster_if_present(team_id: str) -> None:
    """Cleanup for a seeded roster: put back a shape the delete path always accepts."""
    if _team_rows(team_id):
        write_rows(ROSTER_SQL, ("[]", team_id))


def _delete_user_if_present(gateway: Gateway, user_id: str) -> None:
    if read_rows(USER_SQL, (user_id,)):
        response: Final = gateway.request("POST", "/user/delete", {"user_ids": [user_id]})
        assert response.status_code == 200, response.text
    assert read_rows(USER_SQL, (user_id,)) == []


def _own_team(scenario: Scenario, **fields: JsonValue) -> str:
    """A team the test deletes itself, so cleanup tolerates the row already being gone."""
    created: Final = scenario.gateway.post("/team/new", {"team_alias": f"integration-{uuid.uuid4().hex}", **fields})
    team_id: Final = string_value(created["team_id"])
    scenario.cleanups.callback(_delete_team_if_present, scenario.gateway, team_id)
    return team_id


def _own_key(scenario: Scenario, **fields: JsonValue) -> str:
    """A key the team delete is expected to remove, so cleanup tolerates it already being gone."""
    created: Final = scenario.gateway.post("/key/generate", fields)
    token: Final = string_value(created["key"])
    scenario.cleanups.callback(delete_key_if_present, scenario.gateway, token)
    return token


def _own_user(scenario: Scenario) -> str:
    """An internal user the test may remove by SQL, so cleanup tolerates the row already being gone."""
    created: Final = scenario.gateway.post(
        "/user/new",
        {"user_id": f"integration-{uuid.uuid4().hex}", "auto_create_key": False, "user_role": "internal_user"},
    )
    user_id: Final = string_value(created["user_id"])
    scenario.cleanups.callback(_delete_user_if_present, scenario.gateway, user_id)
    return user_id


def _seed_roster(scenario: Scenario, team_id: str, entries: list[dict[str, JsonValue]]) -> None:
    write_rows(ROSTER_SQL, (json.dumps(entries), team_id))
    scenario.cleanups.callback(_reset_roster_if_present, team_id)


def _team_admin(scenario: Scenario, team_id: str) -> str:
    """Add a member and flip their roster role to admin by SQL: the API gates that role behind a license."""
    user_id: Final = scenario.member(team_id)
    rows: Final = read_rows(ROSTER_READ_SQL, (team_id,))
    assert len(rows) == 1, rows
    roster: Final = rows[0]["members_with_roles"]
    assert isinstance(roster, list), roster
    promoted: Final = [
        {**object_value(entry), "role": "admin"} if object_value(entry).get("user_id") == user_id else entry
        for entry in roster
    ]
    assert any(object_value(entry).get("user_id") == user_id for entry in promoted), promoted
    write_rows(ROSTER_SQL, (json.dumps(promoted), team_id))
    return user_id


def _membership_user_ids(team_id: str) -> frozenset[str]:
    return frozenset(string_value(row["user_id"]) for row in read_rows(MEMBERSHIP_SQL, (team_id,)))


def _delete(gateway: Gateway, team_ids: JsonValue, *, key: str | None = None) -> httpx.Response:
    return gateway.request("POST", "/team/delete", {"team_ids": team_ids}, key=key)


def _hashed(token: str) -> str:
    return sha256(token.encode()).hexdigest()


@pytest.mark.parametrize(
    ("body", "status", "needle"),
    [
        pytest.param({"team_ids": [UNKNOWN_TEAM]}, 404, f"{NOT_FOUND}{UNKNOWN_TEAM}", id="S1-unknown-id"),
        pytest.param({"team_ids": "abc"}, 422, "list_type", id="S3-string-not-list"),
        pytest.param({"team_ids": [123]}, 422, "string_type", id="S4-integer-item"),
        pytest.param({"team_ids": [""]}, 404, NOT_FOUND, id="S5-empty-id"),
        pytest.param({"team_ids": [FIVE_KB_TEAM]}, 404, NOT_FOUND, id="S6-5kb-id"),
    ],
)
def test_rejects_malformed_team_ids(gateway: Gateway, body: Mapping[str, JsonValue], status: int, needle: str) -> None:
    response: Final = gateway.request("POST", "/team/delete", body)
    assert response.status_code == status, f"{response.status_code} {response.text}"
    assert needle in response.text, response.text


def test_empty_list_deletes_nothing(gateway: Gateway) -> None:
    response: Final = _delete(gateway, [])
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    assert response.json() == {"deleted_teams": []}, response.text


def test_duplicate_ids_delete_once(gateway: Gateway, record_property: RecordProperty) -> None:
    """Repeated ids collapse to one delete: the body names the team once and exactly one tombstone row lands."""
    with gateway.scenario() as scenario:
        team: Final = _own_team(scenario)
        first: Final = scenario.member(team)
        second: Final = scenario.member(team)
        key: Final = _own_key(scenario, team_id=team)
        # The master key's /team/new also seats the proxy admin, so the table holds more than these two.
        assert {first, second} <= _membership_user_ids(team), _membership_user_ids(team)
        response: Final = _delete(gateway, [team, team])
        # Read every table before the first assert so a red cell carries the partial state with it.
        present: Final = _team_rows(team)
        memberships: Final = _membership_user_ids(team)
        key_rows: Final = read_rows(TOKEN_SQL, (_hashed(key),))
        tombstones: Final = read_rows(TOMBSTONE_SQL, (team,))
        audit: Final = read_rows(AUDIT_SQL, (team,))
        state: Final = (
            f"team_present={bool(present)} membership_rows={len(memberships)} key_present={bool(key_rows)} "
            f"tombstone_rows={len(tombstones)} audit_rows={len(audit)}"
        )
        record_property("status", response.status_code)
        record_property("body", response.text)
        record_property("state_after", state)
        record_property("audit_rows", len(audit))  # recorded only: the shared rigs cannot enable audit logging
        assert response.status_code == 200, f"{response.status_code} {response.text}; {state}"
        assert response.json() == {"deleted_teams": [team]}, response.text
        assert present == [], state
        assert memberships == frozenset(), state
        assert key_rows == [], state
        assert len(tombstones) == 1, f"tombstone rows for {team}: {len(tombstones)}; {state}"


def test_missing_authorization_is_401(gateway: Gateway) -> None:
    response: Final = gateway.client.post("/team/delete", json={"team_ids": [UNKNOWN_TEAM]})
    assert response.status_code == 401, f"{response.status_code} {response.text}"
    assert "error" in response.text.lower(), response.text


def test_internal_user_outside_team_is_refused_by_the_route_gate(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        outsider: Final = scenario.user(user_role="internal_user")
        key: Final = scenario.key(user_id=outsider)
        response: Final = _delete(gateway, [team], key=key)
        assert response.status_code == 401, f"{response.status_code} {response.text}"
        assert ROUTE_GATE_MESSAGE in response.text, response.text
        assert len(_team_rows(team)) == 1


def test_admin_of_another_team_is_refused_by_the_route_gate(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        target: Final = scenario.team()
        other: Final = scenario.team()
        admin: Final = _team_admin(scenario, other)
        key: Final = scenario.key(team_id=other, user_id=admin)
        response: Final = _delete(gateway, [target], key=key)
        assert response.status_code == 401, f"{response.status_code} {response.text}"
        assert ROUTE_GATE_MESSAGE in response.text, response.text
        assert len(_team_rows(target)) == 1
        assert len(_team_rows(other)) == 1


def test_team_admin_of_own_team_is_refused_by_the_route_gate(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        admin: Final = _team_admin(scenario, team)
        key: Final = scenario.key(team_id=team, user_id=admin)
        response: Final = _delete(gateway, [team], key=key)
        assert response.status_code == 401, f"{response.status_code} {response.text}"
        assert ROUTE_GATE_MESSAGE in response.text, response.text
        assert len(_team_rows(team)) == 1


def test_roster_user_whose_row_was_removed(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        team: Final = _own_team(scenario)
        ghost: Final = _own_user(scenario)
        gateway.post("/team/member_add", {"team_id": team, "member": {"role": "user", "user_id": ghost}})
        assert ghost in _membership_user_ids(team), _membership_user_ids(team)
        write_rows('DELETE FROM "LiteLLM_UserTable" WHERE user_id = %s', (ghost,))
        assert read_rows(USER_SQL, (ghost,)) == []
        response: Final = _delete(gateway, [team])
        assert response.status_code == 200, f"{response.status_code} {response.text}"
        assert _team_rows(team) == []
        assert read_rows(MEMBERSHIP_SQL, (team,)) == []


def test_email_only_roster_entry_matching_no_user(gateway: Gateway, record_property: RecordProperty) -> None:
    with gateway.scenario() as scenario:
        team: Final = _own_team(scenario)
        _seed_roster(
            scenario,
            team,
            [{"role": "user", "user_id": None, "user_email": f"nobody-{uuid.uuid4().hex}@example.com"}],
        )
        response: Final = _delete(gateway, [team])
        record_property("status", response.status_code)
        record_property("body", response.text)
        assert response.status_code == 200, f"{response.status_code} {response.text}"
        assert _team_rows(team) == []


def test_email_only_roster_entry_matching_two_case_variants(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        tag: Final = uuid.uuid4().hex
        upper: Final = scenario.user(user_role="internal_user")
        lower: Final = scenario.user(user_role="internal_user")
        # /user/new rejects a second email that matches case-insensitively, so the pair is seeded by SQL.
        write_rows(USER_EMAIL_SQL, (f"Case-{tag}@example.com", upper))
        write_rows(USER_EMAIL_SQL, (f"case-{tag}@example.com", lower))
        for user in (upper, lower):
            gateway.chat(model, key=scenario.key(user_id=user, models=[model]))
        with Redis(host=os.environ["REDIS_HOST"], port=int(os.environ["REDIS_PORT"])) as cache:

            def cached() -> dict[str, int]:
                return {user: int(cache.exists(user)) for user in (upper, lower)}

            assert eventually(cached, lambda seen: seen == {upper: 1, lower: 1}, seconds=10) == {upper: 1, lower: 1}
            team: Final = _own_team(scenario)
            _seed_roster(scenario, team, [{"role": "user", "user_id": None, "user_email": f"CASE-{tag}@EXAMPLE.COM"}])
            response: Final = _delete(gateway, [team])
            assert response.status_code == 200, f"{response.status_code} {response.text}"
            assert _team_rows(team) == []
            remaining: Final = eventually(
                cached, lambda seen: seen == {upper: 0, lower: 0}, seconds=10, return_last_on_timeout=True
            )
            assert remaining == {upper: 0, lower: 0}, (
                f"user cache entries still present after /team/delete: "
                f"{upper} exists={remaining[upper]}, {lower} exists={remaining[lower]}"
            )


def test_roster_entry_without_id_or_email_pins_the_500(gateway: Gateway, record_property: RecordProperty) -> None:
    """Pins a pre-existing defect outside this PR's diff until it gets its own ticket: for a roster entry with neither
    id nor email, LiteLLM_TeamTable.model_validate raises outside delete_team's 404 try/except, so the call is a 500
    that writes nothing (team row intact, no tombstone, membership rows untouched)."""
    with gateway.scenario() as scenario:
        team: Final = _own_team(scenario)
        before: Final = _membership_user_ids(team)
        _seed_roster(scenario, team, [{"role": "user", "user_id": None, "user_email": None}])
        response: Final = _delete(gateway, [team])
        present: Final = _team_rows(team)
        tombstones: Final = read_rows(TOMBSTONE_SQL, (team,))
        after: Final = _membership_user_ids(team)
        state: Final = f"team_present={bool(present)} tombstone_rows={len(tombstones)} membership_rows={len(after)}"
        record_property("status", response.status_code)
        record_property("body", response.text)
        record_property("state_after", state)
        assert response.status_code == 500, f"{response.status_code} {response.text}; {state}"
        assert "Internal server error" in response.text, response.text
        assert len(present) == 1, state
        assert tombstones == [], state
        assert after == before, f"membership rows changed: before={sorted(before)} after={sorted(after)}"


def test_failed_delete_leaves_unrelated_key_serving(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        key: Final = scenario.key(models=[model])
        assert object_value(gateway.chat(model, key=key)["usage"])["total_tokens"] == 40
        missing: Final = f"integration-missing-{uuid.uuid4().hex}"
        response: Final = _delete(gateway, [missing])
        assert response.status_code == 404, f"{response.status_code} {response.text}"
        assert f"{NOT_FOUND}{missing}" in response.text, response.text
        completion: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "after failed delete"}]},
            key=key,
        )
        assert completion.status_code == 200, f"{completion.status_code} {completion.text}"
        assert object_value(object_value(completion.json())["usage"])["total_tokens"] == 40
