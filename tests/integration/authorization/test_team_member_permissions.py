from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import httpx
from integration._support.client import Gateway, Scenario, object_value, string_value
from integration._support.database import read_rows
from pydantic import JsonValue

PERMISSION_ERROR: Final = "team_member_permission_error"


@dataclass(frozen=True, slots=True)
class Member:
    team_id: str
    team_key: str
    member_key: str


def _member(scenario: Scenario, permissions: list[JsonValue] | None) -> Member:
    team_id: Final = scenario.team() if permissions is None else scenario.team(team_member_permissions=permissions)
    team_key: Final = scenario.key(team_id=team_id, metadata={"owner": "team"})
    member: Final = scenario.member(team_id)
    return Member(team_id, team_key, scenario.key(user_id=member))


def _team_key_row(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(
        'SELECT team_id, metadata FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (sha256(key.encode()).hexdigest(),),
    )


def _team_key_count(team_id: str) -> int:
    return len(read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE team_id = %s', (team_id,)))


def _refused(response: httpx.Response, status: int, error_type: str | None = None) -> None:
    assert response.status_code == status, response.text
    if error_type is not None:
        assert object_value(response.json()["error"])["type"] == error_type, response.text


def test_default_member_permissions_only_allow_reading_team_keys(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        member: Final = _member(scenario, None)
        generated: Final = gateway.request("POST", "/key/generate", {"team_id": member.team_id}, key=member.member_key)
        _refused(generated, 401, PERMISSION_ERROR)
        updated: Final = gateway.request(
            "POST",
            "/key/update",
            {"key": member.team_key, "team_id": "ATTACKER_TEAM_ID", "metadata": {"owner": "member"}},
            key=member.member_key,
        )
        _refused(updated, 401, PERMISSION_ERROR)
        _refused(gateway.request("POST", "/key/delete", {"keys": [member.team_key]}, key=member.member_key), 403)
        _refused(gateway.request("POST", "/key/regenerate", {"key": member.team_key}, key=member.member_key), 401)
        info: Final = gateway.request("GET", "/key/info", key=member.member_key, params={"key": member.team_key})
        assert info.status_code == 200, info.text
        assert object_value(info.json()["info"])["team_id"] == member.team_id
        assert _team_key_row(member.team_key) == [{"team_id": member.team_id, "metadata": {"owner": "team"}}]
        assert _team_key_count(member.team_id) == 1


def test_update_and_delete_permissions_let_a_member_edit_but_not_create_delete_or_regenerate(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        member: Final = _member(scenario, ["/key/update", "/key/delete", "/key/info"])
        updated: Final = gateway.request(
            "POST",
            "/key/update",
            {"key": member.team_key, "team_id": member.team_id, "metadata": {"owner": "member"}},
            key=member.member_key,
        )
        assert updated.status_code == 200, updated.text
        assert _team_key_row(member.team_key) == [{"team_id": member.team_id, "metadata": {"owner": "member"}}]
        _refused(gateway.request("POST", "/key/delete", {"keys": [member.team_key]}, key=member.member_key), 403)
        generated: Final = gateway.request("POST", "/key/generate", {"team_id": member.team_id}, key=member.member_key)
        _refused(generated, 401, PERMISSION_ERROR)
        regenerated: Final = gateway.request(
            "POST", "/key/regenerate", {"key": member.team_key, "team_id": member.team_id}, key=member.member_key
        )
        _refused(regenerated, 401)
        assert _team_key_count(member.team_id) == 1


def test_generate_permission_lets_a_member_create_team_keys_but_not_change_existing_ones(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        member: Final = _member(scenario, ["/key/generate"])
        generated: Final = gateway.request("POST", "/key/generate", {"team_id": member.team_id}, key=member.member_key)
        assert generated.status_code == 200, generated.text
        created: Final = string_value(generated.json()["key"])
        scenario.cleanups.callback(scenario.delete_key, created)
        assert _team_key_row(created) == [{"team_id": member.team_id, "metadata": {}}]
        updated: Final = gateway.request(
            "POST",
            "/key/update",
            {"key": member.team_key, "team_id": member.team_id, "metadata": {"owner": "member"}},
            key=member.member_key,
        )
        _refused(updated, 401, PERMISSION_ERROR)
        assert _team_key_row(member.team_key) == [{"team_id": member.team_id, "metadata": {"owner": "team"}}]
        _refused(gateway.request("POST", "/key/delete", {"keys": [member.team_key]}, key=member.member_key), 403)
        regenerated: Final = gateway.request(
            "POST", "/key/regenerate", {"key": member.team_key, "team_id": member.team_id}, key=member.member_key
        )
        _refused(regenerated, 401, PERMISSION_ERROR)
        assert _team_key_count(member.team_id) == 2
