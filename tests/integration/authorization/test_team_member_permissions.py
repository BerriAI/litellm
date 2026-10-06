import json
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

import httpx
import pytest
from integration._support.client import (
    Gateway,
    Scenario,
    delete_key_if_present,
    object_value,
    string_value,
)
from integration._support.database import read_rows, write_rows
from pydantic import BaseModel, JsonValue

PERMISSION_ERROR: Final = "team_member_permission_error"


class _ObservedRequest(BaseModel):
    body: dict[str, JsonValue]


class _Observations(BaseModel):
    requests: list[_ObservedRequest]


class _TeamPermissions(BaseModel):
    team_id: str
    team_member_permissions: list[str] | None = None


class _PermissionsList(BaseModel):
    team_id: str
    team_member_permissions: list[str]


class _TeamPermissionRow(BaseModel):
    team_id: str
    team_member_permissions: list[str] | None = None


class _BulkPermissionsUpdate(BaseModel):
    message: str
    teams_updated: int
    permissions_appended: list[str] | None = None


class _GeneratedKey(BaseModel):
    key: str


class _Error(BaseModel):
    type: str


class _ErrorResponse(BaseModel):
    error: _Error


class _HTTPErrorDetail(BaseModel):
    error: str


class _HTTPErrorResponse(BaseModel):
    detail: _HTTPErrorDetail


@pytest.fixture
def upstream(gateway: Gateway) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=gateway.upstream_url, timeout=15, trust_env=False) as client:
        response: Final = client.get("/__observations")
        assert response.status_code == 200, response.text
        yield client


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


def _update_team_key(proxy: Gateway, member: Member, owner: str) -> httpx.Response:
    return proxy.request(
        "POST",
        "/key/update",
        {"key": member.team_key, "team_id": member.team_id, "metadata": {"owner": owner}},
        key=member.member_key,
    )


def _team_key_count(team_id: str) -> int:
    return len(read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE team_id = %s', (team_id,)))


def _refused(response: httpx.Response, status: int, error_type: str | None = None) -> None:
    assert response.status_code == status, response.text
    if error_type is not None:
        assert object_value(response.json()["error"])["type"] == error_type, response.text


def _observed_models(upstream: httpx.Client) -> tuple[str, ...]:
    response: Final = upstream.get("/__observations")
    assert response.status_code == 200, response.text
    observations: Final = _Observations.model_validate_json(response.text)
    return tuple(string_value(request.body["model"]) for request in observations.requests)


def _permissions(gateway: Gateway, team_id: str) -> _PermissionsList:
    response: Final = gateway.request("GET", "/team/permissions_list", params={"team_id": team_id})
    assert response.status_code == 200, response.text
    return _PermissionsList.model_validate_json(response.text)


def _team_permission_rows(team_ids: tuple[str, ...]) -> tuple[_TeamPermissionRow, ...]:
    return tuple(
        _TeamPermissionRow.model_validate(row)
        for row in read_rows(
            'SELECT team_id, team_member_permissions FROM "LiteLLM_TeamTable" WHERE team_id = ANY(%s) ORDER BY team_id',
            (list(team_ids),),
        )
    )


def _restore_team_permissions(team_id: str, permissions: list[str] | None) -> None:
    if permissions is None:
        write_rows(
            'UPDATE "LiteLLM_TeamTable" SET team_member_permissions = NULL WHERE team_id = %s',
            (team_id,),
        )
        return
    write_rows(
        'UPDATE "LiteLLM_TeamTable" SET team_member_permissions = ARRAY(SELECT jsonb_array_elements_text(%s::jsonb)) WHERE team_id = %s',
        (json.dumps(permissions), team_id),
    )


def test_permissions_update_persists_requested_member_permissions(gateway: Gateway, upstream: httpx.Client) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        requested: Final = ["/key/generate", "/key/info"]
        updated: Final = gateway.request(
            "POST",
            "/team/permissions_update",
            {"team_id": team, "team_member_permissions": requested},
        )
        assert updated.status_code == 200, updated.text
        response: Final = _TeamPermissions.model_validate_json(updated.text)
        assert response.team_id == team, updated.text
        assert response.team_member_permissions == requested, updated.text
        readback: Final = _permissions(gateway, team)
        assert readback.team_member_permissions == requested, repr(readback)
        assert read_rows(
            'SELECT team_member_permissions FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        ) == [{"team_member_permissions": requested}], updated.text
        observed: Final = _observed_models(upstream)
        assert observed == (), repr(observed)


def test_permissions_update_grants_and_revokes_key_generate_for_a_warmed_member_on_both_proxies(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team()
        member: Final = scenario.member(team)
        member_key: Final = scenario.key(user_id=member)
        denied_gateway: Final = gateway.request("POST", "/key/generate", {"team_id": team}, key=member_key)
        _refused(denied_gateway, 401, PERMISSION_ERROR)
        denied_peer: Final = peer.request("POST", "/key/generate", {"team_id": team}, key=member_key)
        _refused(denied_peer, 401, PERMISSION_ERROR)

        granted: Final = gateway.request(
            "POST",
            "/team/permissions_update",
            {"team_id": team, "team_member_permissions": ["/key/generate", "/key/info"]},
        )
        assert granted.status_code == 200, granted.text
        added: Final = _permissions(gateway, team)
        assert added.team_member_permissions == ["/key/generate", "/key/info"], repr(added)
        assert read_rows(
            'SELECT team_member_permissions FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        ) == [{"team_member_permissions": ["/key/generate", "/key/info"]}], granted.text
        generated_gateway: Final = gateway.request("POST", "/key/generate", {"team_id": team}, key=member_key)
        assert generated_gateway.status_code == 200, generated_gateway.text
        gateway_key: Final = _GeneratedKey.model_validate_json(generated_gateway.text).key
        scenario.cleanups.callback(delete_key_if_present, gateway, gateway_key)
        generated_peer: Final = peer.request("POST", "/key/generate", {"team_id": team}, key=member_key)
        assert generated_peer.status_code == 200, generated_peer.text
        peer_key: Final = _GeneratedKey.model_validate_json(generated_peer.text).key
        scenario.cleanups.callback(delete_key_if_present, gateway, peer_key)

        revoked: Final = gateway.request(
            "POST",
            "/team/permissions_update",
            {"team_id": team, "team_member_permissions": ["/key/info"]},
        )
        assert revoked.status_code == 200, revoked.text
        revoked_readback: Final = _permissions(gateway, team)
        assert revoked_readback.team_member_permissions == ["/key/info"], repr(revoked_readback)
        assert read_rows(
            'SELECT team_member_permissions FROM "LiteLLM_TeamTable" WHERE team_id = %s',
            (team,),
        ) == [{"team_member_permissions": ["/key/info"]}], revoked.text
        revoked_gateway: Final = gateway.request("POST", "/key/generate", {"team_id": team}, key=member_key)
        _refused(revoked_gateway, 401, PERMISSION_ERROR)
        revoked_peer: Final = peer.request("POST", "/key/generate", {"team_id": team}, key=member_key)
        _refused(revoked_peer, 401, PERMISSION_ERROR)
        observed: Final = _observed_models(upstream)
        assert observed == (), repr(observed)


def test_permissions_bulk_update_appends_only_to_the_listed_teams(
    gateway: Gateway, peer: Gateway, upstream: httpx.Client
) -> None:
    with gateway.scenario() as scenario:
        listed: Final = _member(scenario, ["/key/info"])
        team_two: Final = scenario.team(team_member_permissions=["/key/info", "/key/update"])
        unlisted: Final = _member(scenario, ["/key/info"])
        team_ids: Final = (listed.team_id, team_two, unlisted.team_id)
        for proxy in (gateway, peer):
            for member in (listed, unlisted):
                _refused(_update_team_key(proxy, member, "member before bulk"), 401, PERMISSION_ERROR)
        before_non_admin: Final = _team_permission_rows(team_ids)
        caller: Final = scenario.user(user_role="internal_user")
        caller_key: Final = scenario.key(user_id=caller, allowed_routes=["/team/permissions_bulk_update"])
        denied: Final = gateway.request(
            "POST",
            "/team/permissions_bulk_update",
            {"permissions": ["/key/update"], "team_ids": [listed.team_id, team_two]},
            key=caller_key,
        )
        assert denied.status_code == 403, denied.text
        assert _HTTPErrorResponse.model_validate_json(denied.text) == _HTTPErrorResponse(
            detail=_HTTPErrorDetail(error="Only proxy admins can bulk-update team permissions")
        ), denied.text
        after_denied: Final = _team_permission_rows(team_ids)
        assert after_denied == before_non_admin, repr(after_denied)

        selected: Final = gateway.request(
            "POST",
            "/team/permissions_bulk_update",
            {"permissions": ["/key/update"], "team_ids": [listed.team_id, team_two]},
        )
        assert selected.status_code == 200, selected.text
        selected_response: Final = _BulkPermissionsUpdate.model_validate_json(selected.text)
        assert selected_response == _BulkPermissionsUpdate(
            message="Team permissions updated successfully",
            teams_updated=1,
            permissions_appended=["/key/update"],
        ), selected.text
        selected_rows: Final = _team_permission_rows(team_ids)
        assert tuple(sorted(selected_rows, key=lambda row: row.team_id)) == tuple(
            sorted(
                (
                    _TeamPermissionRow(team_id=listed.team_id, team_member_permissions=["/key/info", "/key/update"]),
                    _TeamPermissionRow(team_id=team_two, team_member_permissions=["/key/info", "/key/update"]),
                    _TeamPermissionRow(team_id=unlisted.team_id, team_member_permissions=["/key/info"]),
                ),
                key=lambda row: row.team_id,
            )
        ), repr(selected_rows)
        listed_readback: Final = _permissions(gateway, listed.team_id)
        assert listed_readback.team_member_permissions == ["/key/info", "/key/update"], repr(listed_readback)
        unlisted_readback: Final = _permissions(gateway, unlisted.team_id)
        assert unlisted_readback.team_member_permissions == ["/key/info"], repr(unlisted_readback)

        gateway_update: Final = _update_team_key(gateway, listed, "member via gateway")
        assert gateway_update.status_code == 200, gateway_update.text
        assert _team_key_row(listed.team_key) == [
            {"team_id": listed.team_id, "metadata": {"owner": "member via gateway"}}
        ], gateway_update.text
        peer_update: Final = _update_team_key(peer, listed, "member via peer")
        assert peer_update.status_code == 200, peer_update.text
        assert _team_key_row(listed.team_key) == [
            {"team_id": listed.team_id, "metadata": {"owner": "member via peer"}}
        ], peer_update.text
        for proxy in (gateway, peer):
            _refused(_update_team_key(proxy, unlisted, "unlisted member"), 401, PERMISSION_ERROR)
        assert _team_key_row(unlisted.team_key) == [{"team_id": unlisted.team_id, "metadata": {"owner": "team"}}]

        all_before: Final = tuple(
            _TeamPermissionRow.model_validate(row)
            for row in read_rows(
                'SELECT team_id, team_member_permissions FROM "LiteLLM_TeamTable" ORDER BY team_id',
                (),
            )
        )
        for row in all_before:
            if row.team_id not in team_ids and "/key/update" not in (row.team_member_permissions or []):
                scenario.cleanups.callback(_restore_team_permissions, row.team_id, row.team_member_permissions)
        missing: Final = tuple(
            row.team_id for row in all_before if "/key/update" not in (row.team_member_permissions or [])
        )
        all_updated: Final = gateway.request(
            "POST",
            "/team/permissions_bulk_update",
            {"permissions": ["/key/update"], "apply_to_all_teams": True},
        )
        assert all_updated.status_code == 200, all_updated.text
        all_response: Final = _BulkPermissionsUpdate.model_validate_json(all_updated.text)
        assert all_response == _BulkPermissionsUpdate(
            message="Team permissions updated successfully",
            teams_updated=len(missing),
            permissions_appended=["/key/update"],
        ), all_updated.text
        all_after: Final = tuple(
            _TeamPermissionRow.model_validate(row)
            for row in read_rows(
                'SELECT team_id, team_member_permissions FROM "LiteLLM_TeamTable" ORDER BY team_id',
                (),
            )
        )
        assert tuple(row.team_id for row in all_after) == tuple(row.team_id for row in all_before), repr(all_after)
        assert all("/key/update" in (row.team_member_permissions or []) for row in all_after), repr(all_after)
        unlisted_after_all: Final = _update_team_key(peer, unlisted, "unlisted member after all teams")
        assert unlisted_after_all.status_code == 200, unlisted_after_all.text
        assert _team_key_row(unlisted.team_key) == [
            {"team_id": unlisted.team_id, "metadata": {"owner": "unlisted member after all teams"}}
        ], unlisted_after_all.text
        observed: Final = _observed_models(upstream)
        assert observed == (), repr(observed)


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
