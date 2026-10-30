from typing import Final, Literal

import pytest
from integration._support.client import Gateway, eventually, object_value, string_value, team_admin_permissions
from pydantic import JsonValue

POLICY: Final[dict[str, JsonValue]] = {
    "mcp_servers": [{"name": "github", "url": "https://mcp.example.com/github", "tools": ["search_code"]}]
}


@pytest.mark.parametrize("scope", ("key", "team"))
@pytest.mark.parametrize("nested", (False, True), ids=("first-class", "metadata"))
def test_native_mcp_policy_create_update_and_remove_roundtrip(
    gateway: Gateway, peer: Gateway, scope: Literal["key", "team"], nested: bool
) -> None:
    initial: Final[dict[str, JsonValue]] = (
        {"metadata": {"native_mcp_policy": POLICY}} if nested else {"native_mcp_policy": POLICY}
    )
    with gateway.scenario() as scenario:
        identity: Final = scenario.key(**initial) if scope == "key" else scenario.team(**initial)
        field: Final = "key" if scope == "key" else "team_id"

        def stored() -> JsonValue:
            response: Final = peer.get(f"/{scope}/info", {field: identity})
            info: Final = object_value(response["info" if scope == "key" else "team_info"])
            return object_value(info.get("metadata") or {}).get("native_mcp_policy")

        assert eventually(stored, lambda value: value == POLICY) == POLICY
        gateway.post(f"/{scope}/update", {field: identity, "native_mcp_policy": {}})
        assert eventually(stored, lambda value: value == {"mcp_servers": []}) == {"mcp_servers": []}
        gateway.post(f"/{scope}/update", {field: identity, "native_mcp_policy": None})
        assert eventually(stored, lambda value: value is None) is None


@pytest.mark.parametrize("scope", ("key", "team"))
@pytest.mark.parametrize("nested", (False, True), ids=("first-class", "metadata"))
def test_native_mcp_policy_invalid_update_preserves_stored_grant(
    gateway: Gateway, scope: Literal["key", "team"], nested: bool
) -> None:
    invalid: Final[dict[str, JsonValue]] = {
        "mcp_servers": [{"name": "github", "url": "https://user:password@example.com/mcp", "tools": ["search_code"]}]
    }
    with gateway.scenario() as scenario:
        identity: Final = (
            scenario.key(native_mcp_policy=POLICY) if scope == "key" else scenario.team(native_mcp_policy=POLICY)
        )
        field: Final = "key" if scope == "key" else "team_id"
        update: Final[dict[str, JsonValue]] = (
            {"metadata": {"native_mcp_policy": invalid}} if nested else {"native_mcp_policy": invalid}
        )
        response: Final = gateway.request("POST", f"/{scope}/update", {field: identity, **update})
        assert response.status_code in (400, 422), response.text
        saved: Final = gateway.get(f"/{scope}/info", {field: identity})
        info: Final = object_value(saved["info" if scope == "key" else "team_info"])
        assert object_value(info["metadata"])["native_mcp_policy"] == POLICY


@pytest.mark.parametrize(
    "update", ({"native_mcp_policy": {}}, {"metadata": {"native_mcp_policy": {}}}, {"metadata": {}})
)
@pytest.mark.parametrize("role", ("admin", "user"), ids=("team-admin", "key-owner"))
def test_non_proxy_admin_cannot_change_or_remove_key_policy(
    gateway: Gateway, update: dict[str, JsonValue], role: str
) -> None:
    with gateway.scenario() as scenario:
        team: Final = scenario.team(team_member_permissions=["/key/update", "/key/info"])
        user: Final = scenario.member(team, role=role)
        caller: Final = scenario.key(user_id=user, team_id=team)
        key: Final = scenario.key(user_id=user, team_id=team, native_mcp_policy=POLICY)
        unchanged: Final = gateway.request(
            "POST", "/key/update", {"key": key, "key_alias": "policy-preserved"}, key=caller
        )
        assert unchanged.status_code == 200, unchanged.text
        denied: Final = gateway.request("POST", "/key/update", {"key": key, **update}, key=caller)
        assert denied.status_code == 403, denied.text
        assert "native_mcp_policy" in denied.text
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert object_value(info["metadata"])["native_mcp_policy"] == POLICY


def test_native_mcp_policy_can_roundtrip_key_creation_response(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        created: Final = gateway.post("/key/generate", {"metadata": {"native_mcp_policy": POLICY}})
        key: Final = string_value(created["key"])
        scenario.cleanups.callback(scenario.delete_key, key)
        echo: Final = {name: created[name] for name in ("metadata", "native_mcp_policy") if name in created}
        response: Final = gateway.request("POST", "/key/update", {"key": key, **echo})
        assert response.status_code == 200, response.text
        info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
        assert object_value(info["metadata"])["native_mcp_policy"] == POLICY


@pytest.mark.parametrize("nested", (False, True), ids=("first-class", "metadata"))
def test_team_admin_can_edit_metadata_while_preserving_empty_policy(gateway: Gateway, nested: bool) -> None:
    with team_admin_permissions(gateway, ["metadata"]), gateway.scenario() as scenario:
        team: Final = scenario.team(metadata={"native_mcp_policy": POLICY, "keep": "original"})
        caller: Final = scenario.key(user_id=scenario.member(team, role="admin"), team_id=team)
        unchanged: Final = gateway.request(
            "POST",
            "/team/update",
            {"team_id": team, "metadata": {"native_mcp_policy": POLICY, "keep": "original", "note": "control"}},
            key=caller,
        )
        assert unchanged.status_code == 200, unchanged.text
        gateway.post("/team/update", {"team_id": team, "native_mcp_policy": {}, "metadata": {"keep": "original"}})
        info: Final = object_value(gateway.get("/team/info", {"team_id": team})["team_info"])
        metadata: Final = object_value(info["metadata"])
        update: Final[dict[str, JsonValue]] = (
            {"metadata": {**metadata, "note": "unrelated"}}
            if nested
            else {"native_mcp_policy": {}, "metadata": {"keep": "original", "note": "unrelated"}}
        )
        response: Final = gateway.request(
            "POST", "/team/update", {"team_id": team, **update}, key=caller
        )
        assert response.status_code == 200, response.text
        saved: Final = object_value(gateway.get("/team/info", {"team_id": team})["team_info"])
        assert object_value(saved["metadata"]) == {**metadata, "note": "unrelated"}
