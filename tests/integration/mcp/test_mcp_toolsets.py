import uuid
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway, Scenario, object_value
from integration._support.mcp import INITIALIZE, Outcome, _outcome_from_rpc, mcp_peer, register_mcp, tool_calls
from integration._support.mcp_grants import create_toolset

from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken

ADD: Final = {"a": 4, "b": 5}


def _dashboard_ui_session_token(user_id: str) -> str:
    user: Final = LiteLLM_UserTable(user_id=user_id, user_role="internal_user", models=[])
    return ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(user)


def _toolset(scenario: Scenario, server_id: str, tool: str) -> tuple[str, str]:
    name: Final = "lit6029_" + uuid.uuid4().hex[:10]
    return create_toolset(scenario, ((server_id, tool),), toolset_name=name), name


def _toolset_rpc(
    gateway: Gateway, headers: dict[str, str], name: str, method: str, params: dict[str, object]
) -> Outcome:
    def post(rpc_method: str, rpc_params: dict[str, object]) -> httpx.Response:
        return gateway.client.post(
            f"/toolset/{name}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": rpc_method, "params": rpc_params},
            headers={**headers, "Accept": "application/json, text/event-stream"},
        )

    initialized: Final = _outcome_from_rpc(post("initialize", dict(INITIALIZE)))
    if not initialized.ok:
        return initialized
    return _outcome_from_rpc(post(method, params))


def _listed_toolset_ids(gateway: Gateway, headers: dict[str, str]) -> tuple[str, ...]:
    response: Final = gateway.client.get("/v1/mcp/toolset", headers=headers)
    assert response.status_code == 200, response.text
    return tuple(toolset["toolset_id"] for toolset in response.json())


def _assert_team_grants_only(gateway: Gateway, team_id: str, key: str, toolset_id: str) -> None:
    team: Final = object_value(gateway.get("/team/info", {"team_id": team_id})["team_info"])
    assert object_value(team["object_permission"])["mcp_toolsets"] == [toolset_id], team
    key_info: Final = object_value(gateway.get("/key/info", {"key": key})["info"])
    assert key_info.get("object_permission") is None, f"key must carry no grant of its own: {key_info}"


def test_team_granted_toolset_is_listed_and_served_to_a_team_key(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029_" + uuid.uuid4().hex[:8]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        withheld_id, withheld_name = _toolset(scenario, server_id, "multiply")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id]})
        key: Final = scenario.key(team_id=team_id)
        _assert_team_grants_only(gateway, team_id, key, granted_id)
        headers: Final = {"Authorization": f"Bearer {key}"}

        assert _listed_toolset_ids(gateway, headers) == (granted_id,)
        detail: Final = gateway.client.get(f"/v1/mcp/toolset/{granted_id}", headers=headers)
        assert detail.status_code == 200, detail.text
        assert detail.json()["toolset_name"] == granted_name, detail.text
        withheld_detail: Final = gateway.client.get(f"/v1/mcp/toolset/{withheld_id}", headers=headers)
        assert withheld_detail.status_code == 403, withheld_detail.text

        listed: Final = _toolset_rpc(gateway, headers, granted_name, "tools/list", {})
        assert listed.ok, listed.raw
        assert listed.tools == (f"{alias}-add",), listed.raw
        peer.drain()
        called: Final = _toolset_rpc(
            gateway, headers, granted_name, "tools/call", {"name": f"{alias}-add", "arguments": ADD}
        )
        assert called.ok and called.text == "9", called.raw
        assert len(tool_calls(peer.drain())) == 1
        denied: Final = _toolset_rpc(gateway, headers, withheld_name, "tools/list", {})
        assert denied.status == 403, denied.raw


def test_dashboard_session_of_a_team_member_lists_the_team_granted_toolset(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029_" + uuid.uuid4().hex[:8]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        _toolset(scenario, server_id, "multiply")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id]})
        user_id: Final = scenario.user(user_role="internal_user", teams=[team_id])
        user: Final = object_value(gateway.get("/user/info", {"user_id": user_id})["user_info"])
        assert user["teams"] == [team_id], user
        headers: Final = {"Authorization": f"Bearer {_dashboard_ui_session_token(user_id)}"}

        assert _listed_toolset_ids(gateway, headers) == (granted_id,)
        detail: Final = gateway.client.get(f"/v1/mcp/toolset/{granted_id}", headers=headers)
        assert detail.status_code == 200, detail.text
        assert detail.json()["toolset_name"] == granted_name, detail.text


def test_direct_grants_no_grants_and_admin_listing_are_unchanged_by_team_resolution(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029_" + uuid.uuid4().hex[:8]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        withheld_id, withheld_name = _toolset(scenario, server_id, "multiply")
        direct: Final = {"Authorization": f"Bearer {scenario.key(object_permission={'mcp_toolsets': [granted_id]})}"}
        ungranted_team: Final = scenario.team()
        no_grant: Final = {"Authorization": f"Bearer {scenario.key(team_id=ungranted_team)}"}
        admin: Final = {"Authorization": f"Bearer {gateway.key}"}

        assert _listed_toolset_ids(gateway, direct) == (granted_id,)
        assert _toolset_rpc(gateway, direct, granted_name, "tools/list", {}).tools == (f"{alias}-add",)
        assert _toolset_rpc(gateway, direct, withheld_name, "tools/list", {}).status == 403
        assert gateway.client.get(f"/v1/mcp/toolset/{withheld_id}", headers=direct).status_code == 403

        assert _listed_toolset_ids(gateway, no_grant) == ()
        assert gateway.client.get(f"/v1/mcp/toolset/{granted_id}", headers=no_grant).status_code == 403
        assert _toolset_rpc(gateway, no_grant, granted_name, "tools/list", {}).status == 403

        assert {granted_id, withheld_id} <= set(_listed_toolset_ids(gateway, admin))
        assert gateway.client.get(f"/v1/mcp/toolset/{withheld_id}", headers=admin).status_code == 200
        assert _toolset_rpc(gateway, admin, withheld_name, "tools/list", {}).tools == (f"{alias}-multiply",)


def test_a_key_with_its_own_toolset_grant_does_not_inherit_the_team_toolset(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029_" + uuid.uuid4().hex[:8]
        server_id: Final = register_mcp(scenario, peer, alias)
        own_id, own_name = _toolset(scenario, server_id, "add")
        team_only_id, team_only_name = _toolset(scenario, server_id, "multiply")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [own_id, team_only_id]})
        key: Final = scenario.key(team_id=team_id, object_permission={"mcp_toolsets": [own_id]})
        headers: Final = {"Authorization": f"Bearer {key}"}

        assert _listed_toolset_ids(gateway, headers) == (own_id,)
        assert gateway.client.get(f"/v1/mcp/toolset/{team_only_id}", headers=headers).status_code == 403
        assert _toolset_rpc(gateway, headers, team_only_name, "tools/list", {}).status == 403
        assert _toolset_rpc(gateway, headers, own_name, "tools/list", {}).tools == (f"{alias}-add",)


def _team_member_with_own_grant(scenario: Scenario, team_id: str, own_server_id: str) -> str:
    user_id: Final = scenario.user(user_role="internal_user", object_permission={"mcp_servers": [own_server_id]})
    scenario.gateway.post("/team/member_add", {"team_id": team_id, "member": {"role": "user", "user_id": user_id}})
    return user_id


def test_dashboard_session_serves_the_team_toolset_despite_a_disjoint_grant_on_the_user_row(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The member's own row grants a different server outright. That grant must not cap the team's toolset
    to nothing, and the team's sibling toolset must not leak onto the granted toolset's route."""
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029_" + uuid.uuid4().hex[:8]
        server_id: Final = register_mcp(scenario, peer, alias)
        own_server_id: Final = register_mcp(scenario, peer, "lit6029_own_" + uuid.uuid4().hex[:8])
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        sibling_id, sibling_name = _toolset(scenario, server_id, "multiply")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id, sibling_id]})
        user_id: Final = _team_member_with_own_grant(scenario, team_id, own_server_id)
        headers: Final = {"Authorization": f"Bearer {_dashboard_ui_session_token(user_id)}"}

        assert set(_listed_toolset_ids(gateway, headers)) == {granted_id, sibling_id}
        listed: Final = _toolset_rpc(gateway, headers, granted_name, "tools/list", {})
        assert listed.ok, listed.raw
        assert listed.tools == (f"{alias}-add",), listed.raw
        assert _toolset_rpc(gateway, headers, sibling_name, "tools/list", {}).tools == (f"{alias}-multiply",)
        peer.drain()
        called: Final = _toolset_rpc(
            gateway, headers, granted_name, "tools/call", {"name": f"{alias}-add", "arguments": ADD}
        )
        assert called.ok and called.text == "9", called.raw
        assert len(tool_calls(peer.drain())) == 1
        stranger: Final = {
            "Authorization": f"Bearer {_dashboard_ui_session_token(scenario.user(user_role='internal_user'))}"
        }
        assert _toolset_rpc(gateway, stranger, granted_name, "tools/list", {}).status == 403


def test_a_member_removed_from_the_team_loses_its_toolset_on_the_dashboard_session(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029_" + uuid.uuid4().hex[:8]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id]})
        user_id: Final = scenario.member(team_id)
        headers: Final = {"Authorization": f"Bearer {_dashboard_ui_session_token(user_id)}"}
        assert _listed_toolset_ids(gateway, headers) == (granted_id,)
        assert _toolset_rpc(gateway, headers, granted_name, "tools/list", {}).tools == (f"{alias}-add",)

        gateway.post("/team/member_delete", {"team_id": team_id, "user_id": user_id})

        assert _listed_toolset_ids(gateway, headers) == ()
        assert gateway.client.get(f"/v1/mcp/toolset/{granted_id}", headers=headers).status_code == 403
        assert _toolset_rpc(gateway, headers, granted_name, "tools/list", {}).status == 403
