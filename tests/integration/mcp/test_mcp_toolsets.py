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
