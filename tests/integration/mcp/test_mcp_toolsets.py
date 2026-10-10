import re
import secrets
import signal
import textwrap
import uuid
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from pydantic import TypeAdapter
from integration._support.client import Gateway, Scenario, gateway_from_environment, object_value
from integration._support.mcp import (
    INITIALIZE,
    Outcome,
    ScriptedTool,
    _outcome_from_rest,
    _outcome_from_rpc,
    mcp_peer,
    register_mcp,
    scripted_peer,
    text_result,
    tool_calls,
)
from integration._support.mcp_grants import create_toolset
from integration._support.process import owned_proxy, owned_proxy_process

from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import LITELLM_SESSION_TOKEN_PREFIX, ExperimentalUIJWTToken
from litellm.proxy.common_utils.encrypt_decrypt_utils import encrypt_bearer_token

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


def _bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _expired_dashboard_token(user_id: str) -> str:
    expired: Final = (datetime.now(timezone.utc) - timedelta(minutes=5)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    stale: Final = UserAPIKeyAuth(
        token="ui-token",
        key_name="ui-token",
        key_alias="ui-token",
        expires=expired + "+00:00",
        user_id=user_id,
        team_id="litellm-dashboard",
        models=[],
        user_role=LitellmUserRoles.INTERNAL_USER,
    )
    return encrypt_bearer_token(stale.model_dump_json(exclude_none=True), prefix=LITELLM_SESSION_TOKEN_PREFIX)


def _rest_list(gateway: Gateway, headers: dict[str, str], params: object) -> httpx.Response:
    return gateway.client.get("/mcp-rest/tools/list", headers=headers, params=params)


def _rest_call(gateway: Gateway, headers: dict[str, str], name: str, server_id: str) -> Outcome:
    return _outcome_from_rest(
        gateway.client.post(
            "/mcp-rest/tools/call",
            headers=headers,
            json={"name": name, "arguments": dict(ADD), "server_id": server_id},
        )
    )


def _route_tools(gateway: Gateway, headers: dict[str, str], name: str) -> Outcome:
    return _toolset_rpc(gateway, headers, name, "tools/list", {})


def _route_call(gateway: Gateway, headers: dict[str, str], name: str, tool: str) -> Outcome:
    return _toolset_rpc(gateway, headers, name, "tools/call", {"name": tool, "arguments": dict(ADD)})


def _strict_config(directory: Path) -> Path:
    base: Final = yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
    strict: Final = {
        **base,
        "general_settings": {**base.get("general_settings", {}), "require_key_mcp_access_defined": True},
    }
    path: Final = directory / "require_key_mcp_access.yaml"
    path.write_text(yaml.safe_dump(strict))
    return path


def test_mcp_rest_toolset_name_narrows_the_list_and_serves_the_call_for_a_team_key_and_a_dashboard_member(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029rest" + uuid.uuid4().hex[:6]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        _, withheld_name = _toolset(scenario, server_id, "multiply")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id]})
        key: Final = scenario.key(team_id=team_id)
        member: Final = scenario.member(team_id)
        for headers in (_bearer(key), _bearer(_dashboard_ui_session_token(member))):
            listed: Final = _rest_list(gateway, headers, {"toolset_name": granted_name})
            assert listed.status_code == 200, listed.text
            listed_names: Final = tuple(tool["name"] for tool in listed.json()["tools"])
            assert len(listed_names) == 1 and listed_names[0].endswith("add"), listed.text
            denied: Final = _rest_list(gateway, headers, {"toolset_name": withheld_name})
            assert denied.status_code == 200 and denied.json()["tools"] == [], denied.text
            assert "does not have access to toolset" in denied.json()["message"], denied.text
            peer.drain()
            called: Final = _rest_call(gateway, headers, listed_names[0], server_id)
            assert called.ok and called.text == "9", called.raw
            assert len(tool_calls(peer.drain())) == 1


def test_a_key_restricted_to_its_own_servers_does_not_inherit_the_team_toolset(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029own" + uuid.uuid4().hex[:6]
        server_id: Final = register_mcp(scenario, peer, alias)
        other_id: Final = register_mcp(scenario, peer, "lit6029other" + uuid.uuid4().hex[:6])
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id], "mcp_servers": [other_id]})
        key: Final = scenario.key(team_id=team_id, object_permission={"mcp_servers": [other_id]})
        headers: Final = _bearer(key)
        assert _listed_toolset_ids(gateway, headers) == ()
        assert gateway.client.get(f"/v1/mcp/toolset/{granted_id}", headers=headers).status_code == 403
        assert _route_tools(gateway, headers, granted_name).status == 403
        assert tool_calls(peer.drain()) == ()


def test_a_team_with_an_empty_or_absent_toolset_grant_gives_its_keys_nothing(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        server_id: Final = register_mcp(scenario, peer, "lit6029empty" + uuid.uuid4().hex[:6])
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        teams: Final = (
            scenario.team(object_permission={"mcp_toolsets": []}),
            scenario.team(object_permission={"mcp_toolsets": None}),
            scenario.team(),
        )
        for team_id in teams:
            headers: Final = _bearer(scenario.key(team_id=team_id))
            assert _listed_toolset_ids(gateway, headers) == (), team_id
            assert gateway.client.get(f"/v1/mcp/toolset/{granted_id}", headers=headers).status_code == 403
            assert _route_tools(gateway, headers, granted_name).status == 403, team_id
        assert tool_calls(peer.drain()) == ()


def test_an_unknown_or_malformed_toolset_name_is_refused_without_peer_traffic_and_the_route_keeps_serving(
    gateway: Gateway,
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029bad" + uuid.uuid4().hex[:6]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        _, withheld_name = _toolset(scenario, server_id, "multiply")
        headers: Final = _bearer(scenario.key(team_id=scenario.team(object_permission={"mcp_toolsets": [granted_id]})))
        unknown: Final = "missing" + uuid.uuid4().hex[:8]
        assert _route_tools(gateway, headers, unknown).status == 404
        assert _rest_list(gateway, headers, {"toolset_name": unknown}).status_code == 404
        malformed: Final = (
            [("toolset_name", granted_name), ("toolset_name", withheld_name)],
            {"toolset_name": "x" * 5000},
            {"toolset_name": ""},
            {"toolset_name": granted_name + "\x00"},
        )
        statuses: Final = tuple(_rest_list(gateway, headers, params).status_code for params in malformed)
        assert all(status < 500 for status in statuses), statuses
        assert tool_calls(peer.drain()) == ()
        assert gateway.client.get("/health/liveliness").status_code == 200
        served: Final = _route_tools(gateway, headers, granted_name)
        assert served.tools == (f"{alias}-add",), served.raw


def test_garbage_expired_and_tampered_credentials_are_refused_on_every_toolset_surface(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with mcp_peer() as peer, gateway.scenario() as scenario:
        server_id: Final = register_mcp(scenario, peer, "lit6029cred" + uuid.uuid4().hex[:6])
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id]})
        member: Final = scenario.member(team_id)
        forged: Final = (
            "sk-" + secrets.token_urlsafe(24),
            _expired_dashboard_token(member),
            "llm_session_" + secrets.token_urlsafe(32),
        )
        for bearer in forged:
            headers: Final = _bearer(bearer)
            listed: Final = gateway.client.get("/v1/mcp/toolset", headers=headers)
            assert listed.status_code == 401, (bearer[:12], listed.text)
            detail: Final = gateway.client.get(f"/v1/mcp/toolset/{granted_id}", headers=headers)
            assert detail.status_code == 401, (bearer[:12], detail.text)
            routed: Final = _route_tools(gateway, headers, granted_name)
            assert routed.status == 401, (bearer[:12], routed.raw)
            rest: Final = _rest_list(gateway, headers, {"toolset_name": granted_name})
            assert rest.status_code == 401, (bearer[:12], rest.text)
        assert peer.drain() == ()


def test_a_dashboard_member_of_a_deleted_team_loses_the_toolset_while_a_direct_grant_survives(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029gone" + uuid.uuid4().hex[:6]
        server_id: Final = register_mcp(scenario, peer, alias)
        team_id_, team_name = _toolset(scenario, server_id, "add")
        own_id, own_name = _toolset(scenario, server_id, "multiply")
        doomed: Final = scenario.gateway.post(
            "/team/new",
            {"team_alias": f"integration-{uuid.uuid4().hex}", "object_permission": {"mcp_toolsets": [team_id_]}},
        )
        doomed_team: Final = str(doomed["team_id"])
        member: Final = scenario.user(user_role="internal_user", teams=[doomed_team])
        granted: Final = scenario.user(
            user_role="internal_user", teams=[doomed_team], object_permission={"mcp_toolsets": [own_id]}
        )
        member_headers: Final = _bearer(_dashboard_ui_session_token(member))
        granted_headers: Final = _bearer(_dashboard_ui_session_token(granted))
        assert _listed_toolset_ids(gateway, member_headers) == (team_id_,)
        assert set(_listed_toolset_ids(gateway, granted_headers)) == {team_id_, own_id}
        scenario.delete_team(doomed_team)
        assert _listed_toolset_ids(gateway, member_headers) == ()
        assert gateway.client.get(f"/v1/mcp/toolset/{team_id_}", headers=member_headers).status_code == 403
        assert _route_tools(gateway, member_headers, team_name).status == 403
        assert _listed_toolset_ids(gateway, granted_headers) == (own_id,)
        assert _route_tools(gateway, granted_headers, team_name).status == 403
        assert _route_tools(gateway, granted_headers, own_name).tools == (f"{alias}-multiply",)
        peer.drain()
        kept: Final = _route_call(gateway, granted_headers, own_name, f"{alias}-multiply")
        assert kept.ok and kept.text == "20", kept.raw
        assert [call["body"]["params"]["name"] for call in tool_calls(peer.drain())] == ["multiply"]


def test_require_key_mcp_access_defined_stops_key_inheritance_but_not_the_dashboard_member(
    gateway: Gateway, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with (
        owned_proxy(gateway, tmp_path, {}, config=_strict_config(tmp_path), workers=2) as strict,
        mcp_peer() as peer,
        strict.scenario() as scenario,
    ):
        alias: Final = "lit6029strict" + uuid.uuid4().hex[:6]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id]})
        inheriting: Final = _bearer(scenario.key(team_id=team_id))
        own: Final = _bearer(scenario.key(team_id=team_id, object_permission={"mcp_toolsets": [granted_id]}))
        member: Final = _bearer(_dashboard_ui_session_token(scenario.member(team_id)))
        assert _listed_toolset_ids(strict, inheriting) == ()
        assert _route_tools(strict, inheriting, granted_name).status == 403
        assert _listed_toolset_ids(strict, own) == (granted_id,)
        own_tools: Final = _route_tools(strict, own, granted_name)
        assert own_tools.tools == (f"{alias}-add",), (own_tools.status, own_tools.raw)
        assert _listed_toolset_ids(strict, member) == (granted_id,)
        member_tools: Final = _route_tools(strict, member, granted_name)
        assert member_tools.tools == (f"{alias}-add",), (member_tools.status, member_tools.raw)
        peer.drain()
        called: Final = _route_call(strict, member, granted_name, f"{alias}-add")
        assert called.ok and called.text == "9", called.raw
        assert len(tool_calls(peer.drain())) == 1


def test_a_key_with_only_a_vector_store_grant_still_inherits_the_team_toolset(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029vs" + uuid.uuid4().hex[:6]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id]})
        headers: Final = _bearer(
            scenario.key(team_id=team_id, object_permission={"vector_stores": ["vs-" + uuid.uuid4().hex[:8]]})
        )
        assert _listed_toolset_ids(gateway, headers) == (granted_id,)
        assert _route_tools(gateway, headers, granted_name).tools == (f"{alias}-add",)
        peer.drain()
        called: Final = _route_call(gateway, headers, granted_name, f"{alias}-add")
        assert called.ok and called.text == "9", called.raw
        assert len(tool_calls(peer.drain())) == 1


def test_a_member_added_after_the_team_was_cached_sees_the_toolset_on_every_following_request(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with mcp_peer() as peer, gateway.scenario() as scenario:
        server_id: Final = register_mcp(scenario, peer, "lit6029cache" + uuid.uuid4().hex[:6])
        granted_id, _ = _toolset(scenario, server_id, "add")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id]})
        user_id: Final = scenario.user(user_role="internal_user")
        headers: Final = _bearer(_dashboard_ui_session_token(user_id))
        warm: Final = _bearer(scenario.key(team_id=team_id))
        assert tuple(_listed_toolset_ids(gateway, warm) for _ in range(4)) == ((granted_id,),) * 4
        assert tuple(_listed_toolset_ids(gateway, headers) for _ in range(4)) == ((),) * 4
        added: Final = gateway.request(
            "POST", "/team/member_add", {"team_id": team_id, "member": {"user_id": user_id, "role": "user"}}
        )
        assert added.status_code == 200, added.text
        listings: Final = tuple(_listed_toolset_ids(gateway, headers) for _ in range(8))
        assert listings == ((granted_id,),) * 8, listings


def test_a_member_of_two_teams_sees_the_union_and_each_route_stays_narrowed_to_its_own_toolset(
    gateway: Gateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029two" + uuid.uuid4().hex[:6]
        server_id: Final = register_mcp(scenario, peer, alias)
        first_id, first_name = _toolset(scenario, server_id, "add")
        second_id, second_name = _toolset(scenario, server_id, "multiply")
        teams: Final = (
            scenario.team(object_permission={"mcp_toolsets": [first_id]}),
            scenario.team(object_permission={"mcp_toolsets": [second_id]}),
        )
        headers: Final = _bearer(
            _dashboard_ui_session_token(scenario.user(user_role="internal_user", teams=list(teams)))
        )
        assert set(_listed_toolset_ids(gateway, headers)) == {first_id, second_id}
        assert _route_tools(gateway, headers, first_name).tools == (f"{alias}-add",)
        assert _route_tools(gateway, headers, second_name).tools == (f"{alias}-multiply",)
        peer.drain()
        crossed: Final = _route_call(gateway, headers, first_name, f"{alias}-multiply")
        assert not crossed.ok, crossed.raw
        assert tool_calls(peer.drain()) == ()


def test_owned_proxy_loss_retains_exit_status_and_first_http_failure(gateway: Gateway, tmp_path: Path) -> None:
    with pytest.raises(httpx.TransportError) as failure:  # noqa: PT012  # HTTP failure must propagate through proxy cleanup
        with owned_proxy_process(gateway, tmp_path, {}) as owned:
            response: Final = owned.gateway.client.get("/health/readiness")
            assert response.status_code == 200, response.text
            owned.process.kill()
            owned.process.wait(timeout=3)
            owned.gateway.client.get("/health/readiness")
    record: Final = TypeAdapter(dict[str, int | float | str | None]).validate_json(
        owned.log.with_suffix(".exit.json").read_text()
    )
    assert record["root_pid"] == owned.process.pid
    assert record["exit_before_stop"] == -signal.SIGKILL
    assert record["exit_after_stop"] == -signal.SIGKILL
    assert record["exception_before_stop"] == failure.type.__name__
    assert record["exception_at_record"] == failure.type.__name__


_PROBE: Final = "catalog-probe"
_ECHO: Final = "catalog-echo"
_UNLISTED: Final = ""
_GUARDRAIL_CODE: Final = (
    "def apply_guardrail(inputs, request_data, input_type):\n"
    f'    if "{_PROBE}" not in list(inputs.get("texts") or []):\n'
    "        return allow()\n"
    '    function = inputs.get("tools", [{}])[0].get("function", {})\n'
    f'    return block("{_ECHO}[" + function.get("description") + "]")\n'
)


_ECHO_GUARDRAIL_YAML: Final = (
    "guardrails:\n"
    "  - guardrail_name: catalog-echo\n"
    "    litellm_params:\n"
    "      guardrail: custom_code\n"
    "      mode: pre_mcp_call\n"
    "      default_on: true\n"
    "      custom_code: |\n" + textwrap.indent(_GUARDRAIL_CODE, 8 * " ")
)


@pytest.fixture(scope="module")
def echo_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("catalog-echo")
    path: Final = directory / "catalog_echo.yaml"
    path.write_text((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text() + _ECHO_GUARDRAIL_YAML)
    with gateway_from_environment() as gateway, owned_proxy(gateway, directory, {}, config=path, workers=2) as rig:
        yield rig


def _echoed_description(outcome: Outcome) -> str:
    found: Final = re.search(rf"{_ECHO}\[(.*?)\]", outcome.raw)
    assert found is not None, outcome.raw
    return found.group(1)


def test_a_team_keys_toolset_route_listing_feeds_its_own_calls_but_not_a_team_mates(echo_rig: Gateway) -> None:
    described: Final = "Adds for the team " + uuid.uuid4().hex[:8]
    tool: Final = ScriptedTool("add", lambda _: text_result("9"), description=described)
    with scripted_peer(tool) as peer, echo_rig.scenario() as scenario:
        alias: Final = "lit6029echo" + uuid.uuid4().hex[:6]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_id, granted_name = _toolset(scenario, server_id, "add")
        team_id: Final = scenario.team(object_permission={"mcp_toolsets": [granted_id]})
        key: Final = scenario.key(team_id=team_id)
        team_mate: Final = scenario.key(team_id=team_id)
        _assert_team_grants_only(echo_rig, team_id, key, granted_id)
        listed: Final = _toolset_rpc(echo_rig, _bearer(key), granted_name, "tools/list", {})
        assert listed.ok and listed.tools == (f"{alias}-add",), listed.raw
        probe: Final[dict[str, object]] = {"name": f"{alias}-add", "arguments": {"probe": _PROBE}}
        own: Final = _echoed_description(_toolset_rpc(echo_rig, _bearer(key), granted_name, "tools/call", probe))
        mate: Final = _echoed_description(_toolset_rpc(echo_rig, _bearer(team_mate), granted_name, "tools/call", probe))
        assert (own, mate) == (described, _UNLISTED), (
            "the slot is keyed by the hashed key, so a team-mate that never listed is handed nothing"
        )
        assert tool_calls(peer.drain()) == (), "a blocked probe reached the peer"
