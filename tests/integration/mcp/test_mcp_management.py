import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.mcp import (
    McpCaller,
    call_tool,
    delete_mcp,
    forget_mcp,
    mcp_peer,
    register_mcp,
    tool_calls,
    tool_names,
)
from integration._support.process import owned_proxy

from litellm.models.user import LiteLLM_UserTable
from litellm.proxy.auth.auth_checks import ExperimentalUIJWTToken

ADD: Final = {"a": 4, "b": 5}


def _dashboard_ui_session_token(user_id: str) -> str:
    user: Final = LiteLLM_UserTable(user_id=user_id, user_role="internal_user", models=[])
    return ExperimentalUIJWTToken.get_experimental_ui_login_jwt_auth_token(user)


def _servers(gateway: Gateway, key: str | None = None) -> dict[str, dict[str, object]]:
    response: Final = gateway.client.get("/v1/mcp/server", headers={"x-litellm-api-key": key or gateway.key})
    assert response.status_code == 200, response.text
    return {server["server_id"]: server for server in response.json()}


def test_non_admin_key_cannot_create_edit_or_delete_servers(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "mgmt" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        plain: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        headers: Final = {"x-litellm-api-key": plain}
        created: Final = gateway.client.post(
            "/v1/mcp/server",
            json={"server_name": alias + "x", "alias": alias + "x", **peer.registration()},
            headers=headers,
        )
        assert created.status_code == 403, created.text
        edited: Final = gateway.client.put(
            "/v1/mcp/server", json={"server_id": identity, "server_name": "hijacked"}, headers=headers
        )
        assert edited.status_code == 403, edited.text
        deleted: Final = gateway.client.delete(f"/v1/mcp/server/{identity}", headers=headers)
        assert deleted.status_code == 403, deleted.text
        assert _servers(gateway)[identity]["server_name"] == alias
        assert call_tool(gateway, plain, identity, tool_names(gateway, plain, identity)["add"], ADD).status_code == 200


def test_secrets_never_appear_in_server_listing_or_detail(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        secret: Final = "shh-" + uuid.uuid4().hex
        header_secret: Final = "hdr-" + uuid.uuid4().hex
        alias: Final = "mgmt" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario,
            peer,
            alias,
            auth_type="bearer_token",
            credentials={"auth_value": secret},
            static_headers={"X-Integration-Secret": header_secret},
        )
        viewer: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        for key in (gateway.key, viewer):
            listing: Final = gateway.client.get("/v1/mcp/server", headers={"x-litellm-api-key": key})
            detail: Final = gateway.client.get(f"/v1/mcp/server/{identity}", headers={"x-litellm-api-key": key})
            assert listing.status_code == 200 and detail.status_code == 200, (listing.text, detail.text)
            assert secret not in listing.text + detail.text, key == gateway.key
        viewed: Final = gateway.client.get("/v1/mcp/server", headers={"x-litellm-api-key": viewer})
        assert header_secret not in viewed.text, viewed.text
        peer.drain()
        assert (
            call_tool(gateway, viewer, identity, tool_names(gateway, viewer, identity)["add"], ADD).status_code == 200
        )
        sent: Final = tool_calls(peer.drain())
        assert [call["headers"][b"authorization"] for call in sent] == [f"Bearer {secret}".encode()]
        assert [call["headers"][b"x-integration-secret"] for call in sent] == [header_secret.encode()]


def test_edit_url_moves_calls_to_the_new_peer_without_touching_grants(gateway: Gateway) -> None:
    with mcp_peer() as first, mcp_peer() as second, gateway.scenario() as scenario:
        alias: Final = "mgmt" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, first, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        name: Final = tool_names(gateway, key, identity)["add"]
        assert call_tool(gateway, key, identity, name, ADD).status_code == 200
        assert len(tool_calls(first.drain())) == 1
        moved: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": identity, "url": second.url})
        assert moved.status_code == 202, moved.text
        second.drain()
        response: Final = eventually(
            lambda: call_tool(gateway, key, identity, name, ADD),
            lambda value: value.status_code == 200 and len(tool_calls(second.drain())) == 1,
        )
        assert response.json()["content"][0]["text"] == "9", response.text
        assert tool_calls(first.drain()) == ()


def test_delete_removes_listing_calls_and_database_row(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "mgmt" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        name: Final = tool_names(gateway, key, identity)["add"]
        delete_mcp(gateway, identity)
        assert identity not in _servers(gateway)
        listing: Final = gateway.client.get(
            "/mcp-rest/tools/list", headers={"x-litellm-api-key": key}, params={"server_id": identity}
        )
        assert listing.status_code >= 400 or listing.json() == [], listing.text
        peer.drain()
        response: Final = call_tool(gateway, key, identity, name, ADD)
        assert response.status_code >= 400, response.text
        assert tool_calls(peer.drain()) == ()
        caller: Final = McpCaller(gateway, key, "server_mcp", alias)
        assert caller.list_tools().tools == (), caller.list_tools().raw


def test_duplicate_alias_is_rejected_so_tool_prefixes_cannot_collide(gateway: Gateway) -> None:
    import concurrent.futures

    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "mgmt" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})

        duplicate: Final = gateway.request(
            "POST", "/v1/mcp/server", {"server_name": alias, "alias": alias, **peer.registration()}
        )
        assert duplicate.status_code == 400, duplicate.text
        assert alias in duplicate.json()["detail"]["error"], duplicate.text

        same_alias: Final = gateway.request(
            "POST", "/v1/mcp/server", {"server_name": alias + "other", "alias": alias, **peer.registration()}
        )
        assert same_alias.status_code == 400, same_alias.text
        assert alias in same_alias.json()["detail"]["error"], same_alias.text

        case_variant: Final = gateway.request(
            "POST", "/v1/mcp/server", {"server_name": alias.upper(), "alias": alias.upper(), **peer.registration()}
        )
        assert case_variant.status_code == 400, case_variant.text

        same_name_no_alias: Final = gateway.request(
            "POST", "/v1/mcp/server", {"server_name": alias, **peer.registration()}
        )
        assert same_name_no_alias.status_code == 400, same_name_no_alias.text

        second_alias: Final = alias + "2"
        second_identity: Final = register_mcp(scenario, peer, second_alias)
        colliding_rename: Final = gateway.request(
            "PUT", "/v1/mcp/server", {"server_id": second_identity, "alias": alias}
        )
        assert colliding_rename.status_code == 400, colliding_rename.text

        cleared_alias: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": second_identity, "alias": None})
        assert cleared_alias.status_code == 202, cleared_alias.text

        name: Final = tool_names(gateway, key, identity)["add"]
        response: Final = call_tool(gateway, key, identity, name, ADD)
        assert response.status_code == 200, response.text
        assert response.json()["content"][0]["text"] == "9", response.text

        racing_alias: Final = "race" + uuid.uuid4().hex[:8]

        def try_register() -> int:
            response: Final = gateway.request(
                "POST", "/v1/mcp/server", {"server_name": racing_alias, "alias": racing_alias, **peer.registration()}
            )
            return response.status_code

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            statuses: Final = tuple(pool.map(lambda _i: try_register(), range(8)))

        assert statuses.count(201) == 1, statuses
        assert statuses.count(400) == 7, statuses
        winner: Final = next(
            server["server_id"] for server in _servers(gateway).values() if server["alias"] == racing_alias
        )
        scenario.cleanups.callback(forget_mcp, gateway, winner)


def test_invalid_registrations_are_rejected(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "mgmt" + uuid.uuid4().hex[:8]
        register_mcp(scenario, peer, alias)
        no_url: Final = gateway.request("POST", "/v1/mcp/server", {"server_name": alias + "b", "transport": "http"})
        assert no_url.status_code in (400, 422), no_url.text
        bad_command: Final = gateway.request(
            "POST",
            "/v1/mcp/server",
            {"server_name": alias + "c", "transport": "stdio", "command": "/bin/sh", "args": ["-c", "true"]},
        )
        assert bad_command.status_code in (400, 422), bad_command.text
        hyphenless: Final = gateway.request(
            "POST", "/v1/mcp/server", {"server_name": "bad name!", **peer.registration()}
        )
        assert hyphenless.status_code in (400, 422), hyphenless.text
        assert len([s for s in _servers(gateway).values() if str(s["server_name"]).startswith(alias)]) == 1


def test_access_group_membership_follows_edits(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        group: Final = "grp" + uuid.uuid4().hex[:8]
        alias: Final = "mgmt" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, mcp_access_groups=[group])
        key: Final = scenario.key(object_permission={"mcp_access_groups": [group]})
        groups: Final = gateway.client.get("/v1/mcp/access_groups", headers={"x-litellm-api-key": gateway.key})
        assert groups.status_code == 200 and group in groups.text, groups.text
        assert "add" in tool_names(gateway, key, identity)
        removed: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": identity, "mcp_access_groups": []})
        assert removed.status_code == 202, removed.text
        eventually(
            lambda: gateway.client.get(
                "/mcp-rest/tools/list", headers={"x-litellm-api-key": key}, params={"server_id": identity}
            ),
            lambda value: value.status_code >= 400 or value.json() == [],
        )
        peer.drain()
        denied: Final = call_tool(gateway, key, identity, f"{alias}-add", ADD)
        assert denied.status_code >= 400, denied.text
        assert tool_calls(peer.drain()) == ()


def test_peer_worker_observes_create_edit_and_delete_without_restart(gateway: Gateway, peer: Gateway) -> None:
    with mcp_peer() as first, mcp_peer() as second, gateway.scenario() as scenario:
        alias: Final = "mgmt" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, first, alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        eventually(
            lambda: peer.client.get(
                "/mcp-rest/tools/list", headers={"x-litellm-api-key": key}, params={"server_id": identity}
            ),
            lambda value: value.status_code == 200 and value.json() != [],
            seconds=40,
        )
        names: Final = tool_names(peer, key, identity)
        assert call_tool(peer, key, identity, names["add"], ADD).status_code == 200
        assert len(tool_calls(first.drain())) == 1
        moved: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": identity, "url": second.url})
        assert moved.status_code == 202, moved.text
        eventually(
            lambda: call_tool(peer, key, identity, names["add"], ADD),
            lambda value: value.status_code == 200 and len(tool_calls(second.drain())) == 1,
            seconds=40,
        )
        delete_mcp(gateway, identity)
        eventually(
            lambda: peer.client.get(
                "/mcp-rest/tools/list", headers={"x-litellm-api-key": key}, params={"server_id": identity}
            ),
            lambda value: value.status_code >= 400 or value.json() == [],
            seconds=40,
        )
        second.drain()
        assert call_tool(peer, key, identity, names["add"], ADD).status_code >= 400
        assert tool_calls(second.drain()) == ()


def test_config_declared_server_behaves_like_database_server_but_is_read_only(gateway: Gateway, tmp_path: Path) -> None:
    with mcp_peer() as declared_peer, mcp_peer() as database_peer:
        config: Final = yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
        declared: Final = "declared" + uuid.uuid4().hex[:8]
        config["mcp_servers"] = {declared: {**declared_peer.registration(), "static_headers": {"X-From": "config"}}}
        path: Final = tmp_path / "mcp.yaml"
        path.write_text(yaml.safe_dump(config))
        with owned_proxy(gateway, tmp_path, {}, config=path) as candidate, candidate.scenario() as scenario:
            servers: Final = _servers(candidate)
            declared_id: Final = next(identity for identity, s in servers.items() if s["server_name"] == declared)
            created: Final = register_mcp(scenario, database_peer, "database" + uuid.uuid4().hex[:8])
            key: Final = scenario.key(object_permission={"mcp_servers": [declared_id, created]})
            declared_names: Final = tool_names(candidate, key, declared_id)
            assert set(declared_names) == set(tool_names(candidate, key, created)) == {"add", "multiply", "fail"}
            declared_peer.drain()
            response: Final = call_tool(candidate, key, declared_id, declared_names["add"], ADD)
            assert response.status_code == 200 and response.json()["content"][0]["text"] == "9", response.text
            sent: Final = tool_calls(declared_peer.drain())
            assert [call["headers"][b"x-from"] for call in sent] == [b"config"]
            edited: Final = candidate.request(
                "PUT", "/v1/mcp/server", {"server_id": declared_id, "url": database_peer.url}
            )
            assert edited.status_code >= 400, edited.text
            deleted: Final = candidate.request("DELETE", f"/v1/mcp/server/{declared_id}")
            assert deleted.status_code >= 400, deleted.text
            assert declared_id in _servers(candidate)
            assert call_tool(candidate, key, declared_id, declared_names["add"], ADD).status_code == 200
            assert len(tool_calls(declared_peer.drain())) == 1 and tool_calls(database_peer.drain()) == ()


@pytest.mark.xfail(
    strict=True,
    raises=pytest.RaisesExc(AssertionError, match="LIT-3974 detail access should succeed"),
    reason="LIT-3974 A: team-granted detail access",
)
def test_team_granted_database_server_detail_is_available_to_team_key(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit3974_team_" + uuid.uuid4().hex[:8]
        server_id: Final = register_mcp(scenario, peer, alias)
        team_id: Final = scenario.team(object_permission={"mcp_servers": [server_id]})
        key: Final = scenario.key(team_id=team_id)

        response: Final = gateway.request("GET", f"/v1/mcp/server/{server_id}", key=key)

        assert response.status_code == 200, f"LIT-3974 detail access should succeed: {response.text}"
        assert response.json()["server_id"] == server_id, response.text
        assert response.json()["alias"] == alias, response.text


@pytest.mark.xfail(
    strict=True,
    raises=pytest.RaisesExc(AssertionError, match="LIT-3974 detail access should succeed"),
    reason="LIT-3974 A: team-granted detail access",
)
def test_ui_session_lists_and_fetches_team_granted_config_server(
    gateway: Gateway,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "sk-integration-salt")
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit3974_config_" + uuid.uuid4().hex[:8]
        server_id: Final = "lit3974-" + uuid.uuid4().hex[:12]
        team_id: Final = scenario.team(object_permission={"mcp_servers": [server_id]})
        user_id: Final = scenario.user(user_role="internal_user", teams=[team_id])
        config: Final = yaml.safe_load((Path(__file__).resolve().parents[1] / "proxy_config.yaml").read_text())
        config["mcp_servers"] = {alias: {**peer.registration(), "alias": alias, "server_id": server_id}}
        config_path: Final = tmp_path / "lit3974-mcp.yaml"
        config_path.write_text(yaml.safe_dump(config))

        with owned_proxy(gateway, tmp_path, {}, config=config_path) as candidate:
            token: Final = _dashboard_ui_session_token(user_id)
            headers: Final = {"Authorization": f"Bearer {token}"}
            listed: Final = candidate.client.get("/v1/mcp/server", headers=headers)
            assert listed.status_code == 200, listed.text
            assert [server["server_id"] for server in listed.json()] == [server_id], listed.text

            detail: Final = candidate.client.get(f"/v1/mcp/server/{server_id}", headers=headers)

        assert detail.status_code == 200, f"LIT-3974 detail access should succeed: {detail.text}"
        assert detail.json()["server_id"] == server_id, detail.text
        assert detail.json()["alias"] == alias, detail.text
