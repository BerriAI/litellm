import json
import queue
import uuid
from urllib.parse import parse_qs, urlsplit
from typing import Final, Literal
from pathlib import Path

import pytest

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import McpPeer, call_tool, mcp_peer, register_mcp, tool_names
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.mcp.oauth.discovery_cannot_erase_configured_authorization_endpoint")
def test_partial_discovery_and_unrelated_edit_keep_actual_authorization_destination(
    gateway: Gateway, tmp_path: Path
) -> None:
    def discovery(request: Request) -> Reply:
        if request.target.startswith("/configured-authorize"):
            return Reply(body=b'{"synthetic_authorization_endpoint":true}')
        if request.target == "/mcp":
            return Reply(body=b'{"synthetic_resource":true}')
        if request.target.startswith("/.well-known/oauth-protected-resource"):
            return Reply(
                body=json.dumps(
                    {
                        "resource": wire.url + "/mcp",
                        "authorization_servers": [wire.url],
                        "scopes_supported": ["tools.read"],
                    }
                ).encode()
            )
        if request.method == "GET":
            return Reply(
                body=json.dumps(
                    {
                        "issuer": wire.url,
                        "token_endpoint": wire.url + "/discovered-token",
                        "scopes_supported": ["tools.read"],
                    }
                ).encode()
            )
        return Reply(status=401, body=b'{"error":"synthetic OAuth requirement"}')

    with (
        wire_server(discovery) as wire,
        owned_proxy(gateway, tmp_path, {"LITELLM_MCP_OAUTH_DISCOVERY_ON_STARTUP": "true"}) as candidate,
        candidate.scenario() as scenario,
    ):
        gateway = candidate
        alias: Final = "integration" + uuid.uuid4().hex
        endpoint: Final = wire.url + "/configured-authorize"
        identity: Final = register_mcp(
            scenario,
            McpPeer(wire.url + "/mcp", queue.Queue()),
            alias,
            auth_type="oauth2",
            authorization_url=endpoint,
            token_url=wire.url + "/configured-token",
            oauth2_flow="authorization_code",
            credentials={"client_id": "synthetic-oauth-client"},
        )
        discovered = []

        def observed() -> tuple[Request, ...]:
            discovered.extend(wire.drain())
            return tuple(item for item in discovered if item.method == "GET" and ".well-known/" in item.target)

        assert eventually(observed, bool, seconds=10)
        for generation in range(2):
            rows: Final = read_rows(
                'SELECT authorization_url FROM "LiteLLM_MCPServerTable" WHERE server_id=%s', (identity,)
            )
            assert rows == [{"authorization_url": endpoint}]
            response: Final = gateway.request(
                "GET",
                f"/v1/mcp/server/oauth/{identity}/authorize",
                params={
                    "redirect_uri": "http://127.0.0.1:8765/callback",
                    "state": "synthetic-state",
                    "code_challenge": "A" * 43,
                    "code_challenge_method": "S256",
                    "response_type": "code",
                },
            )
            assert response.status_code in (302, 307), response.text
            location: Final = urlsplit(response.headers["location"])
            assert location.scheme + "://" + location.netloc + location.path == endpoint
            query: Final = parse_qs(location.query)
            assert query["client_id"] == ["synthetic-oauth-client"]
            assert query["scope"] == ["tools.read"], (
                "Discovery metadata must be applied before checking endpoint preservation"
            )
            assert query["code_challenge"] == ["A" * 43] and query["code_challenge_method"] == ["S256"]
            selected: Final = gateway.client.get(response.headers["location"])
            assert selected.status_code == 200 and selected.json() == {"synthetic_authorization_endpoint": True}
            if generation == 0:
                updated: Final = gateway.request(
                    "PUT", "/v1/mcp/server", {"server_id": identity, "server_name": alias + "renamed"}
                )
                assert updated.status_code == 202, updated.text


@pytest.mark.covers("other.mcp.oauth.same_url_credentials_are_isolated_by_user_and_server")
@pytest.mark.parametrize("transition", ("revoke", "expire"))
def test_same_url_oauth_credentials_and_revocation_are_isolated_by_user_and_server(
    gateway: Gateway,
    transition: Literal["revoke", "expire"],
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        servers: Final = tuple(
            register_mcp(
                scenario,
                peer,
                "oauth" + uuid.uuid4().hex,
                auth_type="oauth2",
                oauth2_flow="authorization_code",
                authorization_url=peer.url + "/authorize",
                token_url=peer.url + "/token",
                credentials={"client_id": "synthetic-oauth-client"},
            )
            for _ in range(2)
        )
        users: Final = tuple(scenario.user(user_role="internal_user") for _ in range(2))
        keys: Final = tuple(
            scenario.key(user_id=user, object_permission={"mcp_servers": list(servers)}) for user in users
        )
        for user_index, key in enumerate(keys):
            for server_index, server_id in enumerate(servers):
                stored: Final = gateway.request(
                    "POST",
                    f"/v1/mcp/server/{server_id}/oauth-user-credential",
                    {"access_token": f"synthetic-user-{user_index}-server-{server_index}", "expires_in": 3600},
                    key=key,
                )
                assert stored.status_code == 200 and stored.json()["has_credential"] is True, stored.text
                scenario.cleanups.callback(
                    gateway.request,
                    "DELETE",
                    f"/v1/mcp/server/{server_id}/oauth-user-credential",
                    key=key,
                )
        names: Final = tuple(tool_names(gateway, keys[0], server) for server in servers)
        for generation in range(2):
            for user_index, key in enumerate(keys):
                for server_index, server_id in enumerate(servers):
                    peer.drain()
                    discovery: Final = gateway.request(
                        "GET",
                        "/mcp-rest/tools/list",
                        key=key,
                        params={"server_id": server_id},
                    )
                    call: Final = call_tool(gateway, key, server_id, names[server_index]["add"], {"a": 3, "b": 5})
                    observed: Final = peer.drain()
                    if generation == 1 and user_index == 0 and server_index == 0:
                        for rejected in (discovery, call):
                            assert rejected.status_code == 401, rejected.text
                            assert rejected.json() == {"detail": "Unauthorized"}, rejected.text
                            assert "resource_metadata=" in rejected.headers["www-authenticate"]
                        assert observed == (), "unusable credentials must not fall back to another user or server"
                    else:
                        assert discovery.status_code == 200, discovery.text
                        assert {tool["name"] for tool in discovery.json()["tools"]} == set(names[server_index].values())
                        assert call.status_code == 200 and call.json()["isError"] is False, call.text
                        assert call.json()["content"][0]["text"] == "8", call.text
                        calls: Final = tuple(item for item in observed if item["body"].get("method") == "tools/call")
                        assert len(calls) == 1
                        expected: Final = f"Bearer synthetic-user-{user_index}-server-{server_index}".encode()
                        assert calls[0]["headers"][b"authorization"] == expected
                        assert all(item["headers"].get(b"authorization") == expected for item in observed)
            if generation == 0:
                changed: Final = gateway.request(
                    "DELETE" if transition == "revoke" else "POST",
                    f"/v1/mcp/server/{servers[0]}/oauth-user-credential",
                    None
                    if transition == "revoke"
                    else {
                        "access_token": "synthetic-expired-user-0-server-0",
                        "expires_in": -60,
                    },
                    key=keys[0],
                )
                assert changed.status_code == 200, changed.text
                assert changed.json()["has_credential"] is (transition == "expire"), changed.text
