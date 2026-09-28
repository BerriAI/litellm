import json
import queue
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal
from urllib.parse import parse_qs, urlsplit

import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.mcp import McpPeer, call_tool, mcp_peer, register_mcp, tool_names
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import TypeAdapter

_Upstream = Callable[[Request], Reply]


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


@dataclass(frozen=True, slots=True)
class _Hold:
    armed: threading.Event = field(default_factory=threading.Event)
    released: threading.Event = field(default_factory=threading.Event)


def _idp_upstream(origin: Callable[[], str], moved: threading.Event, hold: _Hold | None = None) -> _Upstream:
    def issuer() -> str:
        return origin() + ("/idp-after" if moved.is_set() else "/idp-before")

    def respond(request: Request) -> Reply:
        if "oauth-authorization-server" in request.target or "openid-configuration" in request.target:
            current: Final = issuer()
            return Reply(
                body=json.dumps(
                    {
                        "issuer": current,
                        "authorization_endpoint": current + "/authorize",
                        "token_endpoint": current + "/token",
                    }
                ).encode()
            )
        if request.target.startswith("/.well-known/oauth-protected-resource"):
            body: Final = json.dumps({"resource": origin() + "/mcp", "authorization_servers": [issuer()]}).encode()
            if hold is not None and hold.armed.is_set():
                assert hold.released.wait(timeout=15), "the held upstream metadata reply was never released"
            return Reply(body=body)
        return Reply(status=404, body=b'{"error":"unexpected"}')

    return respond


def _register_pass_through(scenario: Scenario, wire: Wire, alias: str) -> str:
    return register_mcp(scenario, McpPeer(wire.url + "/mcp", queue.Queue()), alias, auth_type="true_passthrough")


def _wire_requests(wire: Wire, seen: list[Request]) -> Callable[[], tuple[Request, ...]]:
    def observed() -> tuple[Request, ...]:
        seen.extend(wire.drain())
        return tuple(seen)

    return observed


def _registration_discovery_settled(requests: tuple[Request, ...]) -> bool:
    return any(
        "oauth-authorization-server" in item.target or "openid-configuration" in item.target for item in requests
    )


def _advertised_authorization_servers(gateway: Gateway, alias: str) -> tuple[str, ...]:
    response: Final = gateway.client.get(f"/.well-known/oauth-protected-resource/{alias}/mcp")
    assert response.status_code == 200, response.text
    return tuple(TypeAdapter(list[str]).validate_python(response.json()["authorization_servers"]))


def test_saving_a_pass_through_server_refetches_its_upstream_oauth_metadata(gateway: Gateway) -> None:
    moved: Final = threading.Event()
    with wire_server(_idp_upstream(lambda: wire.url, moved)) as wire, gateway.scenario() as scenario:
        alias: Final = "pt" + uuid.uuid4().hex[:8]
        identity: Final = _register_pass_through(scenario, wire, alias)
        assert _advertised_authorization_servers(gateway, alias) == (wire.url + "/idp-before",)
        assert _advertised_authorization_servers(gateway, alias) == (wire.url + "/idp-before",)
        moved.set()
        wire.drain()
        saved: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": identity, "description": "IdP moved"})
        assert saved.status_code == 202, saved.text
        assert _advertised_authorization_servers(gateway, alias) == (wire.url + "/idp-after",)
        assert any(request.target.startswith("/.well-known/oauth-protected-resource") for request in wire.drain()), (
            "the save must send protected-resource discovery back to the upstream"
        )


def test_metadata_fetched_before_a_save_cannot_repopulate_the_cache_after_it(gateway: Gateway) -> None:
    moved: Final = threading.Event()
    hold: Final = _Hold()
    with (
        wire_server(_idp_upstream(lambda: wire.url, moved, hold)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        alias: Final = "pt" + uuid.uuid4().hex[:8]
        identity: Final = _register_pass_through(scenario, wire, alias)
        seen: Final[list[Request]] = []
        observed: Final = _wire_requests(wire, seen)
        eventually(observed, _registration_discovery_settled, seconds=10)
        settled: Final = len(seen)
        hold.armed.set()
        stale: Final = pool.submit(_advertised_authorization_servers, gateway, alias)
        eventually(observed, lambda requests: len(requests) > settled, seconds=10)
        assert seen[settled].target.startswith("/.well-known/oauth-protected-resource"), seen[settled:]
        moved.set()
        saved: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": identity, "description": "IdP moved"})
        assert saved.status_code == 202, saved.text
        hold.released.set()
        assert stale.result(timeout=30) == (wire.url + "/idp-before",)
        assert _advertised_authorization_servers(gateway, alias) == (wire.url + "/idp-after",)


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
