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

import httpx
import pytest
from integration._support.client import Gateway, Scenario, eventually
from integration._support.database import read_rows
from integration._support.mcp import McpPeer, call_tool, mcp_peer, register_mcp, tool_names
from integration._support.oauth_server import oauth_server
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import TypeAdapter

_Upstream = Callable[[Request], Reply]
CLIENT_REDIRECT: Final = "http://127.0.0.1:9/cb"


def _response_object(response: httpx.Response) -> dict[str, object]:
    return TypeAdapter(dict[str, object]).validate_python(response.json())


def _start_callback_flow(
    gateway: Gateway,
    alias: str,
    key: str,
    client_id: str,
    expected_authorization_url: str | None = None,
) -> tuple[str, str, str]:
    started: Final = gateway.client.get(
        f"/{alias}/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": CLIENT_REDIRECT,
            "response_type": "code",
            "state": "client-state",
            "code_challenge": "A" * 43,
            "code_challenge_method": "S256",
        },
        headers={"x-litellm-api-key": key},
    )
    assert started.status_code in (302, 307), started.text
    location: Final = urlsplit(started.headers["location"])
    if expected_authorization_url is not None:
        assert location.scheme + "://" + location.netloc + location.path == expected_authorization_url, started.text
    relay_state: Final = parse_qs(location.query)["state"][0]
    cookie_name: Final = f"mcp_oauth_state_{relay_state}"
    assert cookie_name in started.cookies, started.headers
    return relay_state, cookie_name, started.cookies[cookie_name]


def _assert_cleared_oauth_state_cookie(response: httpx.Response, cookie_name: str) -> None:
    cleared: Final = response.headers["set-cookie"]
    assert cleared.startswith(f'{cookie_name}="";'), cleared
    assert "Max-Age=0" in cleared, cleared
    assert "Path=/" in cleared, cleared
    assert "HttpOnly" in cleared, cleared
    assert "SameSite=lax" in cleared, cleared


def test_callback_forwards_the_code_only_for_the_sealed_issuer_and_relays_idp_errors(
    gateway: Gateway,
) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "rt4" + uuid.uuid4().hex[:10]
        identity: Final = register_mcp(
            scenario,
            peer,
            alias,
            issuer=auth.issuer,
            authorization_url=auth.issuer + "/authorize",
            token_url=auth.issuer + "/token",
            registration_url=auth.issuer + "/register",
            auth_type="oauth2",
            oauth2_flow="authorization_code",
            credentials={"client_id": "ac-client", "client_secret": "ac-secret", "scopes": ["tools.call"]},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        registered: Final = gateway.client.post(
            f"/{alias}/register", json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
        )
        assert registered.status_code in (200, 201), registered.text
        client_id: Final = str(_response_object(registered)["client_id"])

        wrong_relay, wrong_cookie, wrong_cookie_value = _start_callback_flow(gateway, alias, key, client_id)
        wrong_code: Final = "callback-code-" + uuid.uuid4().hex
        wrong: Final = gateway.client.get(
            "/callback",
            params={"code": wrong_code, "state": wrong_relay, "iss": "http://127.0.0.1:1/other"},
            cookies={wrong_cookie: wrong_cookie_value},
        )
        assert wrong.status_code == 400, wrong.text
        assert wrong.headers["content-type"] == "text/html; charset=utf-8", wrong.headers
        assert wrong.text == (
            "<html><body><h2>Authentication failed</h2><p><strong>Error:</strong> invalid_issuer</p>"
            "<p>This authorization response came from a different identity provider than the one this "
            "MCP server is configured to use.</p>"
            "<p>You can close this window and try again.</p></body></html>"
        ), wrong.text
        assert "location" not in wrong.headers, wrong.headers
        assert wrong_code not in wrong.text, wrong.text
        _assert_cleared_oauth_state_cookie(wrong, wrong_cookie)

        correct_relay, correct_cookie, correct_cookie_value = _start_callback_flow(gateway, alias, key, client_id)
        correct: Final = gateway.client.get(
            "/callback",
            params={"code": "c", "state": correct_relay, "iss": auth.issuer},
            cookies={correct_cookie: correct_cookie_value},
        )
        assert correct.status_code == 302, correct.text
        correct_location: Final = urlsplit(correct.headers["location"])
        assert correct_location.scheme + "://" + correct_location.netloc + correct_location.path == CLIENT_REDIRECT, (
            correct.headers["location"]
        )
        assert parse_qs(correct_location.query) == {"code": ["c"], "state": ["client-state"]}, correct.headers[
            "location"
        ]
        _assert_cleared_oauth_state_cookie(correct, correct_cookie)

        error_relay, error_cookie, error_cookie_value = _start_callback_flow(gateway, alias, key, client_id)
        error: Final = gateway.client.get(
            "/callback",
            params={"error": "access_denied", "error_description": "x", "state": error_relay},
            cookies={error_cookie: error_cookie_value},
        )
        assert error.status_code == 302, error.text
        error_location: Final = urlsplit(error.headers["location"])
        assert parse_qs(error_location.query) == {
            "error": ["access_denied"],
            "error_description": ["x"],
            "state": ["client-state"],
        }, error.headers["location"]
        _assert_cleared_oauth_state_cookie(error, error_cookie)

        error_issuer_relay, error_issuer_cookie, error_issuer_cookie_value = _start_callback_flow(
            gateway, alias, key, client_id
        )
        error_issuer: Final = gateway.client.get(
            "/callback",
            params={
                "error": "access_denied",
                "error_description": "x",
                "state": error_issuer_relay,
                "iss": "http://127.0.0.1:1/other",
            },
            cookies={error_issuer_cookie: error_issuer_cookie_value},
        )
        assert error_issuer.status_code == 400, error_issuer.text
        assert error_issuer.headers["content-type"] == "text/html; charset=utf-8", error_issuer.headers
        assert error_issuer.text == (
            "<html><body><h2>Authentication failed</h2><p><strong>Error:</strong> invalid_issuer</p>"
            "<p>Unexpected authorization issuer</p>"
            "<p>You can close this window and try again.</p></body></html>"
        ), error_issuer.text
        assert "location" not in error_issuer.headers, error_issuer.headers
        _assert_cleared_oauth_state_cookie(error_issuer, error_issuer_cookie)

        missing: Final = gateway.client.get("/callback")
        assert missing.status_code == 400, missing.text
        assert missing.headers["content-type"] == "text/html; charset=utf-8", missing.headers
        assert missing.text == (
            "<html><body><h2>Authentication failed</h2><p><strong>Error:</strong> invalid_request</p>"
            "<p>Missing authorization &#x27;code&#x27; and &#x27;state&#x27; parameter(s).</p>"
            "<p>You can close this window and try again.</p></body></html>"
        ), missing.text


def test_callback_refuses_a_response_without_iss_when_the_authorization_server_advertises_iss_support(
    gateway: Gateway,
) -> None:
    wire_holder: Final[list[Wire]] = []

    def respond(request: Request) -> Reply:
        path: Final = urlsplit(request.target).path
        if request.method == "GET" and path in (
            "/.well-known/oauth-authorization-server",
            "/.well-known/openid-configuration",
        ):
            issuer: Final = wire_holder[0].url
            return Reply(
                body=json.dumps(
                    {
                        "issuer": issuer,
                        "authorization_endpoint": issuer + "/authorize",
                        "token_endpoint": issuer + "/token",
                        "registration_endpoint": issuer + "/register",
                        "response_types_supported": ["code"],
                        "grant_types_supported": ["authorization_code", "refresh_token"],
                        "code_challenge_methods_supported": ["S256"],
                        "token_endpoint_auth_methods_supported": ["client_secret_post"],
                        "authorization_response_iss_parameter_supported": True,
                    }
                ).encode()
            )
        return Reply(status=404, body=json.dumps({"error": "not_found"}).encode())

    with wire_server(respond) as wire, mcp_peer() as peer, gateway.scenario() as scenario:
        wire_holder.append(wire)
        alias: Final = "rt5" + uuid.uuid4().hex[:10]
        identity: Final = register_mcp(
            scenario,
            peer,
            alias,
            issuer=wire.url,
            auth_type="oauth2",
            oauth2_flow="authorization_code",
            credentials={"client_id": "ac-client", "client_secret": "ac-secret", "scopes": ["tools.call"]},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})

        relay, cookie, cookie_value = _start_callback_flow(
            gateway, alias, key, "ac-client", expected_authorization_url=wire.url + "/authorize"
        )
        code: Final = "callback-code-" + uuid.uuid4().hex
        missing_issuer: Final = gateway.client.get(
            "/callback",
            params={"code": code, "state": relay},
            cookies={cookie: cookie_value},
        )
        assert missing_issuer.status_code == 400, missing_issuer.text
        assert missing_issuer.headers["content-type"] == "text/html; charset=utf-8", missing_issuer.headers
        assert missing_issuer.text == (
            "<html><body><h2>Authentication failed</h2><p><strong>Error:</strong> invalid_issuer</p>"
            "<p>This authorization response came from a different identity provider than the one this "
            "MCP server is configured to use.</p>"
            "<p>You can close this window and try again.</p></body></html>"
        ), missing_issuer.text
        assert "location" not in missing_issuer.headers, missing_issuer.headers
        assert code not in missing_issuer.text, missing_issuer.text
        _assert_cleared_oauth_state_cookie(missing_issuer, cookie)

        trusted_relay, trusted_cookie, trusted_cookie_value = _start_callback_flow(
            gateway, alias, key, "ac-client", expected_authorization_url=wire.url + "/authorize"
        )
        trusted_code: Final = "callback-code-" + uuid.uuid4().hex
        trusted: Final = gateway.client.get(
            "/callback",
            params={"code": trusted_code, "state": trusted_relay, "iss": wire.url},
            cookies={trusted_cookie: trusted_cookie_value},
        )
        assert trusted.status_code == 302, trusted.text
        trusted_location: Final = urlsplit(trusted.headers["location"])
        assert trusted_location.scheme + "://" + trusted_location.netloc + trusted_location.path == CLIENT_REDIRECT, (
            trusted.headers["location"]
        )
        assert parse_qs(trusted_location.query) == {
            "code": [trusted_code],
            "state": ["client-state"],
        }, trusted.headers["location"]
        _assert_cleared_oauth_state_cookie(trusted, trusted_cookie)

        wire_requests: Final = wire.drain()
        wire_method_paths: Final = tuple((request.method, urlsplit(request.target).path) for request in wire_requests)
        assert wire_method_paths == (("GET", "/.well-known/oauth-authorization-server"),), wire_method_paths
        assert not any(method == "POST" and path == "/token" for method, path in wire_method_paths), wire_method_paths


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


def _eventually_advertises(gateway: Gateway, alias: str, issuer: str) -> None:
    eventually(
        lambda: gateway.client.get(f"/.well-known/oauth-protected-resource/{alias}/mcp"),
        lambda response: response.status_code == 200 and response.json()["authorization_servers"] == [issuer],
        seconds=40,
    )


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


def test_peer_worker_stops_advertising_the_old_idp_after_a_save_on_another_worker(
    gateway: Gateway, peer: Gateway
) -> None:
    moved: Final = threading.Event()
    with wire_server(_idp_upstream(lambda: wire.url, moved)) as wire, gateway.scenario() as scenario:
        alias: Final = "pt" + uuid.uuid4().hex[:8]
        identity: Final = _register_pass_through(scenario, wire, alias)
        assert _advertised_authorization_servers(gateway, alias) == (wire.url + "/idp-before",)
        _eventually_advertises(peer, alias, wire.url + "/idp-before")
        moved.set()
        saved: Final = gateway.request("PUT", "/v1/mcp/server", {"server_id": identity, "description": "IdP moved"})
        assert saved.status_code == 202, saved.text
        assert _advertised_authorization_servers(gateway, alias) == (wire.url + "/idp-after",)
        _eventually_advertises(peer, alias, wire.url + "/idp-after")


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
