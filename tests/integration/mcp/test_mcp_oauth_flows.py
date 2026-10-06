import base64
import hashlib
import json
import os
import re
import secrets
import textwrap
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urljoin, urlsplit

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, generate_private_key
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    INITIALIZE,
    EntryPoint,
    McpCaller,
    McpPeer,
    Outcome,
    _outcome_from_rpc,
    call_tool,
    mcp_peer,
    register_mcp,
    tool_calls,
)
from integration._support.mcp_grants import create_toolset
from integration._support.oauth_server import AuthorizationServer, oauth_server
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import TypeAdapter

ADD: Final = {"a": 2, "b": 3}
CLIENT_REDIRECT: Final = "http://127.0.0.1:9/cb"
ACCEPT: Final = {"Accept": "application/json, text/event-stream"}


def _base(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _response_object(response: httpx.Response) -> dict[str, object]:
    return TypeAdapter(dict[str, object]).validate_python(response.json())


def _open_aliases() -> frozenset[str]:
    rows: Final = read_rows('SELECT alias FROM "LiteLLM_MCPServerTable" WHERE allow_all_keys', ())
    return frozenset(str(row["alias"]) for row in rows)


def _without_foreign_open_servers(tools: tuple[str, ...], open_aliases: frozenset[str]) -> set[str]:
    prefixes: Final = tuple(f"{alias}-" for alias in open_aliases)
    return {tool for tool in tools if not tool.startswith(prefixes)}


def _assert_upstream_call(call: Mapping[str, object], name: str, arguments: Mapping[str, object]) -> None:
    body: Final = call["body"]
    assert isinstance(body, Mapping), call
    params: Final = body["params"]
    assert isinstance(params, Mapping), call
    assert set(params) == {"name", "arguments", "_meta"}, call
    assert params["name"] == name, call
    assert params["arguments"] == arguments, call
    metadata: Final = params["_meta"]
    assert isinstance(metadata, Mapping), call
    assert set(metadata) == {"progressToken"}, call


def _authorizations(peer: McpPeer) -> tuple[bytes | None, ...]:
    return tuple(
        value if isinstance(value := call["headers"].get(b"authorization"), bytes) else None
        for call in tool_calls(peer.drain())
        if isinstance(call["headers"], dict)
    )


def _issued_token(issued: dict[str, object]) -> str:
    token: Final = issued["access_token"]
    assert isinstance(token, str)
    return token


def _register_oauth(scenario, peer: McpPeer, auth: AuthorizationServer, alias: str, **fields: object) -> str:
    return register_mcp(
        scenario,
        peer,
        alias,
        issuer=auth.issuer,
        authorization_url=auth.issuer + "/authorize",
        token_url=auth.issuer + "/token",
        registration_url=auth.issuer + "/register",
        **fields,
    )


def _plaintext_credential_rows(identity: str, secret: str) -> list[dict[str, object]]:
    return read_rows(
        'SELECT server_id FROM "LiteLLM_MCPServerTable" WHERE server_id = %s AND credentials::text LIKE %s',
        (identity, f"%{secret}%"),
    )


def test_client_credentials_token_is_minted_once_and_sent_as_bearer(gateway: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "cc" + uuid.uuid4().hex[:8]
        secret: Final = "cc-secret-" + uuid.uuid4().hex
        identity: Final = _register_oauth(
            scenario,
            peer,
            auth,
            alias,
            auth_type="oauth2",
            oauth2_flow="client_credentials",
            credentials={"client_id": "cc-client", "client_secret": secret, "scopes": ["tools.call"]},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        peer.drain()
        for _ in range(2):
            response: Final = call_tool(gateway, key, identity, f"{alias}-add", ADD)
            assert response.status_code == 200, response.text
        minted: Final = auth.token_requests()
        assert [request["grant_type"] for request in minted] == ["client_credentials"], minted
        assert minted[0]["client_id"] == "cc-client" and minted[0]["client_secret"] == secret
        assert minted[0]["scope"] == "tools.call"
        seen: Final = _authorizations(peer)
        assert len(seen) == 2 and len(set(seen)) == 1, seen
        assert seen[0] is not None and auth.is_live(seen[0].decode().removeprefix("Bearer ")), seen
        assert secret.encode() not in (seen[0] or b""), "client secret forwarded to the peer"
        assert _plaintext_credential_rows(identity, secret) == []


def test_rotating_the_client_secret_forces_a_fresh_token(gateway: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "cc" + uuid.uuid4().hex[:8]
        identity: Final = _register_oauth(
            scenario,
            peer,
            auth,
            alias,
            auth_type="oauth2",
            oauth2_flow="client_credentials",
            credentials={"client_id": "cc-client", "client_secret": "first-" + uuid.uuid4().hex},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        assert call_tool(gateway, key, identity, f"{alias}-add", ADD).status_code == 200
        before: Final = _authorizations(peer)
        auth.drain()
        rotated: Final = "second-" + uuid.uuid4().hex
        edited: Final = gateway.request(
            "PUT",
            "/v1/mcp/server",
            {"server_id": identity, "credentials": {"client_id": "cc-client", "client_secret": rotated}},
        )
        assert edited.status_code == 202, edited.text
        after: Final = eventually(
            lambda: (call_tool(gateway, key, identity, f"{alias}-add", ADD).status_code, _authorizations(peer)),
            lambda value: value[0] == 200 and value[1] != () and value[1][-1] not in before,
        )
        assert [request["client_secret"] for request in auth.token_requests()][-1] == rotated
        assert after[1][-1] not in before


def test_token_exchange_swaps_the_callers_subject_token_and_never_forwards_it(gateway: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "te" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario,
            peer,
            alias,
            auth_type="oauth2_token_exchange",
            token_exchange_endpoint=auth.issuer + "/token",
            audience="urn:integration:peer",
            credentials={"client_id": "te-client", "client_secret": "te-secret"},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        subject: Final = "subject-" + uuid.uuid4().hex
        peer.drain()
        auth.drain()
        response: Final = gateway.client.post(
            "/mcp-rest/tools/call",
            headers={"x-litellm-api-key": key, "Authorization": f"Bearer {subject}"},
            json={"name": f"{alias}-add", "arguments": ADD, "server_id": identity},
        )
        assert response.status_code == 200, response.text
        exchanged: Final = auth.token_requests()
        assert len(exchanged) == 1, exchanged
        assert exchanged[0]["grant_type"] == "urn:ietf:params:oauth:grant-type:token-exchange"
        assert exchanged[0]["subject_token"] == subject
        assert exchanged[0]["audience"] == "urn:integration:peer"
        seen: Final = _authorizations(peer)
        assert len(seen) == 1 and seen[0] is not None and subject.encode() not in seen[0], seen
        assert seen[0].startswith(b"Bearer ") and auth.is_live(seen[0].decode().removeprefix("Bearer "))


def test_token_exchange_without_a_subject_token_is_rejected_before_any_upstream_request(gateway: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "te" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario,
            peer,
            alias,
            auth_type="oauth2_token_exchange",
            token_exchange_endpoint=auth.issuer + "/token",
            credentials={"client_id": "te-client", "client_secret": "te-secret"},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        peer.drain()
        auth.drain()
        cold: Final = call_tool(gateway, key, identity, f"{alias}-add", ADD)
        assert tool_calls(peer.drain()) == ()
        assert auth.token_requests() == ()
        _assert_subject_token_challenge(cold, alias)
        warmed: Final = gateway.client.post(
            "/mcp-rest/tools/call",
            headers={"x-litellm-api-key": key, "Authorization": "Bearer subject-" + uuid.uuid4().hex},
            json={"name": f"{alias}-add", "arguments": ADD, "server_id": identity},
        )
        assert warmed.status_code == 200, warmed.text
        assert len(tool_calls(peer.drain())) == 1 and len(auth.token_requests()) == 1
        auth.drain()
        warm: Final = call_tool(gateway, key, identity, f"{alias}-add", ADD)
        assert tool_calls(peer.drain()) == ()
        assert auth.token_requests() == ()
        _assert_subject_token_challenge(warm, alias)
        as_subject: Final = gateway.client.post(
            "/mcp-rest/tools/call",
            headers={"x-litellm-api-key": key, "Authorization": f"Bearer {key}"},
            json={"name": f"{alias}-add", "arguments": ADD, "server_id": identity},
        )
        assert tool_calls(peer.drain()) == ()
        assert auth.token_requests() == ()
        _assert_subject_token_challenge(as_subject, alias)


def _assert_subject_token_challenge(response: httpx.Response, alias: str) -> None:
    assert response.status_code == 401, response.text
    challenge: Final = response.headers["www-authenticate"]
    assert challenge.startswith("Bearer ") and 'error="invalid_token"' in challenge, challenge
    assert f'resource_metadata="/.well-known/oauth-protected-resource/mcp/{alias}"' in challenge, challenge


@pytest.mark.parametrize("entry", ENTRY_POINTS)
def test_delegated_auth_forwards_the_callers_bearer_untouched(gateway: Gateway, entry: EntryPoint) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "dl" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias, auth_type="oauth_delegate")
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        token: Final = "user-" + uuid.uuid4().hex
        caller: Final = McpCaller(gateway, key, entry, alias, headers={"Authorization": f"Bearer {token}"})
        peer.drain()
        outcome: Final = caller.call(f"{alias}-add", ADD, identity if entry in ("mcp", "root", "sse", "rest") else None)
        assert outcome.ok, outcome.raw
        assert _authorizations(peer) == (f"Bearer {token}".encode(),)


@dataclass(frozen=True, slots=True)
class _Pkce:
    verifier: str

    @property
    def challenge(self) -> str:
        digest: Final = hashlib.sha256(self.verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def _authorize_through_gateway(
    gateway: Gateway, auth: AuthorizationServer, alias: str, key: str, client_id: str, pkce: _Pkce
) -> str:
    started: Final = gateway.client.get(
        f"/{alias}/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": CLIENT_REDIRECT,
            "response_type": "code",
            "state": "client-state",
            "code_challenge": pkce.challenge,
            "code_challenge_method": "S256",
            "scope": "tools.call",
        },
        headers={"x-litellm-api-key": key},
    )
    assert started.status_code in (302, 307), started.text
    upstream: Final = started.headers["location"]
    assert upstream.startswith(auth.issuer + "/authorize"), upstream
    upstream_query: Final = parse_qs(urlsplit(upstream).query)
    assert upstream_query["code_challenge_method"] == ["S256"]
    assert upstream_query["redirect_uri"] != [CLIENT_REDIRECT], "client redirect relayed upstream"
    consent: Final = httpx.get(upstream, follow_redirects=False)
    assert consent.status_code == 302, consent.text
    callback: Final = consent.headers["location"]
    assert callback.startswith(_base(gateway)), callback
    returned: Final = gateway.client.get(
        callback.removeprefix(_base(gateway)), headers={"x-litellm-api-key": key}, cookies=started.cookies
    )
    assert returned.status_code == 302, returned.text
    final: Final = parse_qs(urlsplit(returned.headers["location"]).query)
    assert returned.headers["location"].startswith(CLIENT_REDIRECT)
    assert final["state"] == ["client-state"], final
    return final["code"][0]


def _redeem(gateway: Gateway, alias: str, key: str, client_id: str, code: str, pkce: _Pkce) -> httpx.Response:
    return gateway.client.post(
        f"/{alias}/token",
        headers={"x-litellm-api-key": key},
        data={
            "grant_type": "authorization_code",
            "code": code,
            "code_verifier": pkce.verifier,
            "client_id": client_id,
            "redirect_uri": CLIENT_REDIRECT,
        },
    )


@pytest.mark.parametrize("auth_method", ("client_secret_post", "client_secret_basic"))
def test_upstream_token_requests_carry_the_gateway_callback_and_server_credentials(
    gateway: Gateway, auth_method: str
) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "rt3" + uuid.uuid4().hex[:10]
        secret: Final = "rt3-secret-" + uuid.uuid4().hex
        identity: Final = _register_oauth(
            scenario,
            peer,
            auth,
            alias,
            auth_type="oauth2",
            oauth2_flow="authorization_code",
            credentials={
                "client_id": "ac-client",
                "client_secret": secret,
                "scopes": ["tools.call"],
                "token_endpoint_auth_method": auth_method,
            },
        )
        persisted: Final = gateway.request("GET", f"/v1/mcp/server/{identity}")
        assert persisted.status_code == 200, persisted.text
        persisted_body: Final = _response_object(persisted)
        assert persisted_body["credentials"] == {"scopes": ["tools.call"]}, persisted.text
        owner: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        registered: Final = gateway.client.post(
            f"/{alias}/register", json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
        )
        assert registered.status_code in (200, 201), registered.text
        client_id: Final = str(_response_object(registered)["client_id"])
        pkce: Final = _Pkce(secrets.token_urlsafe(32))
        code: Final = _authorize_through_gateway(gateway, auth, alias, owner, client_id, pkce)
        authorize_requests: Final = tuple(item for item in auth.drain() if item.target.startswith("/authorize"))
        assert len(authorize_requests) == 1, authorize_requests
        authorize_query: Final = parse_qs(urlsplit(authorize_requests[0].target).query)
        assert authorize_query["redirect_uri"] == [f"{_base(gateway)}/callback"], authorize_query

        redeemed: Final = gateway.client.post(
            f"/{alias}/token",
            headers={"x-litellm-api-key": owner},
            data={
                "grant_type": "authorization_code",
                "code": code,
                "code_verifier": pkce.verifier,
                "client_id": client_id,
                "client_secret": "dummy",
                "redirect_uri": CLIENT_REDIRECT,
            },
        )
        assert redeemed.status_code == 200, redeemed.text
        issued: Final = _response_object(redeemed)
        initial_token_requests: Final = tuple(item for item in auth.drain() if item.target.startswith("/token"))
        assert len(initial_token_requests) == 1, initial_token_requests
        initial_request: Final = initial_token_requests[0]
        initial_form: Final = {name: values[0] for name, values in parse_qs(initial_request.body.decode()).items()}
        expected_authorization: Final = "Basic " + base64.b64encode(f"ac-client:{secret}".encode()).decode()
        if auth_method == "client_secret_post":
            assert initial_form == {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": f"{_base(gateway)}/callback",
                "code_verifier": pkce.verifier,
                "client_id": "ac-client",
                "client_secret": secret,
            }, initial_form
            assert initial_request.headers.get("authorization") is None, initial_request.headers
        else:
            assert initial_form == {
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": f"{_base(gateway)}/callback",
                "code_verifier": pkce.verifier,
            }, initial_form
            assert initial_request.headers.get("authorization") == expected_authorization, initial_request.headers

        upstream_refresh_token: Final = str(issued["refresh_token"])
        refreshed: Final = gateway.client.post(
            f"/{alias}/token",
            headers={"x-litellm-api-key": owner},
            data={
                "grant_type": "refresh_token",
                "refresh_token": upstream_refresh_token,
                "client_id": client_id,
                "client_secret": "dummy",
                "scope": "tools.call",
            },
        )
        assert refreshed.status_code == 200, refreshed.text
        assert _response_object(refreshed)["access_token"] != issued["access_token"], refreshed.text
        refresh_requests: Final = tuple(item for item in auth.drain() if item.target.startswith("/token"))
        assert len(refresh_requests) == 1, refresh_requests
        refresh_request: Final = refresh_requests[0]
        refresh_form: Final = {name: values[0] for name, values in parse_qs(refresh_request.body.decode()).items()}
        expected_refresh_form: Final = {
            "grant_type": "refresh_token",
            "refresh_token": upstream_refresh_token,
            "scope": "tools.call",
            **({"client_id": "ac-client", "client_secret": secret} if auth_method == "client_secret_post" else {}),
        }
        assert refresh_form == expected_refresh_form, refresh_form
        if auth_method == "client_secret_post":
            assert refresh_request.headers.get("authorization") is None, refresh_request.headers
        else:
            assert refresh_request.headers.get("authorization") == expected_authorization, refresh_request.headers
        assert "dummy" not in refresh_request.body.decode() and "dummy" not in repr(refresh_request.headers), (
            refresh_request
        )


def test_per_user_authorization_code_with_pkce_binds_the_token_to_the_authorizing_user(gateway: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "ac" + uuid.uuid4().hex[:8]
        identity: Final = _register_oauth(
            scenario,
            peer,
            auth,
            alias,
            auth_type="oauth2",
            oauth2_flow="authorization_code",
            credentials={"client_id": "ac-client", "client_secret": "ac-secret", "scopes": ["tools.call"]},
        )
        owner: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        stranger: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        anonymous: Final = gateway.client.post(
            f"/{alias}/mcp", headers=ACCEPT, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
        )
        assert anonymous.status_code == 401, anonymous.text
        metadata_url: Final = anonymous.headers["www-authenticate"].split('resource_metadata="')[1].rstrip('"')
        metadata: Final = httpx.get(metadata_url)
        assert metadata.status_code == 200 and metadata.json()["resource"] == f"{_base(gateway)}/{alias}/mcp"
        registered: Final = gateway.client.post(
            f"/{alias}/register", json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
        )
        assert registered.status_code in (200, 201), registered.text
        client_id: Final = str(_response_object(registered)["client_id"])
        pkce: Final = _Pkce(secrets.token_urlsafe(32))
        code: Final = _authorize_through_gateway(gateway, auth, alias, owner, client_id, pkce)
        wrong_verifier: Final = _redeem(gateway, alias, owner, client_id, code, _Pkce("wrong-" + pkce.verifier))
        assert wrong_verifier.status_code == 400, wrong_verifier.text
        assert tool_calls(peer.drain()) == ()
        code2: Final = _authorize_through_gateway(gateway, auth, alias, owner, client_id, pkce)
        redeemed: Final = _redeem(gateway, alias, owner, client_id, code2, pkce)
        assert redeemed.status_code == 200, redeemed.text
        issued: Final = redeemed.json()
        assert auth.is_live(_issued_token(issued))
        reused: Final = _redeem(gateway, alias, owner, client_id, code2, pkce)
        assert reused.status_code == 400, reused.text
        peer.drain()
        as_owner: Final = call_tool(gateway, owner, identity, f"{alias}-add", ADD)
        assert as_owner.status_code == 200, as_owner.text
        assert _authorizations(peer) == (f"Bearer {_issued_token(issued)}".encode(),)
        as_stranger: Final = call_tool(gateway, stranger, identity, f"{alias}-add", ADD)
        assert as_stranger.status_code == 401, as_stranger.text
        assert tool_calls(peer.drain()) == ()
        upstream_only: Final = gateway.client.post(
            f"/{alias}/mcp",
            headers={**ACCEPT, "Authorization": f"Bearer {_issued_token(issued)}"},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert upstream_only.status_code == 401, upstream_only.text
        assert tool_calls(peer.drain()) == ()
        refreshed: Final = gateway.client.post(
            f"/{alias}/token",
            headers={"x-litellm-api-key": owner},
            data={"grant_type": "refresh_token", "refresh_token": issued["refresh_token"], "client_id": client_id},
        )
        assert refreshed.status_code == 200, refreshed.text
        assert refreshed.json()["access_token"] != issued["access_token"]
        assert (
            read_rows(
                'SELECT 1 FROM "LiteLLM_MCPServerTable" WHERE server_id = %s AND credentials::text LIKE %s',
                (identity, "%ac-secret%"),
            )
            == []
        )


def test_authorization_request_without_pkce_is_refused_before_reaching_the_authorization_server(
    gateway: Gateway,
) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "br" + uuid.uuid4().hex[:8]
        identity: Final = _register_oauth(scenario, peer, auth, alias, auth_type="oauth_delegate", dcr_bridge=True)
        key: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        auth.drain()
        refused: Final = gateway.client.get(
            f"/{alias}/authorize",
            params={"client_id": "c", "redirect_uri": CLIENT_REDIRECT, "response_type": "code", "state": "s"},
            headers={"x-litellm-api-key": key},
        )
        assert refused.status_code == 400, refused.text
        assert "PKCE" in refused.text
        assert auth.drain() == ()


def test_dcr_bridge_relays_client_registration_and_advertises_gateway_endpoints(gateway: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, gateway.scenario() as scenario:
        alias: Final = "dcr" + uuid.uuid4().hex[:8]
        _register_oauth(scenario, peer, auth, alias, auth_type="oauth_delegate", dcr_bridge=True)
        auth.drain()
        registered: Final = gateway.client.post(
            f"/{alias}/register", json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
        )
        assert registered.status_code in (200, 201), registered.text
        assert registered.json()["client_id"].startswith("dcr-"), registered.text
        assert [(request.method, urlsplit(request.target).path) for request in auth.drain()] == [("POST", "/register")]
        resource: Final = gateway.client.get(f"/.well-known/oauth-protected-resource/{alias}/mcp")
        assert resource.status_code == 200, resource.text
        assert resource.json()["authorization_servers"] == [f"{_base(gateway)}/{alias}"]
        issuer: Final = gateway.client.get(f"/.well-known/oauth-authorization-server/{alias}/mcp")
        assert issuer.status_code == 200, issuer.text
        assert issuer.json()["authorization_endpoint"] == f"{_base(gateway)}/{alias}/authorize"
        assert issuer.json()["token_endpoint"] == f"{_base(gateway)}/{alias}/token"
        assert "S256" in issuer.json()["code_challenge_methods_supported"]


def _ui_session_cookie(gateway: Gateway, user_id: str) -> dict[str, str]:
    claims: Final = {"user_id": user_id, "login_method": "username_password", "exp": int(time.time()) + 600}
    return {"token": jwt.encode(claims, gateway.key, algorithm="HS256")}


def _gateway_session_sign_in(
    gateway: Gateway, user_id: str, resource: str | None = None
) -> tuple[dict[str, object], str]:
    registered: Final = gateway.client.post(
        "/register", json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
    )
    assert registered.status_code in (200, 201), registered.text
    client_id: Final = str(_response_object(registered)["client_id"])
    pkce: Final = _Pkce(secrets.token_urlsafe(48))
    cookies: Final = _ui_session_cookie(gateway, user_id)
    started: Final = gateway.client.get(
        "/authorize",
        params={
            "client_id": client_id,
            "redirect_uri": CLIENT_REDIRECT,
            "response_type": "code",
            "state": "lit6029",
            "code_challenge": pkce.challenge,
            "code_challenge_method": "S256",
            **({} if resource is None else {"resource": resource}),
        },
        cookies=cookies,
    )
    assert started.status_code == 303, started.text
    handle: Final = parse_qs(urlsplit(started.headers["location"]).query)["connect_flow"][0]
    completed: Final = gateway.client.post(
        "/authorize/complete", data={"flow": handle}, cookies={**cookies, **dict(started.cookies)}
    )
    assert completed.status_code == 303, completed.text
    callback: Final = parse_qs(urlsplit(completed.headers["location"]).query)
    assert "code" in callback, completed.headers["location"]
    issued: Final = gateway.client.post(
        "/token",
        data={
            "grant_type": "authorization_code",
            "code": callback["code"][0],
            "redirect_uri": CLIENT_REDIRECT,
            "client_id": client_id,
            "code_verifier": pkce.verifier,
        },
    )
    assert issued.status_code == 200, issued.text
    return _response_object(issued), client_id


def _gateway_session_bearer(gateway: Gateway, user_id: str, resource: str | None = None) -> str:
    issued, _ = _gateway_session_sign_in(gateway, user_id, resource)
    return _issued_token(issued)


def test_spec_client_discovers_the_aggregate_authorization_server_and_signs_in_to_a_session_bearer(
    gateway: Gateway,
) -> None:
    base: Final = _base(gateway)
    protected: Final = gateway.client.get("/.well-known/oauth-protected-resource/mcp")
    assert protected.status_code == 200, protected.text
    assert _response_object(protected) == {
        "authorization_servers": [f"{base}/mcp"],
        "resource": f"{base}/mcp",
        "scopes_supported": [],
    }, protected.text
    authorization_server: Final = gateway.client.get("/.well-known/oauth-authorization-server/mcp")
    assert authorization_server.status_code == 200, authorization_server.text
    authorization_metadata: Final = _response_object(authorization_server)
    assert authorization_metadata == {
        "issuer": f"{base}/mcp",
        "authorization_endpoint": f"{base}/authorize/mcp-session",
        "token_endpoint": f"{base}/token",
        "introspection_endpoint": f"{base}/introspect",
        "registration_endpoint": f"{base}/register",
        "response_types_supported": ["code"],
        "scopes_supported": [],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none", "client_secret_post"],
    }, authorization_server.text
    registration_endpoint: Final = urlsplit(str(authorization_metadata["registration_endpoint"])).path
    authorization_endpoint: Final = urlsplit(str(authorization_metadata["authorization_endpoint"])).path
    token_endpoint: Final = urlsplit(str(authorization_metadata["token_endpoint"])).path
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "rt5" + uuid.uuid4().hex[:10]
        identity: Final = register_mcp(scenario, peer, alias)
        member: Final = scenario.member(scenario.team(object_permission={"mcp_servers": [identity]}))
        registered: Final = gateway.client.post(
            registration_endpoint, json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
        )
        assert registered.status_code in (200, 201), registered.text
        client_id: Final = str(_response_object(registered)["client_id"])
        pkce: Final = _Pkce(secrets.token_urlsafe(48))
        cookies: Final = _ui_session_cookie(gateway, member)
        started: Final = gateway.client.get(
            authorization_endpoint,
            params={
                "client_id": client_id,
                "redirect_uri": CLIENT_REDIRECT,
                "response_type": "code",
                "state": "rt5-state",
                "code_challenge": pkce.challenge,
                "code_challenge_method": "S256",
                "resource": f"{base}/mcp",
            },
            cookies=cookies,
        )
        assert started.status_code == 303, started.text
        connect_location: Final = urlsplit(started.headers["location"])
        assert (
            connect_location.scheme + "://" + connect_location.netloc + connect_location.path == f"{base}/ui/connect"
        ), started.headers["location"]
        connect_flow: Final = parse_qs(connect_location.query)["connect_flow"][0]
        flow_cookie: Final = f"mcp_connect_flow_{connect_flow}"
        assert flow_cookie in started.cookies, started.headers
        flow_cookies: Final = {**cookies, flow_cookie: started.cookies[flow_cookie]}
        described: Final = gateway.client.get("/authorize/flow", params={"flow": connect_flow}, cookies=flow_cookies)
        assert described.status_code == 200, described.text
        assert _response_object(described) == {
            "state": "unscoped",
            "client_origin": "http://127.0.0.1:9",
            "server_id": None,
            "server_name": None,
            "connected": None,
        }, described.text
        completed: Final = gateway.client.post("/authorize/complete", data={"flow": connect_flow}, cookies=flow_cookies)
        assert completed.status_code == 303, completed.text
        callback: Final = parse_qs(urlsplit(completed.headers["location"]).query)
        assert callback["state"] == ["rt5-state"], completed.headers["location"]
        assert set(callback) == {"code", "state"}, callback
        token: Final = gateway.client.post(
            token_endpoint,
            data={
                "grant_type": "authorization_code",
                "code": callback["code"][0],
                "redirect_uri": CLIENT_REDIRECT,
                "client_id": client_id,
                "code_verifier": pkce.verifier,
            },
        )
        assert token.status_code == 200, token.text
        token_body: Final = _response_object(token)
        assert set(token_body) == {"access_token", "token_type", "expires_in", "refresh_token"}, token.text
        assert token_body["token_type"] == "Bearer", token.text
        assert isinstance(token_body["expires_in"], int) and token_body["expires_in"] > 0, token.text
        assert str(token_body["access_token"]).startswith("llm_session_"), token.text
        assert str(token_body["refresh_token"]).startswith("llm_srefresh_"), token.text
        bearer: Final = str(token_body["access_token"])
        initialized: Final = gateway.client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {bearer}", **ACCEPT},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": dict(INITIALIZE)},
        )
        assert initialized.status_code == 200, initialized.text
        open_before: Final = _open_aliases()
        listed_response: Final = gateway.client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {bearer}", **ACCEPT},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert listed_response.status_code == 200, listed_response.text
        listed: Final = _outcome_from_rpc(listed_response)
        assert listed.ok, listed.raw
        open_after: Final = open_before | _open_aliases()
        assert _without_foreign_open_servers(listed.tools, open_after) == {
            f"{alias}-add",
            f"{alias}-multiply",
            f"{alias}-fail",
        }, listed.tools
        peer.drain()
        called_response: Final = gateway.client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {bearer}", **ACCEPT},
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": f"{alias}-add", "arguments": {"a": 20, "b": 22}},
            },
        )
        assert called_response.status_code == 200, called_response.text
        called: Final = _outcome_from_rpc(called_response)
        assert called.ok and called.text == "42", called.raw
        calls: Final = tool_calls(peer.drain())
        assert len(calls) == 1, calls
        _assert_upstream_call(calls[0], "add", {"a": 20, "b": 22})


def test_rotated_session_refresh_token_cannot_be_replayed_after_revocation(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "rt6" + uuid.uuid4().hex[:10]
        identity: Final = register_mcp(scenario, peer, alias)
        member: Final = scenario.member(scenario.team(object_permission={"mcp_servers": [identity]}))
        issued, client_id = _gateway_session_sign_in(gateway, member)
        r1: Final = str(issued["refresh_token"])
        a1: Final = str(issued["access_token"])
        assert r1.startswith("llm_srefresh_"), r1
        assert a1.startswith("llm_session_"), a1
        rotated: Final = gateway.client.post(
            "/token",
            data={"grant_type": "refresh_token", "refresh_token": r1, "client_id": client_id},
        )
        assert rotated.status_code == 200, rotated.text
        rotated_body: Final = _response_object(rotated)
        assert set(rotated_body) == {"access_token", "token_type", "expires_in", "refresh_token"}, rotated.text
        r2: Final = str(rotated_body["refresh_token"])
        a2: Final = str(rotated_body["access_token"])
        assert r2 != r1 and r2.startswith("llm_srefresh_"), rotated.text
        assert a2 != a1 and a2.startswith("llm_session_"), rotated.text
        replayed: Final = gateway.client.post(
            "/token",
            data={"grant_type": "refresh_token", "refresh_token": r1, "client_id": client_id},
        )
        assert replayed.status_code == 400, replayed.text
        assert _response_object(replayed) == {
            "error": "invalid_grant",
            "error_description": "the refresh token was already used",
        }, replayed.text
        revoked: Final = gateway.client.post("/revoke", data={"token": r2, "client_id": client_id})
        assert revoked.status_code == 200, revoked.text
        assert _response_object(revoked) == {}, revoked.text
        revoked_again: Final = gateway.client.post("/revoke", data={"token": r2, "client_id": client_id})
        assert revoked_again.status_code == 200, revoked_again.text
        assert _response_object(revoked_again) == {}, revoked_again.text
        after_revoke: Final = gateway.client.post(
            "/token",
            data={"grant_type": "refresh_token", "refresh_token": r2, "client_id": client_id},
        )
        assert after_revoke.status_code == 400, after_revoke.text
        assert _response_object(after_revoke) == {
            "error": "invalid_grant",
            "error_description": "the refresh token was already used",
        }, after_revoke.text
        unknown_client: Final = gateway.client.post(
            "/revoke", data={"token": r2, "client_id": "dcr-unknown-" + uuid.uuid4().hex}
        )
        assert unknown_client.status_code == 401, unknown_client.text
        assert _response_object(unknown_client) == {
            "error": "invalid_client",
            "error_description": "unknown or malformed client_id",
        }, unknown_client.text
        open_before: Final = _open_aliases()
        listed_response: Final = gateway.client.post(
            "/mcp",
            headers={"Authorization": f"Bearer {a2}", **ACCEPT},
            json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
        )
        assert listed_response.status_code == 200, listed_response.text
        listed: Final = _outcome_from_rpc(listed_response)
        assert listed.ok, listed.raw
        open_after: Final = open_before | _open_aliases()
        assert _without_foreign_open_servers(listed.tools, open_after) == {
            f"{alias}-add",
            f"{alias}-multiply",
            f"{alias}-fail",
        }, listed.tools


def _consent_form_fields(page: str) -> dict[str, str]:
    return dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)">', page))


def test_lite_logout_revokes_the_cli_refresh_token_sent_the_way_the_cli_sends_it(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_id: Final = scenario.team()
        member: Final = scenario.member(team_id)
        discovered: Final = gateway.client.get("/.well-known/litellm-cli-auth")
        assert discovered.status_code == 200, discovered.text
        contract: Final = _response_object(discovered)
        base: Final = _base(gateway)
        assert (contract["resource"], contract["token_endpoint"], contract["revocation_endpoint"]) == (
            base,
            f"{base}/token",
            f"{base}/revoke",
        ), discovered.text
        resource: Final = str(contract["resource"])
        registered: Final = gateway.client.post(
            str(contract["registration_endpoint"]),
            json={
                "client_name": "LiteLLM CLI",
                "redirect_uris": [CLIENT_REDIRECT],
                "grant_types": ["authorization_code", "refresh_token"],
                "response_types": ["code"],
                "token_endpoint_auth_method": "none",
            },
        )
        assert registered.status_code in (200, 201), registered.text
        client_id: Final = str(_response_object(registered)["client_id"])
        pkce: Final = _Pkce(secrets.token_urlsafe(64))
        cookies: Final = _ui_session_cookie(gateway, member)
        consent: Final = gateway.client.get(
            str(contract["authorization_endpoint"]),
            params={
                "response_type": "code",
                "client_id": client_id,
                "redirect_uri": CLIENT_REDIRECT,
                "state": "lite-login-state",
                "code_challenge": pkce.challenge,
                "code_challenge_method": "S256",
                "resource": resource,
            },
            cookies=cookies,
        )
        assert consent.status_code == 200, consent.text
        consent_fields: Final = _consent_form_fields(consent.text)
        assert set(consent_fields) == {"flow", "team_id"} and consent_fields["team_id"] == team_id, consent.text
        approved: Final = gateway.client.post(
            "/authorize/complete",
            data={**consent_fields, "decision": "approve"},
            cookies={**cookies, **dict(consent.cookies)},
        )
        assert approved.status_code == 303, approved.text
        callback: Final = parse_qs(urlsplit(approved.headers["location"]).query)
        assert callback["state"] == ["lite-login-state"], approved.headers["location"]
        issued: Final = gateway.client.post(
            str(contract["token_endpoint"]),
            data={
                "grant_type": "authorization_code",
                "code": callback["code"][0],
                "redirect_uri": CLIENT_REDIRECT,
                "client_id": client_id,
                "code_verifier": pkce.verifier,
                "resource": resource,
            },
        )
        assert issued.status_code == 200, issued.text
        issued_body: Final = _response_object(issued)
        assert (issued_body["user_id"], issued_body["team_id"]) == (member, team_id), issued.text
        r1: Final = str(issued_body["refresh_token"])
        assert r1.startswith("llm_srefresh_"), issued.text

        def refresh(refresh_token: str) -> httpx.Response:
            return gateway.client.post(
                str(contract["token_endpoint"]),
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "resource": resource,
                },
            )

        def revoke(refresh_token: str) -> httpx.Response:
            return gateway.client.post(
                str(contract["revocation_endpoint"]),
                data={"token": refresh_token, "token_type_hint": "refresh_token", "client_id": client_id},
            )

        rotated: Final = refresh(r1)
        assert rotated.status_code == 200, rotated.text
        rotated_body: Final = _response_object(rotated)
        r2: Final = str(rotated_body["refresh_token"])
        assert r2 != r1 and r2.startswith("llm_srefresh_"), rotated.text
        assert (rotated_body["user_id"], rotated_body["team_id"]) == (member, team_id), rotated.text
        replayed: Final = refresh(r1)
        assert replayed.status_code == 400, replayed.text
        assert _response_object(replayed) == {
            "error": "invalid_grant",
            "error_description": "the refresh token was already used",
        }, replayed.text
        logged_out: Final = revoke(r2)
        assert logged_out.status_code == 200, logged_out.text
        assert _response_object(logged_out) == {}, logged_out.text
        logged_out_again: Final = revoke(r2)
        assert logged_out_again.status_code == 200, logged_out_again.text
        assert _response_object(logged_out_again) == {}, logged_out_again.text
        after_logout: Final = refresh(r2)
        assert after_logout.status_code == 400, after_logout.text
        assert _response_object(after_logout) == {
            "error": "invalid_grant",
            "error_description": "the refresh token was already used",
        }, after_logout.text
        completion: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "lite logout access token"}]},
            key=str(rotated_body["access_token"]),
        )
        assert completion.status_code == 200, completion.text


def _toolset_rpc(gateway: Gateway, bearer: str, name: str, method: str, params: dict[str, object]) -> Outcome:
    def post(rpc_method: str, rpc_params: dict[str, object]) -> httpx.Response:
        return gateway.client.post(
            f"/toolset/{name}/mcp",
            json={"jsonrpc": "2.0", "id": 1, "method": rpc_method, "params": rpc_params},
            headers={"Authorization": f"Bearer {bearer}", "Accept": "application/json, text/event-stream"},
        )

    initialized: Final = _outcome_from_rpc(post("initialize", dict(INITIALIZE)))
    if not initialized.ok:
        return initialized
    return _outcome_from_rpc(post(method, params))


def test_gateway_session_bearer_of_a_team_member_is_served_the_team_toolset_on_its_route(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "lit6029sess" + uuid.uuid4().hex[:6]
        server_id: Final = register_mcp(scenario, peer, alias)
        granted_name: Final = "lit6029g" + uuid.uuid4().hex[:8]
        withheld_name: Final = "lit6029w" + uuid.uuid4().hex[:8]
        granted_id: Final = create_toolset(scenario, ((server_id, "add"),), toolset_name=granted_name)
        create_toolset(scenario, ((server_id, "multiply"),), toolset_name=withheld_name)
        member: Final = scenario.member(scenario.team(object_permission={"mcp_toolsets": [granted_id]}))
        bearer: Final = _gateway_session_bearer(gateway, member)
        assert bearer.startswith("llm_session_"), bearer[:16]
        listed: Final = _toolset_rpc(gateway, bearer, granted_name, "tools/list", {})
        assert listed.tools == (f"{alias}-add",), listed.raw
        peer.drain()
        called: Final = _toolset_rpc(
            gateway, bearer, granted_name, "tools/call", {"name": f"{alias}-add", "arguments": {"a": 4, "b": 5}}
        )
        assert called.ok and called.text == "9", called.raw
        assert len(tool_calls(peer.drain())) == 1
        denied: Final = _toolset_rpc(gateway, bearer, withheld_name, "tools/list", {})
        assert denied.status == 403, denied.raw
        assert tool_calls(peer.drain()) == ()


def test_resource_scoped_session_bearer_opens_a_team_toolset_inside_its_server_and_none_outside(
    gateway: Gateway,
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        inside: Final = "lit6029in" + uuid.uuid4().hex[:6]
        outside: Final = "lit6029out" + uuid.uuid4().hex[:6]
        inside_server: Final = register_mcp(scenario, peer, inside)
        outside_server: Final = register_mcp(scenario, peer, outside)
        inside_name: Final = "lit6029i" + uuid.uuid4().hex[:8]
        outside_name: Final = "lit6029o" + uuid.uuid4().hex[:8]
        inside_id: Final = create_toolset(scenario, ((inside_server, "add"),), toolset_name=inside_name)
        outside_id: Final = create_toolset(scenario, ((outside_server, "add"),), toolset_name=outside_name)
        member: Final = scenario.member(scenario.team(object_permission={"mcp_toolsets": [inside_id, outside_id]}))
        bearer: Final = _gateway_session_bearer(gateway, member, resource=f"{_base(gateway)}/{inside}/mcp")
        assert bearer.startswith("llm_session_"), bearer[:16]
        listed: Final = _toolset_rpc(gateway, bearer, inside_name, "tools/list", {})
        assert listed.tools == (f"{inside}-add",), listed.raw
        peer.drain()
        called: Final = _toolset_rpc(
            gateway, bearer, inside_name, "tools/call", {"name": f"{inside}-add", "arguments": {"a": 4, "b": 5}}
        )
        assert called.ok and called.text == "9", called.raw
        assert len(tool_calls(peer.drain())) == 1
        refused: Final = _toolset_rpc(gateway, bearer, outside_name, "tools/list", {})
        assert refused.status == 403, refused.raw
        assert tool_calls(peer.drain()) == ()


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


def test_token_exchange_callers_with_different_subject_tokens_own_separate_listings(echo_rig: Gateway) -> None:
    with mcp_peer() as peer, oauth_server() as auth, echo_rig.scenario() as scenario:
        alias: Final = "te" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(
            scenario,
            peer,
            alias,
            auth_type="oauth2_token_exchange",
            token_exchange_endpoint=auth.issuer + "/token",
            credentials={"client_id": "te-client", "client_secret": "te-secret"},
        )
        key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
        first_subject: Final = "subject-" + uuid.uuid4().hex
        first: Final = McpCaller(echo_rig, key, "mcp", alias, {"Authorization": f"Bearer {first_subject}"})
        second: Final = McpCaller(echo_rig, key, "mcp", alias, {"Authorization": "Bearer subject-" + uuid.uuid4().hex})
        auth.drain()
        assert first.list_tools().ok
        assert [request["subject_token"] for request in auth.token_requests()] == [first_subject]
        probe: Final = {"probe": _PROBE}
        own: Final = _echoed_description(first.call(f"{alias}-add", probe))
        other: Final = _echoed_description(second.call(f"{alias}-add", probe))
        assert (own, other) == ("Add two integers", _UNLISTED), (
            "the caller bearer is part of the identity on a token-exchange server: one subject, one slot"
        )
        assert second.list_tools().ok
        assert _echoed_description(second.call(f"{alias}-add", probe)) == "Add two integers"
        assert tool_calls(peer.drain()) == (), "a blocked probe reached the peer"


@dataclass(frozen=True, slots=True)
class _IdpRig:
    gateway: Gateway
    private_key: RSAPrivateKey
    other_private_key: RSAPrivateKey
    kid: str
    dead_jwks: Wire


def _proxy_config_with_general_settings(directory: Path, name: str, overrides: Mapping[str, object]) -> Path:
    raw_config: Final[object] = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config: Final = TypeAdapter(dict[str, object]).validate_python(raw_config)
    general_settings: Final = TypeAdapter(dict[str, object]).validate_python(config["general_settings"])
    merged: Final = {**config, "general_settings": {**general_settings, **overrides}}
    config_path: Final = directory / name
    config_path.write_text(yaml.safe_dump(merged))
    return config_path


def _idp_proxy_config(directory: Path, team_id_jwt_field: str | None = None) -> Path:
    return _proxy_config_with_general_settings(
        directory,
        "mcp_idp.yaml",
        {
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "user_id_jwt_field": "sub",
                "user_id_upsert": True,
                **({} if team_id_jwt_field is None else {"team_id_jwt_field": team_id_jwt_field}),
            },
            "use_x_forwarded_for": True,
            "mcp_trusted_proxy_ranges": ["127.0.0.1/32"],
        },
    )


def _signed_subject_token(private_key: RSAPrivateKey, claims: Mapping[str, str], kid: str) -> str:
    now: Final = int(time.time())
    return jwt.encode({**claims, "iat": now, "exp": now + 300}, private_key, algorithm="RS256", headers={"kid": kid})


def _token_exchange_form(client_id: str, subject_token: str) -> dict[str, str]:
    return {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "client_id": client_id,
        "subject_token": subject_token,
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
    }


def _registered_dcr_client(gateway: Gateway) -> str:
    registered: Final = gateway.client.post(
        "/register", json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
    )
    assert registered.status_code in (200, 201), registered.text
    return str(_response_object(registered)["client_id"])


@contextmanager
def _idp_gateway(directory: Path, team_id_jwt_field: str | None, workers: int) -> Iterator[_IdpRig]:
    assert os.environ.get("LITELLM_LICENSE"), (
        "enable_jwt_auth is enterprise-only: these tests need LITELLM_LICENSE (CI forwards it); "
        "without it /token answers 400 'JWT auth is an enterprise only feature; no license is set' "
        "and the failure reads as a regression instead of a missing license"
    )
    private_key: Final = generate_private_key(public_exponent=65537, key_size=2048)
    other_private_key: Final = generate_private_key(public_exponent=65537, key_size=2048)
    kid: Final = "kid-a"
    public_jwk: Final = TypeAdapter(dict[str, object]).validate_python(
        json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    )

    def respond_good(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=json.dumps({"keys": [{**public_jwk, "kid": kid}]}).encode())

    def respond_dead(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(drop_connection=True)

    config_path: Final = _idp_proxy_config(directory, team_id_jwt_field)
    with (
        wire_server(respond_good) as good_jwks,
        wire_server(respond_dead) as dead_jwks,
        gateway_from_environment() as gateway,
        owned_proxy(
            gateway,
            directory,
            {"JWT_PUBLIC_KEY_URL": f"{good_jwks.url},{dead_jwks.url}"},
            config=config_path,
            remove_environment=("PROXY_BASE_URL",),
            workers=workers,
        ) as candidate,
    ):
        yield _IdpRig(candidate, private_key, other_private_key, kid, dead_jwks)


@pytest.fixture(scope="module")
def idp_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_IdpRig]:
    with _idp_gateway(tmp_path_factory.mktemp("mcp-idp"), None, workers=2) as rig:
        yield rig


@pytest.fixture(scope="module")
def idp_team_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_IdpRig]:
    with _idp_gateway(tmp_path_factory.mktemp("mcp-idp-team"), "team_id", workers=1) as rig:
        yield rig


@pytest.fixture(scope="module")
def forwarded_gateway(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("mcp-forwarded")
    config: Final = _proxy_config_with_general_settings(
        directory,
        "mcp_forwarded.yaml",
        {"use_x_forwarded_for": True, "mcp_trusted_proxy_ranges": ["127.0.0.1/32"]},
    )
    with (
        gateway_from_environment() as gateway,
        owned_proxy(
            gateway,
            directory,
            {"FORWARDED_ALLOW_IPS": "192.0.2.1"},
            config=config,
            remove_environment=("PROXY_BASE_URL",),
            workers=1,
        ) as candidate,
    ):
        yield candidate


@pytest.fixture(
    scope="module",
    params=(
        {},
        {"use_x_forwarded_for": True, "mcp_trusted_proxy_ranges": ["192.0.2.0/24"]},
    ),
    ids=("xff-off", "outside-trusted-range"),
)
def untrusted_gateway(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("mcp-untrusted-forwarded")
    config: Final = _proxy_config_with_general_settings(
        directory,
        "mcp_untrusted.yaml",
        TypeAdapter(dict[str, object]).validate_python(request.param),
    )
    with (
        gateway_from_environment() as gateway,
        owned_proxy(
            gateway,
            directory,
            {"FORWARDED_ALLOW_IPS": "192.0.2.1"},
            config=config,
            remove_environment=("PROXY_BASE_URL",),
            workers=1,
        ) as candidate,
    ):
        yield candidate


def test_idp_subject_token_exchange_mints_a_credential_for_the_mapped_user_and_refuses_bad_subjects(
    idp_rig: _IdpRig,
) -> None:
    registered: Final = idp_rig.gateway.client.post(
        "/register", json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
    )
    assert registered.status_code in (200, 201), registered.text
    client_id: Final = str(_response_object(registered)["client_id"])
    subject: Final = "mcp-idp-user-" + uuid.uuid4().hex

    def token(private_key: RSAPrivateKey, user_id: str, kid: str) -> str:
        now: Final = int(time.time())
        return jwt.encode(
            {"sub": user_id, "iat": now, "exp": now + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": kid},
        )

    valid_subject: Final = token(idp_rig.private_key, subject, idp_rig.kid)
    exchange_form: Final = {
        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
        "client_id": client_id,
        "subject_token": valid_subject,
        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
    }
    id_token_exchange: Final = idp_rig.gateway.client.post(
        "/token",
        data={
            **exchange_form,
            "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
        },
    )
    assert id_token_exchange.status_code == 200, id_token_exchange.text
    id_token_exchange_body: Final = _response_object(id_token_exchange)
    assert set(id_token_exchange_body) == {
        "access_token",
        "token_type",
        "expires_in",
        "refresh_token",
        "user_id",
        "team_id",
        "issued_token_type",
    }, id_token_exchange.text
    assert id_token_exchange_body["token_type"] == "Bearer", id_token_exchange.text
    assert id_token_exchange_body["issued_token_type"] == "urn:ietf:params:oauth:token-type:access_token", (
        id_token_exchange.text
    )
    assert id_token_exchange_body["user_id"] == subject, id_token_exchange.text
    assert id_token_exchange_body["team_id"] is None, id_token_exchange.text
    assert isinstance(id_token_exchange_body["expires_in"], int) and id_token_exchange_body["expires_in"] > 0, (
        id_token_exchange.text
    )
    assert str(id_token_exchange_body["refresh_token"]).startswith("llm_srefresh_"), id_token_exchange.text
    exchanged: Final = idp_rig.gateway.client.post("/token", data=exchange_form)
    assert exchanged.status_code == 200, exchanged.text
    outage_form: Final = {
        **exchange_form,
        "subject_token": token(idp_rig.other_private_key, subject, "kid-b"),
    }
    unavailable: Final = idp_rig.gateway.client.post("/token", data=outage_form)
    dead_requests: Final = idp_rig.dead_jwks.drain()
    assert unavailable.status_code == 503, unavailable.text
    assert _response_object(unavailable) == {
        "error": "temporarily_unavailable",
        "error_description": (
            "the gateway could not verify subject_token because its identity provider or database is unavailable; retry"
        ),
    }, unavailable.text
    assert tuple(request.method for request in dead_requests) == ("GET", "GET", "GET"), dead_requests
    exchanged_body: Final = _response_object(exchanged)
    assert set(exchanged_body) == {
        "access_token",
        "token_type",
        "expires_in",
        "refresh_token",
        "user_id",
        "team_id",
        "issued_token_type",
    }, exchanged.text
    assert exchanged_body["token_type"] == "Bearer", exchanged.text
    assert exchanged_body["issued_token_type"] == "urn:ietf:params:oauth:token-type:access_token", exchanged.text
    assert exchanged_body["user_id"] == subject, exchanged.text
    assert exchanged_body["team_id"] is None, exchanged.text
    assert isinstance(exchanged_body["expires_in"], int) and exchanged_body["expires_in"] > 0, exchanged.text
    assert str(exchanged_body["refresh_token"]).startswith("llm_srefresh_"), exchanged.text
    access_token: Final = str(exchanged_body["access_token"])
    id_token_access_token: Final = str(id_token_exchange_body["access_token"])
    with idp_rig.gateway.scenario() as scenario:
        model: Final = scenario.model()
        completion: Final = idp_rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "mcp idp spend"}]},
            key=access_token,
        )
        assert completion.status_code == 200, completion.text
        request_id: Final = str(_response_object(completion)["id"])
        id_token_completion: Final = idp_rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "mcp idp id token spend"}]},
            key=id_token_access_token,
        )
        assert id_token_completion.status_code == 200, id_token_completion.text
        id_token_request_id: Final = str(_response_object(id_token_completion)["id"])
        request_ids: Final = tuple(sorted((request_id, id_token_request_id)))
        spend: Final = eventually(
            lambda: read_rows(
                "SELECT request_id, metadata->>'user_api_key_user_id' AS user_api_key_user_id "
                'FROM "LiteLLM_SpendLogs" WHERE request_id IN (%s, %s) ORDER BY request_id',
                request_ids,
            ),
            lambda rows: len(rows) == 2,
            seconds=70,
        )
        assert spend == [
            {"request_id": request_ids[0], "user_api_key_user_id": subject},
            {"request_id": request_ids[1], "user_api_key_user_id": subject},
        ], spend
    user: Final = eventually(
        lambda: read_rows('SELECT user_id, user_role, teams FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,)),
        lambda rows: len(rows) == 1,
        seconds=70,
    )
    assert user == [{"user_id": subject, "user_role": None, "teams": []}], user
    access_token_hash: Final = hashlib.sha256(access_token.encode()).hexdigest()
    verification_tokens: Final = read_rows(
        'SELECT token, user_id FROM "LiteLLM_VerificationToken" WHERE token = %s',
        (access_token_hash,),
    )
    assert verification_tokens == [], verification_tokens
    missing_subject: Final = idp_rig.gateway.client.post(
        "/token",
        data={key: value for key, value in exchange_form.items() if key != "subject_token"},
    )
    assert missing_subject.status_code == 400, missing_subject.text
    assert _response_object(missing_subject) == {
        "error": "invalid_request",
        "error_description": "subject_token and subject_token_type are required",
    }, missing_subject.text
    rejected_subject: Final = idp_rig.gateway.client.post(
        "/token", data={**exchange_form, "subject_token": token(idp_rig.other_private_key, subject, idp_rig.kid)}
    )
    assert rejected_subject.status_code == 400, rejected_subject.text
    assert _response_object(rejected_subject) == {
        "error": "invalid_request",
        "error_description": "subject_token was rejected by the gateway's JWT auth",
    }, rejected_subject.text
    non_jwt: Final = idp_rig.gateway.client.post("/token", data={**exchange_form, "subject_token": "not-a-jwt"})
    assert non_jwt.status_code == 400, non_jwt.text
    assert _response_object(non_jwt) == {
        "error": "invalid_request",
        "error_description": "subject_token is not a JWT",
    }, non_jwt.text


def test_idp_subject_token_exchange_binds_the_team_claim_and_refuses_a_signed_subject_with_no_user(
    idp_team_rig: _IdpRig,
) -> None:
    client_id: Final = _registered_dcr_client(idp_team_rig.gateway)
    subject: Final = "mcp-idp-team-user-" + uuid.uuid4().hex
    with idp_team_rig.gateway.scenario() as scenario:
        model: Final = scenario.model()
        team_id: Final = scenario.team()
        exchanged: Final = idp_team_rig.gateway.client.post(
            "/token",
            data=_token_exchange_form(
                client_id,
                _signed_subject_token(idp_team_rig.private_key, {"sub": subject, "team_id": team_id}, idp_team_rig.kid),
            ),
        )
        assert exchanged.status_code == 200, exchanged.text
        body: Final = _response_object(exchanged)
        assert (body["user_id"], body["team_id"], body["issued_token_type"]) == (
            subject,
            team_id,
            "urn:ietf:params:oauth:token-type:access_token",
        ), exchanged.text
        completion: Final = idp_team_rig.gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "user", "content": "mcp idp team spend"}]},
            key=str(body["access_token"]),
        )
        assert completion.status_code == 200, completion.text
        request_id: Final = str(_response_object(completion)["id"])
        spend: Final = eventually(
            lambda: read_rows(
                "SELECT team_id, metadata->>'user_api_key_user_id' AS user_api_key_user_id, "
                "metadata->>'user_api_key_team_id' AS user_api_key_team_id "
                'FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                (request_id,),
            ),
            lambda rows: len(rows) == 1,
            seconds=70,
        )
        assert spend == [{"team_id": team_id, "user_api_key_user_id": subject, "user_api_key_team_id": team_id}], (
            spend
        )
        no_user: Final = idp_team_rig.gateway.client.post(
            "/token",
            data=_token_exchange_form(
                client_id, _signed_subject_token(idp_team_rig.private_key, {"team_id": team_id}, idp_team_rig.kid)
            ),
        )
        assert no_user.status_code == 400, no_user.text
        assert _response_object(no_user) == {
            "error": "invalid_request",
            "error_description": "subject_token names no user the gateway knows",
        }, no_user.text


def test_idp_jwks_server_error_answers_temporarily_unavailable(tmp_path: Path) -> None:
    pytest.skip(
        "BUG: an IdP JWKS answering HTTP 503 makes the gateway /token token-exchange answer 400 invalid_request "
        "(subject_token rejected) instead of 503 temporarily_unavailable"
    )
    assert os.environ.get("LITELLM_LICENSE"), (
        "enable_jwt_auth is enterprise-only: these tests need LITELLM_LICENSE (CI forwards it); "
        "without it /token answers 400 'JWT auth is an enterprise only feature; no license is set' "
        "and the failure reads as a regression instead of a missing license"
    )
    private_key: Final = generate_private_key(public_exponent=65537, key_size=2048)
    kid: Final = "mcp-runtime-503"

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(status=503, body=b'{"error":"jwks_down"}')

    config: Final = _idp_proxy_config(tmp_path)
    with (
        wire_server(respond) as jwks,
        gateway_from_environment() as gateway,
        owned_proxy(
            gateway,
            tmp_path,
            {"JWT_PUBLIC_KEY_URL": jwks.url},
            config=config,
            remove_environment=("PROXY_BASE_URL",),
            workers=2,
        ) as idp_gateway,
    ):
        registered: Final = idp_gateway.client.post(
            "/register", json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
        )
        assert registered.status_code in (200, 201), registered.text
        client_id: Final = str(_response_object(registered)["client_id"])
        now: Final = int(time.time())
        subject_token: Final = jwt.encode(
            {"sub": "mcp-idp-http-503-" + uuid.uuid4().hex, "iat": now, "exp": now + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": kid},
        )
        exchange_form: Final = {
            "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
            "client_id": client_id,
            "subject_token": subject_token,
            "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
        }
        response: Final = idp_gateway.client.post("/token", data=exchange_form)
        jwks_requests: Final = jwks.drain()
        assert response.status_code == 503, (
            f"request={exchange_form!r}, jwks_requests={jwks_requests!r}, "
            f"response_status={response.status_code}, response={response.text}"
        )
        assert _response_object(response) == {
            "error": "temporarily_unavailable",
            "error_description": (
                "the gateway could not verify subject_token because its identity provider or database is unavailable; "
                "retry"
            ),
        }, response.text


def test_standard_pattern_discovery_names_the_per_server_issuer_and_follows_forwarded_headers(
    gateway: Gateway, forwarded_gateway: Gateway
) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "rt8" + uuid.uuid4().hex[:10]
        register_mcp(
            scenario,
            peer,
            alias,
            auth_type="oauth2",
            oauth2_flow="authorization_code",
            credentials={
                "client_id": "rt8-client",
                "client_secret": "rt8-secret",
                "scopes": ["tools.call"],
            },
        )
        base: Final = _base(gateway)
        protected: Final = gateway.client.get(f"/.well-known/oauth-protected-resource/mcp/{alias}")
        assert protected.status_code == 200, protected.text
        aggregate_issuer: Final = f"{base}/mcp"
        expected_protected: Final = {
            "authorization_servers": [aggregate_issuer],
            "resource": f"{base}/mcp/{alias}",
            "scopes_supported": ["tools.call"],
        }
        assert _response_object(protected) == expected_protected, protected.text
        aggregate_authorization: Final = gateway.client.get("/.well-known/oauth-authorization-server/mcp")
        assert aggregate_authorization.status_code == 200, aggregate_authorization.text
        assert _response_object(aggregate_authorization)["issuer"] == aggregate_issuer, aggregate_authorization.text
        authorization: Final = gateway.client.get(f"/.well-known/oauth-authorization-server/mcp/{alias}")
        assert authorization.status_code == 200, authorization.text
        expected_authorization: Final = {
            "issuer": f"{base}/mcp/{alias}",
            "authorization_endpoint": f"{base}/{alias}/authorize",
            "token_endpoint": f"{base}/{alias}/token",
            "response_types_supported": ["code"],
            "scopes_supported": ["tools.call"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
            "registration_endpoint": f"{base}/{alias}/register",
        }
        assert _response_object(authorization) == expected_authorization, authorization.text
        legacy_authorization: Final = gateway.client.get(f"/.well-known/oauth-authorization-server/{alias}")
        assert legacy_authorization.status_code == 200, legacy_authorization.text
        assert _response_object(legacy_authorization) == {**expected_authorization, "issuer": f"{base}/{alias}"}, (
            legacy_authorization.text
        )
        legacy_protected: Final = gateway.client.get(f"/.well-known/oauth-protected-resource/{alias}/mcp")
        assert legacy_protected.status_code == 200, legacy_protected.text
        assert _response_object(legacy_protected) == {
            "authorization_servers": [f"{base}/mcp"],
            "resource": f"{base}/{alias}/mcp",
            "scopes_supported": ["tools.call"],
        }, legacy_protected.text
        root_protected: Final = gateway.client.get("/.well-known/oauth-protected-resource")
        assert root_protected.status_code == 200, root_protected.text
        assert _response_object(root_protected) == {
            "authorization_servers": [f"{base}/mcp"],
            "resource": base,
            "scopes_supported": [],
        }, root_protected.text
        anonymous: Final = gateway.client.post(
            f"/mcp/{alias}",
            headers=ACCEPT,
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": dict(INITIALIZE)},
        )
        assert anonymous.status_code == 401, anonymous.text
        challenge: Final = anonymous.headers["www-authenticate"]
        metadata_ref: Final = re.search(r'resource_metadata="([^"]+)"', challenge)
        assert metadata_ref is not None, challenge
        metadata: Final = gateway.client.get(urljoin(base + "/", metadata_ref.group(1)))
        assert metadata.status_code == 200, metadata.text
        assert _response_object(metadata) == expected_protected, metadata.text
    with mcp_peer() as peer, forwarded_gateway.scenario() as scenario:
        alias: Final = "rt8f" + uuid.uuid4().hex[:9]
        register_mcp(
            scenario,
            peer,
            alias,
            auth_type="oauth2",
            oauth2_flow="authorization_code",
            credentials={"client_id": "rt8f-client", "client_secret": "rt8f-secret"},
        )
        forwarded_base: Final = "https://gw.example.test"
        forwarded_headers: Final = {
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "gw.example.test",
        }
        forwarded_prm: Final = forwarded_gateway.client.get(
            f"/.well-known/oauth-protected-resource/mcp/{alias}", headers=forwarded_headers
        )
        assert forwarded_prm.status_code == 200, forwarded_prm.text
        assert _response_object(forwarded_prm) == {
            "authorization_servers": [f"{forwarded_base}/mcp"],
            "resource": f"{forwarded_base}/mcp/{alias}",
            "scopes_supported": [],
        }, forwarded_prm.text
        forwarded_as: Final = forwarded_gateway.client.get(
            f"/.well-known/oauth-authorization-server/mcp/{alias}", headers=forwarded_headers
        )
        assert forwarded_as.status_code == 200, forwarded_as.text
        assert _response_object(forwarded_as) == {
            "issuer": f"{forwarded_base}/mcp/{alias}",
            "authorization_endpoint": f"{forwarded_base}/{alias}/authorize",
            "token_endpoint": f"{forwarded_base}/{alias}/token",
            "response_types_supported": ["code"],
            "scopes_supported": [],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
            "registration_endpoint": f"{forwarded_base}/{alias}/register",
        }, forwarded_as.text
        local_base: Final = _base(forwarded_gateway)
        local_as: Final = forwarded_gateway.client.get(f"/.well-known/oauth-authorization-server/mcp/{alias}")
        assert local_as.status_code == 200, local_as.text
        assert _response_object(local_as) == {
            "issuer": f"{local_base}/mcp/{alias}",
            "authorization_endpoint": f"{local_base}/{alias}/authorize",
            "token_endpoint": f"{local_base}/{alias}/token",
            "response_types_supported": ["code"],
            "scopes_supported": [],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
            "registration_endpoint": f"{local_base}/{alias}/register",
        }, local_as.text


def test_standard_pattern_discovery_ignores_forwarded_headers_from_an_untrusted_peer(
    untrusted_gateway: Gateway,
) -> None:
    with mcp_peer() as peer, untrusted_gateway.scenario() as scenario:
        alias: Final = "rt8u" + uuid.uuid4().hex[:9]
        register_mcp(
            scenario,
            peer,
            alias,
            auth_type="oauth2",
            oauth2_flow="authorization_code",
            credentials={
                "client_id": "rt8u-client",
                "client_secret": "rt8u-secret",
                "scopes": ["tools.call"],
            },
        )
        base: Final = _base(untrusted_gateway)
        untrusted_headers: Final = {
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "evil.example",
        }
        authorization: Final = untrusted_gateway.client.get(
            f"/.well-known/oauth-authorization-server/mcp/{alias}", headers=untrusted_headers
        )
        assert authorization.status_code == 200, authorization.text
        assert _response_object(authorization) == {
            "issuer": f"{base}/mcp/{alias}",
            "authorization_endpoint": f"{base}/{alias}/authorize",
            "token_endpoint": f"{base}/{alias}/token",
            "response_types_supported": ["code"],
            "scopes_supported": ["tools.call"],
            "grant_types_supported": ["authorization_code", "refresh_token"],
            "code_challenge_methods_supported": ["S256"],
            "token_endpoint_auth_methods_supported": ["client_secret_post"],
            "registration_endpoint": f"{base}/{alias}/register",
        }, authorization.text
        assert "evil.example" not in authorization.text, authorization.text
        protected: Final = untrusted_gateway.client.get(
            f"/.well-known/oauth-protected-resource/mcp/{alias}", headers=untrusted_headers
        )
        assert protected.status_code == 200, protected.text
        assert _response_object(protected) == {
            "authorization_servers": [f"{base}/mcp"],
            "resource": f"{base}/mcp/{alias}",
            "scopes_supported": ["tools.call"],
        }, protected.text
        assert "evil.example" not in protected.text, protected.text
