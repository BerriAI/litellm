import base64
import hashlib
import secrets
import uuid
from dataclasses import dataclass
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    EntryPoint,
    McpCaller,
    McpPeer,
    call_tool,
    mcp_peer,
    register_mcp,
    tool_calls,
)
from integration._support.oauth_server import AuthorizationServer, oauth_server

ADD: Final = {"a": 2, "b": 3}
CLIENT_REDIRECT: Final = "http://127.0.0.1:9/cb"
ACCEPT: Final = {"Accept": "application/json, text/event-stream"}


def _base(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


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
        client_id: Final = registered.json()["client_id"]
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
