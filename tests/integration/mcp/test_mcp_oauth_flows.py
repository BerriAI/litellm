import base64
import hashlib
import json
import re
import secrets
import textwrap
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

import httpx
import jwt
import pytest
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.database import read_rows
from integration._support.mcp import (
    ENTRY_POINTS,
    INITIALIZE,
    EntryPoint,
    McpCaller,
    McpPeer,
    ScriptedTool,
    scripted_peer,
    text_result,
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
    gateway: Gateway, auth: AuthorizationServer, alias: str, key: str, client_id: str, pkce: _Pkce,
    scope: str = "tools.call",
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
            "scope": scope,
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


def _ui_session_cookie(gateway: Gateway, user_id: str) -> dict[str, str]:
    claims: Final = {"user_id": user_id, "login_method": "username_password", "exp": int(time.time()) + 600}
    return {"token": jwt.encode(claims, gateway.key, algorithm="HS256")}


def _gateway_session_bearer(gateway: Gateway, user_id: str, resource: str | None = None) -> str:
    registered: Final = gateway.client.post(
        "/register", json={"redirect_uris": [CLIENT_REDIRECT], "client_name": "integration"}
    )
    assert registered.status_code in (200, 201), registered.text
    client_id: Final = registered.json()["client_id"]
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
    return _issued_token(issued.json())


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


@contextmanager
def _scope_enforcing_peer(auth: AuthorizationServer) -> Iterator[tuple[McpPeer, Wire]]:
    with scripted_peer(
        ScriptedTool("read_op", lambda _: text_result("read completed")),
        ScriptedTool("write_op", lambda _: text_result("write completed")),
        ScriptedTool("forbid_op", lambda _: text_result("must not execute")),
    ) as inner:
        def enforce(request: Request) -> Reply:
            body: Final = json.loads(request.body) if request.body else {}
            if body.get("method") == "tools/call":
                name: Final = body["params"]["name"]
                if name == "forbid_op":
                    return Reply(status=403, body=b"denied")
                required: Final = "tools.write" if name == "write_op" else "tools.read"
                token: Final = request.headers.get("authorization", "").removeprefix("Bearer ")
                with auth.lock:
                    granted: Final = auth.access_tokens.get(token, {}).get("scope", "")
                if required not in granted.split():
                    return Reply(status=403, headers={"WWW-Authenticate": f'Bearer error="insufficient_scope", scope="{required}"'})
            forwarded: Final = httpx.request(request.method, inner.url, content=request.body, headers=dict(request.headers))
            return Reply(status=forwarded.status_code, body=forwarded.content, content_type=forwarded.headers.get("content-type", "application/json"))

        with wire_server(enforce) as wire:
            yield McpPeer(wire.url + "/mcp", inner.calls), wire


@pytest.mark.parametrize("entry", ["rest", "server_mcp"])
def test_managed_scope_step_up_preserves_read_and_write_without_retry(gateway: Gateway, entry: EntryPoint) -> None:
    with oauth_server(scopes=("tools.read", "tools.write")) as auth, _scope_enforcing_peer(auth) as (peer, wire), gateway.scenario() as scenario:
        alias: Final = "scope" + uuid.uuid4().hex[:8]
        identity: Final = _register_oauth(
            scenario, peer, auth, alias, auth_type="oauth2", oauth2_flow="authorization_code",
            credentials={"client_id": "scope-client", "client_secret": "scope-secret", "scopes": ["tools.read"]},
        )
        owner: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        stranger: Final = scenario.key(user_id=scenario.user(), object_permission={"mcp_servers": [identity]})
        pkce: Final = _Pkce(secrets.token_urlsafe(32))
        code: Final = _authorize_through_gateway(gateway, auth, alias, owner, "scope-client", pkce, "tools.read")
        first: Final = _redeem(gateway, alias, owner, "scope-client", code, pkce)
        assert first.status_code == 200, first.text
        caller: Final = McpCaller(gateway, owner, entry, alias)
        read: Final = caller.call(f"{alias}-read_op", {}, identity)
        assert read.ok and read.text == "read completed", read.raw
        peer.drain()
        wire.drain()
        denied: Final = (
            call_tool(gateway, owner, identity, f"{alias}-write_op", {}) if entry == "rest"
            else caller.rpc("tools/call", {"name": f"{alias}-write_op", "arguments": {}})
        )
        assert denied.status_code == 403, denied.text
        challenge: Final = denied.headers["www-authenticate"]
        assert 'error="insufficient_scope"' in challenge and 'scope="tools.write"' in challenge
        assert f'resource_metadata="{_base(gateway)}/.well-known/oauth-protected-resource/' in challenge
        attempts: Final = tuple(request for request in wire.drain() if request.body and json.loads(request.body).get("method") == "tools/call")
        assert len(attempts) == 1 and tool_calls(peer.drain()) == (), "denied operation was retried or executed"
        second_code: Final = _authorize_through_gateway(gateway, auth, alias, owner, "scope-client", pkce, "tools.write")
        second: Final = _redeem(gateway, alias, owner, "scope-client", second_code, pkce)
        assert second.status_code == 200, second.text
        with auth.lock:
            granted: Final = auth.access_tokens[_issued_token(second.json())]["scope"]
        assert granted.split() == ["tools.read", "tools.write"]
        for operation in ("read", "write"):
            result: Final = caller.call(f"{alias}-{operation}_op", {}, identity)
            assert result.ok and result.text == f"{operation} completed", result.raw
        forbidden: Final = caller.call(f"{alias}-forbid_op", {}, identity)
        assert forbidden.status_code == 200 and not forbidden.ok, forbidden.raw
        unauthorized: Final = McpCaller(gateway, stranger, entry, alias).call(f"{alias}-read_op", {}, identity)
        assert unauthorized.status_code == 401, unauthorized.raw
