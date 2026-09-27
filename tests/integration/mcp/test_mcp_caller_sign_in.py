import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import httpx
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway
from integration._support.mcp import (
    INITIALIZE,
    McpCaller,
    forget_mcp,
    mcp_peer,
    register_mcp,
    tool_calls,
)
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from jwt import algorithms as jwt_algorithms

ADD: Final = {"a": 2, "b": 3}
ACCEPT: Final = {"Accept": "application/json, text/event-stream"}


def _rpc(gateway: Gateway, path: str, key: str, headers: dict[str, str]) -> httpx.Response:
    return gateway.client.post(
        path,
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": INITIALIZE},
        headers={"x-litellm-api-key": key, **ACCEPT, **headers},
    )


def _sign_in_config(guardrail_params: dict[str, object], path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [{"guardrail_name": "signin" + uuid.uuid4().hex, "litellm_params": guardrail_params}]
    path.write_text(yaml.safe_dump(config))
    return path


def test_multi_server_connect_with_a_litellm_key_in_authorization_admits_an_obo_server(
    gateway: Gateway,
) -> None:
    with mcp_peer() as obo_peer, mcp_peer() as math_peer, gateway.scenario() as scenario:
        obo_alias: Final = "obo" + uuid.uuid4().hex[:8]
        math_alias: Final = "math" + uuid.uuid4().hex[:8]
        obo_id: Final = register_mcp(
            scenario,
            obo_peer,
            obo_alias,
            auth_type="oauth2_token_exchange",
            token_exchange_endpoint="http://127.0.0.1:9/token",
            credentials={"client_id": "obo-client", "client_secret": "obo-secret"},
        )
        math_id: Final = register_mcp(scenario, math_peer, math_alias)
        key: Final = scenario.key(object_permission={"mcp_servers": [obo_id, math_id]})
        caller: Final = McpCaller(
            gateway,
            key,
            "root",
            headers={
                "x-mcp-servers": f"{obo_alias},{math_alias}",
                "Authorization": f"Bearer {key}",
            },
        )
        init: Final = caller.initialize()
        assert init.ok, init.raw
        listed: Final = caller.list_tools()
        assert listed.ok, listed.raw
        assert any(name.endswith("add") for name in listed.tools), listed.tools


def test_alias_first_lookup_wins_over_a_server_whose_name_matches_the_alias(gateway: Gateway, tmp_path: Path) -> None:
    with mcp_peer() as math_peer, mcp_peer() as obo_peer:
        name: Final = "gh" + uuid.uuid4().hex[:6]
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["mcp_servers"] = {
            name: {
                "transport": "http",
                "url": obo_peer.url,
                "auth_type": "oauth2_token_exchange",
                "token_exchange_endpoint": "http://127.0.0.1:9/token",
                "credentials": {"client_id": "obo-client", "client_secret": "obo-secret"},
            }
        }
        path: Final = tmp_path / "collision.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy(gateway, tmp_path, {}, config=path) as candidate,
            candidate.scenario() as scenario,
        ):
            second: Final = candidate.request(
                "POST",
                "/v1/mcp/server",
                {"server_name": name + "_public", "alias": name, **math_peer.registration()},
            )
            assert second.status_code == 201, second.text
            second_id: Final = second.json()["server_id"]
            scenario.cleanups.callback(forget_mcp, candidate, second_id)
            key: Final = scenario.key(object_permission={"mcp_servers": [second_id]})

            init: Final = _rpc(candidate, f"/mcp/{name}", key, {})
            assert init.status_code == 200, init.text
            listing: Final = candidate.client.post(
                f"/mcp/{name}",
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
                headers={"x-litellm-api-key": key, **ACCEPT},
            )
            assert listing.status_code == 200, listing.text
            body: Final = json.loads(
                next(line[5:].strip() for line in listing.text.splitlines() if line.startswith("data:"))
            )
            names: Final = {tool["name"] for tool in body["result"]["tools"]}
            assert any(name.endswith("add") for name in names), names


def test_jwt_signer_verifies_the_bearer_that_admitted_the_call(gateway: Gateway, tmp_path: Path) -> None:
    signer_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk: Final = json.loads(jwt_algorithms.RSAAlgorithm.to_jwk(signer_key.public_key()))
    jwk["kid"] = "idp"
    holder: Final = []

    def idp(request: Request) -> Reply:
        issuer: Final = holder[0].url
        if request.target.endswith("/.well-known/openid-configuration"):
            return Reply(body=json.dumps({"issuer": issuer, "jwks_uri": issuer + "/jwks"}).encode())
        return Reply(body=json.dumps({"keys": [jwk]}).encode())

    def introspect(request: Request) -> Reply:
        return Reply(body=b'{"active": true, "sub": "subject-1"}')

    with (
        mcp_peer() as peer,
        wire_server(idp) as verify_idp,
        wire_server(introspect) as introspect_stub,
        gateway.scenario() as scenario,
    ):
        holder.append(verify_idp)
        alias: Final = "sig" + uuid.uuid4().hex[:8]
        identity: Final = register_mcp(scenario, peer, alias)
        config: Final = _sign_in_config(
            {
                "guardrail": "mcp_jwt_signer",
                "mode": "pre_mcp_call",
                "default_on": True,
                "access_token_discovery_uri": verify_idp.url + "/.well-known/openid-configuration",
                "token_introspection_endpoint": introspect_stub.url,
                "required_claims": ["sub"],
            },
            tmp_path / "signer.yaml",
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as owned:
            key: Final = owned.key(object_permission={"mcp_servers": [identity]})

            def call(headers: Mapping[str, str]) -> httpx.Response:
                return candidate.client.post(
                    "/mcp-rest/tools/call",
                    headers=dict(headers),
                    json={"name": "add", "arguments": ADD, "server_id": identity},
                )

            admitted: Final = call({"Authorization": f"Bearer {key}"})
            assert admitted.status_code == 200, admitted.text
            assert len(tool_calls(peer.drain())) == 1
            probes: Final = tuple(item for item in introspect_stub.drain() if item.body)
            assert any(key.encode() in probe.body for probe in probes), probes

            split: Final = call({"x-litellm-api-key": key})
            assert split.status_code == 403, split.text
            assert introspect_stub.drain() == ()


def test_agent_365_gated_server_challenges_at_connect_and_advertises_entra(gateway: Gateway, tmp_path: Path) -> None:
    def nothing(request: Request) -> Reply:
        return Reply(status=500)

    with wire_server(nothing) as api:
        config: Final = _sign_in_config(
            {
                "guardrail": "agent_365",
                "mode": "pre_mcp_call",
                "default_on": True,
                "tenant_id": "00000000-0000-0000-0000-000000000000",
                "client_id": "22222222-2222-2222-2222-222222222222",
                "client_secret": "secret",
                "api_base": api.url,
            },
            tmp_path / "agent365.yaml",
        )
        with (
            owned_proxy(gateway, tmp_path, {}, config=config) as candidate,
            mcp_peer() as peer,
            candidate.scenario() as scenario,
        ):
            alias: Final = "a365" + uuid.uuid4().hex[:8]
            identity: Final = register_mcp(scenario, peer, alias)
            granted: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            denied: Final = scenario.key(object_permission={"mcp_servers": ["no-mcp-servers"]})

            challenged: Final = _rpc(candidate, f"/mcp/{alias}", granted, {})
            assert challenged.status_code == 401, challenged.text
            authenticate: Final = challenged.headers.get("www-authenticate", "")
            assert f'resource_metadata="/.well-known/oauth-protected-resource/mcp/{alias}"' in authenticate
            assert 'error="invalid_token"' in authenticate

            discovery: Final = candidate.client.get(f"/.well-known/oauth-protected-resource/mcp/{alias}")
            assert discovery.status_code == 200, discovery.text
            document: Final = discovery.json()
            assert document["authorization_servers"] == [
                "https://login.microsoftonline.com/00000000-0000-0000-0000-000000000000/v2.0"
            ]
            assert document["scopes_supported"] == ["api://22222222-2222-2222-2222-222222222222/access_as_user"]

            refused: Final = _rpc(candidate, f"/mcp/{alias}", denied, {})
            assert refused.status_code == 403, refused.text
            assert "www-authenticate" not in refused.headers
            assert api.drain() == ()
