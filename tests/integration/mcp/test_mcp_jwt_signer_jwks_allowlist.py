import base64
import json
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Final

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from integration._support.client import Gateway
from integration._support.mcp import mcp_peer, register_mcp, tool_calls, tool_names
from integration._support.wire import Reply, Request, wire_server


@contextmanager
def _jwt_signer_guardrail(gateway: Gateway, discovery_uri: str) -> Iterator[None]:
    created: Final = gateway.client.post(
        "/guardrails",
        headers={"x-litellm-api-key": gateway.key},
        json={
            "guardrail": {
                "guardrail_name": "signer" + uuid.uuid4().hex[:8],
                "litellm_params": {
                    "guardrail": "mcp_jwt_signer",
                    "mode": "pre_mcp_call",
                    "default_on": True,
                    "access_token_discovery_uri": discovery_uri,
                },
            }
        },
    )
    assert created.status_code == 200, created.text
    identity: Final = str(created.json()["guardrail_id"])
    try:
        yield
    finally:
        deleted: Final = gateway.client.delete(f"/guardrails/{identity}", headers={"x-litellm-api-key": gateway.key})
        assert deleted.status_code == 200, deleted.text


@contextmanager
def _idp_server(jwks_keys: list[Mapping[str, object]]) -> Iterator[str]:
    holder: Final = {"url": ""}

    def respond(request: Request) -> Reply:
        if request.target == "/.well-known/openid-configuration":
            return Reply(body=json.dumps({"jwks_uri": holder["url"] + "/.well-known/jwks.json"}).encode())
        if request.target == "/.well-known/jwks.json":
            return Reply(body=json.dumps({"keys": list(jwks_keys)}).encode())
        return Reply(status=404)

    with wire_server(respond) as idp:
        holder["url"] = idp.url
        yield idp.url + "/.well-known/openid-configuration"


def _claims() -> dict[str, object]:
    now: Final = int(time.time())
    return {"sub": "integration-mcp-user", "iat": now, "exp": now + 300}


def _hs256_key_and_token() -> tuple[dict[str, object], str]:
    secret: Final = b"integration-hs256-client-secret-0123456789abcdef"
    key: Final = {
        "kty": "oct",
        "alg": "HS256",
        "kid": "sym",
        "k": base64.urlsafe_b64encode(secret).rstrip(b"=").decode(),
    }
    token: Final = jwt.encode(_claims(), secret, algorithm="HS256", headers={"kid": "sym"})
    return key, token


def _eddsa_key_and_token() -> tuple[dict[str, object], str]:
    private_key: Final = ed25519.Ed25519PrivateKey.generate()
    public_jwk: Final = json.loads(jwt.algorithms.OKPAlgorithm.to_jwk(private_key.public_key()))
    key: Final = {**public_jwk, "alg": "EdDSA", "kid": "ed"}
    token: Final = jwt.encode(_claims(), private_key, algorithm="EdDSA", headers={"kid": "ed"})
    return key, token


def _rs256_key_and_token() -> tuple[dict[str, object], str]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    key: Final = {**public_jwk, "alg": "RS256", "kid": "rsa"}
    token: Final = jwt.encode(_claims(), private_key, algorithm="RS256", headers={"kid": "rsa"})
    return key, token


def _call_with_bearer(
    gateway: Gateway, key: str, identity: str, name: str, arguments: dict[str, object], bearer: str
) -> httpx.Response:
    return gateway.client.post(
        "/mcp-rest/tools/call",
        headers={"x-litellm-api-key": key, "Authorization": f"Bearer {bearer}"},
        json={"server_id": identity, "name": name, "arguments": arguments},
    )


@pytest.mark.parametrize("key_and_token", (_hs256_key_and_token, _eddsa_key_and_token), ids=("oct-HS256", "OKP-EdDSA"))
def test_jwks_key_with_non_approved_alg_cannot_verify_the_incoming_token(
    gateway: Gateway, key_and_token: Callable[[], tuple[dict[str, object], str]]
) -> None:
    jwks_key, token = key_and_token()
    with _idp_server([jwks_key]) as discovery_uri, _jwt_signer_guardrail(gateway, discovery_uri):
        with mcp_peer() as peer, gateway.scenario() as scenario:
            alias: Final = "jwksalg" + uuid.uuid4().hex[:8]
            identity: Final = register_mcp(scenario, peer, alias)
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            name: Final = tool_names(gateway, key, identity)["add"]
            peer.drain()
            response: Final = _call_with_bearer(gateway, key, identity, name, {"a": 4, "b": 5}, token)
            assert response.status_code == 401, response.text
            assert "incoming token verification failed" in response.text, response.text
            assert tool_calls(peer.drain()) == (), "rejected call reached the peer"


def test_jwks_rs256_key_still_verifies_the_incoming_token(gateway: Gateway) -> None:
    jwks_key, token = _rs256_key_and_token()
    with _idp_server([jwks_key]) as discovery_uri, _jwt_signer_guardrail(gateway, discovery_uri):
        with mcp_peer() as peer, gateway.scenario() as scenario:
            alias: Final = "jwksok" + uuid.uuid4().hex[:8]
            identity: Final = register_mcp(scenario, peer, alias)
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            name: Final = tool_names(gateway, key, identity)["add"]
            peer.drain()
            response: Final = _call_with_bearer(gateway, key, identity, name, {"a": 4, "b": 5}, token)
            assert response.status_code == 200, response.text
            assert response.json()["content"][0]["text"] == "9", response.text
            assert len(tool_calls(peer.drain())) == 1, "accepted call never reached the peer"
