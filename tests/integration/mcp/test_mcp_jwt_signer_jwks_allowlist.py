import base64
import json
import os
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Generator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa
from integration._support.client import Gateway, eventually
from integration._support.mcp import mcp_peer, register_mcp, tool_calls, tool_names
from integration._support.process import owned_proxy_process
from integration._support.wire import Reply, Request, wire_server
from jwt.algorithms import OKPAlgorithm, RSAAlgorithm


@contextmanager
def _jwt_signer_guardrail(gateway: Gateway, discovery_uri: str) -> Generator[None]:
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
def _idp_server(jwks_keys: list[Mapping[str, object]]) -> Generator[str]:
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
    public_jwk: Final = json.loads(OKPAlgorithm.to_jwk(private_key.public_key()))
    key: Final = {**public_jwk, "alg": "EdDSA", "kid": "ed"}
    token: Final = jwt.encode(_claims(), private_key, algorithm="EdDSA", headers={"kid": "ed"})
    return key, token


def _rs256_key_and_token() -> tuple[dict[str, object], str]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
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


def _rs256_keypair() -> tuple[rsa.RSAPrivateKey, dict[str, object]]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key, json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))


def _rs256_key_and_token_no_alg() -> tuple[dict[str, object], str]:
    private_key, public_jwk = _rs256_keypair()
    key: Final = {**public_jwk, "kid": "rsa"}
    token: Final = jwt.encode(_claims(), private_key, algorithm="RS256", headers={"kid": "rsa"})
    return key, token


def _signer_call(gateway: Gateway, key: str, identity: str, name: str, bearer: str) -> tuple[int, str]:
    try:
        response: Final = _call_with_bearer(gateway, key, identity, name, {"a": 4, "b": 5}, bearer)
        return response.status_code, response.text
    except httpx.HTTPError as error:
        return 0, repr(error)


def test_jwks_rs256_key_without_alg_field_still_verifies_the_token(gateway: Gateway) -> None:
    jwks_key, token = _rs256_key_and_token_no_alg()
    with _idp_server([jwks_key]) as discovery_uri, _jwt_signer_guardrail(gateway, discovery_uri):
        with mcp_peer() as peer, gateway.scenario() as scenario:
            identity: Final = register_mcp(scenario, peer, "jwksc4" + uuid.uuid4().hex[:8])
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            name: Final = tool_names(gateway, key, identity)["add"]
            peer.drain()
            response: Final = _call_with_bearer(gateway, key, identity, name, {"a": 4, "b": 5}, token)
            assert response.status_code == 200, response.text
            assert response.json()["content"][0]["text"] == "9", response.text
            assert len(tool_calls(peer.drain())) == 1, "accepted call never reached the peer"


def test_rs256_token_verifies_when_jwks_also_carries_a_non_approved_key(gateway: Gateway) -> None:
    okp_key, _eddsa_token = _eddsa_key_and_token()
    jwks_key, token = _rs256_key_and_token()
    with _idp_server([okp_key, jwks_key]) as discovery_uri, _jwt_signer_guardrail(gateway, discovery_uri):
        with mcp_peer() as peer, gateway.scenario() as scenario:
            identity: Final = register_mcp(scenario, peer, "jwksc5" + uuid.uuid4().hex[:8])
            key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
            name: Final = tool_names(gateway, key, identity)["add"]
            peer.drain()
            response: Final = _call_with_bearer(gateway, key, identity, name, {"a": 4, "b": 5}, token)
            assert response.status_code == 200, response.text
            assert response.json()["content"][0]["text"] == "9", response.text
            assert len(tool_calls(peer.drain())) == 1, "accepted call never reached the peer"


@pytest.mark.parametrize(
    ("label", "jwks_reply"),
    (
        ("jwks-404", Reply(status=404, body=b'{"error": "missing"}')),
        ("jwks-not-json", Reply(body=b"this is not json")),
    ),
    ids=("jwks-404", "jwks-not-json"),
)
def test_unusable_jwks_document_rejects_the_incoming_token(gateway: Gateway, label: str, jwks_reply: Reply) -> None:
    _, token = _rs256_key_and_token()

    def respond(request: Request) -> Reply:
        if request.target == "/.well-known/openid-configuration":
            return Reply(body=json.dumps({"jwks_uri": holder["url"] + "/.well-known/jwks.json"}).encode())
        return jwks_reply

    holder: Final = {"url": ""}
    with wire_server(respond) as idp:
        holder["url"] = idp.url
        with _jwt_signer_guardrail(gateway, idp.url + "/.well-known/openid-configuration"):
            with mcp_peer() as peer, gateway.scenario() as scenario:
                identity: Final = register_mcp(scenario, peer, "jwks" + uuid.uuid4().hex[:8])
                key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
                name: Final = tool_names(gateway, key, identity)["add"]
                peer.drain()
                status, text = _signer_call(gateway, key, identity, name, token)
                assert status == 401, (status, text)
                assert "incoming token verification failed" in text, text
                assert tool_calls(peer.drain()) == (), "rejected call reached the peer"
                alive: Final = gateway.client.get("/health/liveliness")
                assert alive.status_code == 200, alive.text


def test_signer_jwks_pause_rejects_then_recovers(gateway: Gateway, tmp_path: Path) -> None:
    private_key, public_jwk = _rs256_keypair()
    jwks_dir: Final = tmp_path / "jwks" / ".well-known"
    jwks_dir.mkdir(parents=True)
    port: Final = _reserve_port()
    (jwks_dir / "jwks.json").write_text(json.dumps({"keys": [{**public_jwk, "kid": "rsa", "alg": "RS256"}]}))
    (jwks_dir / "openid-configuration").write_text(
        json.dumps({"jwks_uri": f"http://127.0.0.1:{port}/.well-known/jwks.json"})
    )
    token: Final = jwt.encode(_claims(), private_key, algorithm="RS256", headers={"kid": "rsa"})
    server: Final = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1", "--directory", str(tmp_path / "jwks")],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with owned_proxy_process(gateway, tmp_path, {}) as owned:
            discovery: Final = f"http://127.0.0.1:{port}/.well-known/openid-configuration"
            with _jwt_signer_guardrail(owned.gateway, discovery):
                with mcp_peer() as peer, owned.gateway.scenario() as scenario:
                    identity: Final = register_mcp(scenario, peer, "jwksd3" + uuid.uuid4().hex[:8])
                    key: Final = scenario.key(object_permission={"mcp_servers": [identity]})
                    name: Final = tool_names(owned.gateway, key, identity)["add"]
                    peer.drain()
                    os.kill(server.pid, signal.SIGSTOP)
                    outcomes: Final = []
                    with ThreadPoolExecutor(max_workers=8) as pool:
                        futures = [
                            pool.submit(_signer_call, owned.gateway, key, identity, name, token) for _ in range(8)
                        ]
                        for future in futures:
                            outcomes.append(future.result(timeout=80))
                    assert all(status != 200 for status, _text in outcomes), outcomes
                    assert any(status in (0, 401, 500) for status, _text in outcomes), outcomes
                    os.kill(server.pid, signal.SIGCONT)
                    recovered: Final = eventually(
                        lambda: _signer_call(owned.gateway, key, identity, name, token),
                        lambda outcome: outcome[0] == 200,
                        seconds=70,
                    )
                    assert recovered[0] == 200, recovered
                    assert tool_calls(peer.drain()) != (), "resumed call never reached the peer"
    finally:
        try:
            os.kill(server.pid, signal.SIGCONT)
        except ProcessLookupError:
            pass
        server.terminate()
        server.wait(timeout=10)


def _reserve_port() -> int:
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        return reserve.getsockname()[1]
