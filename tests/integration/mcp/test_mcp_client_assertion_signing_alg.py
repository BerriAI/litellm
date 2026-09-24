import json
import uuid
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway
from integration._support.database import read_rows
from integration._support.mcp import forget_mcp, mcp_peer, register_mcp


def _pem_private_key() -> str:
    return (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        .decode()
    )


def _stored_client_assertion_signing_alg(identity: str) -> str:
    rows: Final = read_rows('SELECT credentials FROM "LiteLLM_MCPServerTable" WHERE server_id = %s', (identity,))
    assert len(rows) == 1, rows
    credentials: Final = rows[0]["credentials"]
    if credentials is None:
        return "RS256"
    blob: Final = credentials if isinstance(credentials, dict) else json.loads(credentials)
    return str(blob.get("client_assertion_signing_alg") or "RS256")


def test_non_approved_client_assertion_signing_alg_is_rejected_on_create_and_update(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "alg" + uuid.uuid4().hex[:8]
        created: Final = gateway.request(
            "POST",
            "/v1/mcp/server",
            {
                "server_name": alias,
                "alias": alias,
                **peer.registration(),
                "token_exchange_endpoint": "https://idp.integration.invalid/oauth2/token",
                "credentials": {
                    "client_private_key": _pem_private_key(),
                    "client_assertion_signing_alg": "HS256",
                },
            },
        )
        if created.status_code == 201:
            scenario.cleanups.callback(forget_mcp, gateway, str(created.json()["server_id"]))
        assert created.status_code in (400, 422), created.text
        assert "client_assertion_signing_alg" in created.text, created.text

        identity: Final = register_mcp(scenario, peer, alias + "v")
        edited: Final = gateway.request(
            "PUT",
            "/v1/mcp/server",
            {"server_id": identity, "credentials": {"client_assertion_signing_alg": "EdDSA"}},
        )
        assert edited.status_code in (400, 422), edited.text
        assert "client_assertion_signing_alg" in edited.text, edited.text
        assert _stored_client_assertion_signing_alg(identity) == "RS256"


def test_approved_client_assertion_signing_alg_round_trips(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        alias: Final = "algok" + uuid.uuid4().hex[:8]
        created: Final = gateway.request(
            "POST",
            "/v1/mcp/server",
            {
                "server_name": alias,
                "alias": alias,
                **peer.registration(),
                "token_exchange_endpoint": "https://idp.integration.invalid/oauth2/token",
                "credentials": {
                    "client_private_key": _pem_private_key(),
                    "client_assertion_signing_alg": "ES256",
                },
            },
        )
        assert created.status_code == 201, created.text
        identity: Final = str(created.json()["server_id"])
        scenario.cleanups.callback(forget_mcp, gateway, identity)
        assert _stored_client_assertion_signing_alg(identity) == "ES256"
