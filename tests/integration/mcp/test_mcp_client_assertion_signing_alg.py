import json
import os
import uuid
from pathlib import Path
from typing import Final

import httpx
import psycopg
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.mcp import forget_mcp, mcp_peer, register_mcp
from integration._support.process import owned_proxy_process


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


def test_server_row_with_stale_client_assertion_signing_alg_still_loads(gateway: Gateway) -> None:
    """A row persisted before the allowlist (credentials.client_assertion_signing_alg = "HS256")
    must keep loading with the RS256 fallback instead of disappearing on upgrade."""
    identity: Final = "stalealg" + uuid.uuid4().hex[:8]
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_MCPServerTable" (server_id, server_name, url, transport, credentials,'
            " created_at, updated_at) VALUES (%s, %s, %s, %s, %s::jsonb, NOW(), NOW())",
            (
                identity,
                "stalealg" + uuid.uuid4().hex[:8],
                "https://mcp.integration.invalid",
                "http",
                json.dumps({"client_assertion_signing_alg": "HS256"}),
            ),
        )
    try:
        with mcp_peer() as peer, gateway.scenario() as scenario:
            register_mcp(scenario, peer, "staletrigger" + uuid.uuid4().hex[:8])
            listed: Final = eventually(
                lambda: gateway.request("GET", "/v1/mcp/server"),
                lambda response: (
                    response.status_code == 200
                    and any(server.get("server_id") == identity for server in response.json())
                ),
                seconds=30,
            )
            assert listed.status_code == 200, listed.text
    finally:
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute('DELETE FROM "LiteLLM_MCPServerTable" WHERE server_id = %s', (identity,))


APPROVED_ALGS: Final = ("RS256", "RS384", "RS512", "PS256", "PS384", "PS512", "ES256", "ES384", "ES512")


def _post_server(gateway: Gateway, name: str, credentials: object, omit_alg: bool = False) -> httpx.Response:
    blob: Final = {} if omit_alg else {"client_assertion_signing_alg": credentials}
    return gateway.request(
        "POST",
        "/v1/mcp/server",
        {
            "server_name": name,
            "alias": name,
            "url": "https://mcp.integration.invalid",
            "transport": "http",
            "credentials": blob,
        },
    )


def test_post_with_lowercase_hs256_is_rejected_naming_the_field(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        created: Final = _post_server(gateway, "b1hs" + uuid.uuid4().hex[:8], "hs256")
        if created.status_code == 201:
            scenario.cleanups.callback(forget_mcp, gateway, str(created.json()["server_id"]))
        assert created.status_code == 422, created.text
        assert "client_assertion_signing_alg" in created.text, created.text
        for algorithm in APPROVED_ALGS:
            assert algorithm in created.text, created.text


def test_put_with_eddsa_on_existing_server_is_rejected(gateway: Gateway) -> None:
    with mcp_peer() as peer, gateway.scenario() as scenario:
        identity: Final = register_mcp(scenario, peer, "b2ed" + uuid.uuid4().hex[:8])
        edited: Final = gateway.request(
            "PUT",
            "/v1/mcp/server",
            {"server_id": identity, "credentials": {"client_assertion_signing_alg": "EdDSA"}},
        )
        assert edited.status_code == 422, edited.text
        assert "client_assertion_signing_alg" in edited.text, edited.text


@pytest.mark.parametrize("algorithm", APPROVED_ALGS)
def test_each_approved_client_assertion_signing_alg_is_accepted_and_stored(gateway: Gateway, algorithm: str) -> None:
    with gateway.scenario() as scenario:
        name: Final = "b3" + algorithm.lower() + uuid.uuid4().hex[:6]
        created: Final = _post_server(gateway, name, algorithm)
        assert created.status_code == 201, created.text
        identity: Final = str(created.json()["server_id"])
        scenario.cleanups.callback(forget_mcp, gateway, identity)
        listed: Final = gateway.request("GET", "/v1/mcp/server")
        assert listed.status_code == 200, listed.text
        assert any(server.get("server_id") == identity for server in listed.json()), listed.text
        assert _stored_client_assertion_signing_alg(identity) == algorithm


@pytest.mark.parametrize(
    ("label", "value", "expected"),
    (
        ("empty", "", 422),
        ("five-kb", "x" * 5120, 422),
        ("integer", 7, 422),
        ("list", ["RS256"], 422),
        ("null", None, 201),
    ),
)
def test_client_assertion_signing_alg_payload_variants(
    gateway: Gateway, label: str, value: object, expected: int
) -> None:
    with gateway.scenario() as scenario:
        created: Final = _post_server(gateway, f"b{label}" + uuid.uuid4().hex[:8], value)
        if created.status_code == 201:
            scenario.cleanups.callback(forget_mcp, gateway, str(created.json()["server_id"]))
        assert created.status_code == expected, created.text
        if expected == 422 and label in ("empty", "five-kb"):
            assert "client_assertion_signing_alg" in created.text, created.text


def test_server_post_without_credentials_alg_key_is_accepted(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        created: Final = _post_server(gateway, "b9absent" + uuid.uuid4().hex[:8], None, omit_alg=True)
        assert created.status_code == 201, created.text
        scenario.cleanups.callback(forget_mcp, gateway, str(created.json()["server_id"]))


def test_unauthenticated_server_post_with_hs256_is_rejected(gateway: Gateway) -> None:
    response: Final = gateway.client.post(
        "/v1/mcp/server",
        json={
            "server_name": "b10unauth" + uuid.uuid4().hex[:8],
            "credentials": {"client_assertion_signing_alg": "hs256"},
        },
    )
    assert response.status_code == 401, response.text


def _seeded_row_loads_with_fallback_warning(gateway: Gateway, tmp_path: Path, algorithm: str) -> None:
    identity: Final = "dualread" + uuid.uuid4().hex[:8]
    name: Final = "dualread" + uuid.uuid4().hex[:8]
    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        connection.execute(
            'INSERT INTO "LiteLLM_MCPServerTable" (server_id, server_name, url, transport, credentials,'
            " created_at, updated_at) VALUES (%s, %s, %s, %s, %s::jsonb, NOW(), NOW())",
            (
                identity,
                name,
                "https://mcp.integration.invalid",
                "http",
                json.dumps({"client_assertion_signing_alg": algorithm}),
            ),
        )
    try:
        with owned_proxy_process(gateway, tmp_path, {}) as owned:
            listed: Final = eventually(
                lambda: owned.gateway.request("GET", "/v1/mcp/server"),
                lambda response: (
                    response.status_code == 200
                    and any(server.get("server_id") == identity for server in response.json())
                ),
                seconds=30,
            )
            assert listed.status_code == 200, listed.text
            log_text: Final = eventually(
                lambda: owned.log.read_text(),
                lambda text: name in text and "not an approved algorithm" in text and "using RS256" in text,
                seconds=30,
            )
            assert name in log_text and "not an approved algorithm" in log_text and "using RS256" in log_text
    finally:
        with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
            connection.execute('DELETE FROM "LiteLLM_MCPServerTable" WHERE server_id = %s', (identity,))


def test_seeded_hs256_row_loads_with_rs256_fallback_warning(gateway: Gateway, tmp_path: Path) -> None:
    _seeded_row_loads_with_fallback_warning(gateway, tmp_path, "HS256")


def test_seeded_eddsa_row_loads_with_rs256_fallback_warning(gateway: Gateway, tmp_path: Path) -> None:
    _seeded_row_loads_with_fallback_warning(gateway, tmp_path, "EdDSA")


def test_chat_still_works_after_alg_rejection(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        created: Final = _post_server(gateway, "b13chat" + uuid.uuid4().hex[:8], "hs256")
        assert created.status_code in (201, 422), created.text
        if created.status_code == 201:
            scenario.cleanups.callback(forget_mcp, gateway, str(created.json()["server_id"]))
        model: Final = scenario.model()
        body: Final = gateway.chat(model)
        assert body["id"], body
