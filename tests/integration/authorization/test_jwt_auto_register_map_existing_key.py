import json
import os
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.integration._support.client import Gateway, eventually, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.database_relay import held_statement_relay
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

KEY_ID: Final = "integration-jwt-map-existing-key"
MAPPING_INSERT: Final = (b"INSERT INTO", b'"LiteLLM_JWTKeyMapping"')

pytestmark = pytest.mark.timeout(240)


def _hash(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def _config(directory: Path, claim_field: str) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {
        **config["general_settings"],
        "enable_jwt_auth": True,
        "litellm_jwtauth": {
            "user_id_jwt_field": "sub",
            "virtual_key_claim_field": claim_field,
            "unregistered_jwt_client_behavior": "auto_register",
            "auto_register_map_existing_key": True,
        },
    }
    path: Final = directory / f"jwt_map_existing_key_{claim_field}.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


@contextmanager
def _issuer() -> Iterator[tuple[rsa.RSAPrivateKey, str]]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    jwks: Final = json.dumps({"keys": [{**public_jwk, "kid": KEY_ID, "use": "sig", "alg": "RS256"}]}).encode()

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=jwks)

    with wire_server(respond) as server:
        yield private_key, server.url


def _token(private_key: rsa.RSAPrivateKey, subject: str, **claims: str) -> str:
    now: Final = int(time.time())
    return jwt.encode(
        {"sub": subject, **claims, "iat": now, "exp": now + 300},
        private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )


def _chat(candidate: Gateway, model: str, token: str) -> httpx.Response:
    return candidate.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": "map existing key control"}]},
        key=token,
    )


def _mapped_token(claim_name: str, claim_value: str) -> str:
    rows: Final = read_rows(
        'SELECT token FROM "LiteLLM_JWTKeyMapping" WHERE jwt_claim_name = %s AND jwt_claim_value = %s',
        (claim_name, claim_value),
    )
    assert len(rows) == 1, rows
    return string_value(rows[0]["token"])


def _user_key_hashes(user: str) -> frozenset[str]:
    rows: Final = read_rows('SELECT token FROM "LiteLLM_VerificationToken" WHERE user_id = %s', (user,))
    return frozenset(string_value(row["token"]) for row in rows)


def _billed_key(response: httpx.Response) -> str:
    rows: Final = eventually(
        lambda: read_rows(
            'SELECT api_key FROM "LiteLLM_SpendLogs" WHERE request_id = %s', (str(response.json()["id"]),)
        ),
        lambda values: len(values) == 1,
        seconds=70,
    )
    return string_value(rows[0]["api_key"])


def test_first_jwt_call_reuses_the_newest_durable_llm_key_and_skips_every_ineligible_newer_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        older_durable: Final = scenario.key(user_id=user)
        durable: Final = scenario.key(user_id=user)
        skipped: Final = {
            "older_durable": older_durable,
            "expiring": scenario.key(user_id=user, duration="1h"),
            "management_only": scenario.key(user_id=user, allowed_routes=["management_routes"]),
            "auto_registered_look_alike": scenario.key(user_id=user, metadata={"auto_registered": True}),
            "other_team": scenario.key(user_id=user, team_id=scenario.team()),
            "blocked": scenario.key(user_id=user),
        }
        gateway.post("/key/block", {"key": skipped["blocked"]})
        keys_before: Final = _user_key_hashes(user)

        with owned_proxy(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks_url}, config=_config(tmp_path, "sub")
        ) as candidate:
            response: Final = _chat(candidate, model, _token(private_key, user))

        assert response.status_code == 200, response.text
        mapped: Final = _mapped_token("sub", user)
        assert mapped == _hash(durable), {
            "mapped_to": next((name for name, key in skipped.items() if _hash(key) == mapped), mapped)
        }
        assert _user_key_hashes(user) == keys_before, "a key was minted although a reusable one existed"
        assert _billed_key(response) == _hash(durable)


def test_shared_client_claim_never_maps_a_second_user_onto_the_first_users_personal_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    with _issuer() as (private_key, jwks_url), gateway.scenario() as scenario:
        model: Final = scenario.model()
        first_user: Final = scenario.user(user_role="internal_user")
        second_user: Final = scenario.user(user_role="internal_user")
        personal: Final = scenario.key(user_id=first_user)
        client_id: Final = f"integration-shared-client-{uuid.uuid4().hex}"

        with owned_proxy(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks_url}, config=_config(tmp_path, "client_id")
        ) as candidate:
            first: Final = _chat(candidate, model, _token(private_key, first_user, client_id=client_id))
            second: Final = _chat(candidate, model, _token(private_key, second_user, client_id=client_id))

        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        mapped: Final = _mapped_token("client_id", client_id)
        assert mapped != _hash(personal), "the shared client claim was mapped to the first user's personal key"
        assert (_billed_key(first), _billed_key(second)) == (mapped, mapped)
        assert read_rows('SELECT request_id FROM "LiteLLM_SpendLogs" WHERE api_key = %s', (_hash(personal),)) == []


def test_concurrent_first_jwt_calls_of_a_keyless_user_both_succeed_on_one_surviving_mapped_key(
    gateway: Gateway, tmp_path: Path
) -> None:
    writer_url: Final = os.environ.get("INTEGRATION_PROXY_DATABASE_URL") or os.environ["DATABASE_URL"]
    with (
        _issuer() as (private_key, jwks_url),
        gateway.scenario() as scenario,
        held_statement_relay(writer_url, MAPPING_INSERT) as (relay, relayed_url),
    ):
        model: Final = scenario.model()
        user: Final = scenario.user(user_role="internal_user")
        token: Final = _token(private_key, user)
        overrides: Final = {
            "JWT_PUBLIC_KEY_URL": jwks_url,
            "DATABASE_URL": relayed_url,
            "PRISMA_HEALTH_WATCHDOG_ENABLED": "false",
        }

        with (
            owned_proxy(gateway, tmp_path, overrides, config=_config(tmp_path, "sub")) as candidate,
            ThreadPoolExecutor(max_workers=1) as pool,
        ):
            held_call: Final = pool.submit(_chat, candidate, model, token)
            assert relay.held.wait(60), "the first call never reached its mapping insert"
            racing: Final = _chat(candidate, model, token)
            relay.release()
            held: Final = held_call.result(timeout=60)

        assert racing.status_code == 200, racing.text
        assert held.status_code == 200, held.text
        keys: Final = _user_key_hashes(user)
        assert len(keys) == 1, keys
        assert _mapped_token("sub", user) in keys
        assert (_billed_key(held), _billed_key(racing)) == (_mapped_token("sub", user),) * 2
