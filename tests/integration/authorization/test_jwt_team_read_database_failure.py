"""Teams are written straight into Postgres so a JWT request is the proxy's first, uncached read of them."""

from __future__ import annotations

import json
import secrets
import time
import uuid
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import scratch_database, write_rows
from tests.integration._support.database_relay import (
    database_relay,
    dropped_connection_relay,
    plaintext_database_url,
)
from tests.integration._support.process import OwnedProxy, owned_proxy_process
from tests.integration._support.wire import Reply, Request, wire_server

ISSUER: Final = "https://accounts.google.com"
AUDIENCE: Final = "https://llm.example.internal"
KEY_ID: Final = "team-read-signing-key"
UPSTREAM_KEY: Final = "integration-provider-key"
WORKERS: Final = 2
MOCK_REPLY: Final = "Hello! This is a mock response from the fake OpenAI endpoint."


def _google_subject() -> str:
    return "1030950057" + "".join(secrets.choice("0123456789") for _ in range(11))


def _signed_subject(private_key: RSAPrivateKey, subject: str) -> str:
    now: Final = int(time.time())
    return jwt.encode(
        {"iss": ISSUER, "aud": AUDIENCE, "sub": subject, "iat": now, "exp": now + 600},
        private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )


def _jwt_team_config(directory: Path, model: str, upstream_url: str, jwks_url: str) -> Path:
    config: Final = directory / "jwt_team_id_field.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": (
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": "openai/gpt-4o-mini",
                            "api_base": f"{upstream_url}/v1",
                            "api_key": UPSTREAM_KEY,
                        },
                    },
                ),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "disable_model_info_refresh": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    "enable_jwt_auth": True,
                    "litellm_jwtauth": {
                        "issuers": (
                            {
                                "issuer": ISSUER,
                                "jwks_url": jwks_url,
                                "audience": AUDIENCE,
                                "team_id_jwt_field": "sub",
                            },
                        ),
                    },
                },
                "litellm_settings": {
                    "enable_redis_auth_cache": True,
                    "cache": True,
                    "cache_params": {"type": "redis", "host": "os.environ/REDIS_HOST", "port": "os.environ/REDIS_PORT"},
                },
            }
        )
    )
    return config


def _seed_teams(database_url: str, team_ids: tuple[str, ...]) -> None:
    for team_id in team_ids:
        write_rows(
            'INSERT INTO "LiteLLM_TeamTable" (team_id, team_alias, admins, members, models) '
            "VALUES (%s, %s, '{}', '{}', '{}')",
            (team_id, f"google-sa-{team_id[-4:]}"),
            database_url=database_url,
        )


def _chat(proxy: OwnedProxy, private_key: RSAPrivateKey, model: str, team_id: str) -> httpx.Response:
    return proxy.gateway.request(
        "POST",
        "/v1/chat/completions",
        {"model": model, "messages": [{"role": "user", "content": f"team read {team_id}"}]},
        key=_signed_subject(private_key, team_id),
    )


def _error(response: httpx.Response) -> Mapping[str, object]:
    body: Final = response.json()
    assert isinstance(body, dict) and isinstance(body.get("error"), dict), response.text
    return body["error"]


def _assert_served(response: httpx.Response) -> None:
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == MOCK_REPLY, response.text


@contextmanager
def _jwks(private_key: RSAPrivateKey) -> Generator[str]:
    public_jwk: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=json.dumps({"keys": [{**public_jwk, "kid": KEY_ID}]}).encode())

    with wire_server(respond) as server:
        yield f"{server.url}/{uuid.uuid4().hex}/jwks.json"


@contextmanager
def _jwt_proxy(gateway: Gateway, directory: Path, relayed_url: str, jwks_url: str, model: str) -> Generator[OwnedProxy]:
    with owned_proxy_process(
        gateway,
        directory,
        {"DATABASE_URL": plaintext_database_url(relayed_url), "PRISMA_HEALTH_WATCHDOG_ENABLED": "false"},
        config=_jwt_team_config(directory, model, gateway.upstream_url, jwks_url),
        remove_environment=("DATABASE_URL_READ_REPLICA",),
        workers=WORKERS,
    ) as owned:
        yield owned


@pytest.mark.timeout(300)
def test_an_existing_team_answers_503_while_the_database_is_down_and_is_served_once_it_returns(
    gateway: Gateway, tmp_path: Path
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    served_team: Final = _google_subject()
    unread_team: Final = _google_subject()
    missing_team: Final = _google_subject()
    model: Final = f"jwt-team-read-{uuid.uuid4().hex[:8]}"
    with (
        _jwks(private_key) as jwks_url,
        scratch_database() as database_url,
        database_relay(database_url, unread_team.encode()) as (relay, relayed_url),
        _jwt_proxy(gateway, tmp_path, relayed_url, jwks_url, model) as proxy,
    ):
        _seed_teams(database_url, (served_team, unread_team))
        _assert_served(_chat(proxy, private_key, model, served_team))

        missing: Final = _chat(proxy, private_key, model, missing_team)
        assert (missing.status_code, _error(missing).get("message")) == (
            401,
            "Authentication Error, invalid credentials",
        ), missing.text

        relay.arm()
        during: Final = _chat(proxy, private_key, model, unread_team)
        assert relay.tripped.is_set(), f"the team read never reached the database relay: {during.text}"
        assert (during.status_code, _error(during).get("type")) == (503, "no_db_connection"), during.text

        after: Final = eventually(
            lambda: _chat(proxy, private_key, model, unread_team),
            lambda response: response.status_code == 200,
            seconds=90,
            return_last_on_timeout=True,
        )
        assert relay.reconnected.is_set(), "the proxy never reconnected to the database"
        _assert_served(after)


@pytest.mark.timeout(300)
def test_a_team_read_that_loses_its_connection_once_is_retried_and_served(gateway: Gateway, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    served_team: Final = _google_subject()
    dropped_team: Final = _google_subject()
    model: Final = f"jwt-team-read-{uuid.uuid4().hex[:8]}"
    with (
        _jwks(private_key) as jwks_url,
        scratch_database() as database_url,
        dropped_connection_relay(database_url, dropped_team.encode()) as (relay, relayed_url),
        _jwt_proxy(gateway, tmp_path, relayed_url, jwks_url, model) as proxy,
    ):
        _seed_teams(database_url, (served_team, dropped_team))
        _assert_served(_chat(proxy, private_key, model, served_team))

        relay.arm_once()
        response: Final = _chat(proxy, private_key, model, dropped_team)
        assert relay.dropped.is_set(), f"the team read never reached the database relay: {response.text}"
        _assert_served(response)
