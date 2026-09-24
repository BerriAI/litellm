import json
import time
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Final

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from jwt.algorithms import RSAAlgorithm

AUDIENCE: Final = "litellm-integration"
KEY_ID: Final = "integration-signing-key"
CLIENT_CLAIM: Final = "client_id"


def _proxy_config(directory: Path, model: str, upstream_url: str) -> Path:
    config: Final = directory / "jwt_mapped_key_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": "openai/" + model,
                            "api_base": upstream_url + "/v1",
                            "api_key": "sk-upstream",
                        },
                    }
                ],
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "store_model_in_db": True,
                    "proxy_batch_write_at": 1,
                    "proxy_batch_polling_interval": 1,
                    "enable_jwt_auth": True,
                    "litellm_jwtauth": {
                        "user_id_jwt_field": "sub",
                        "user_email_jwt_field": "email",
                        "virtual_key_claim_field": CLIENT_CLAIM,
                    },
                },
                "router_settings": {"disable_cooldowns": True},
            }
        )
    )
    return config


def _signed_token(private_key: rsa.RSAPrivateKey, user_id: str, email: str, client_id: str) -> str:
    now: Final = int(time.time())
    return jwt.encode(
        {"sub": user_id, "email": email, CLIENT_CLAIM: client_id, "aud": AUDIENCE, "iat": now, "exp": now + 300},
        private_key,
        algorithm="RS256",
        headers={"kid": KEY_ID},
    )


@pytest.mark.covers("authorization.jwt.mapped_key_backfills_null_user_email_from_claims")
def test_jwt_mapped_key_request_backfills_null_user_email_from_token_claims(gateway: Gateway, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = json.loads(RSAAlgorithm.to_jwk(private_key.public_key()))
    jwks: Final = json.dumps({"keys": [{**public_jwk, "kid": KEY_ID, "use": "sig", "alg": "RS256"}]}).encode()

    def respond(request: Request) -> Reply:
        assert request.target == "/jwks", request
        return Reply(body=jwks)

    model: Final = "integration-jwt-" + uuid.uuid4().hex
    with wire_server(respond) as issuer:
        config: Final = _proxy_config(tmp_path, model, gateway.upstream_url)
        overrides: Final = {"JWT_PUBLIC_KEY_URL": issuer.url + "/jwks", "JWT_AUDIENCE": AUDIENCE}
        with owned_proxy(gateway, tmp_path, overrides, config=config) as candidate, candidate.scenario() as scenario:
            user: Final = scenario.user()
            key: Final = scenario.key(user_id=user, models=[model])
            client_id: Final = "integration-client-" + uuid.uuid4().hex
            mapping: Final = candidate.post(
                "/jwt/key/mapping/new", {"jwt_claim_name": CLIENT_CLAIM, "jwt_claim_value": client_id, "key": key}
            )
            scenario.cleanups.callback(candidate.post, "/jwt/key/mapping/delete", {"id": mapping["id"]})
            assert read_rows('SELECT user_email FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,)) == [
                {"user_email": None}
            ]
            email: Final = f"{user}@integration.example"
            token: Final = _signed_token(private_key, user, email, client_id)
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt email backfill control"}]},
                key=token,
            )
            assert response.status_code == 200, response.text
            assert read_rows('SELECT user_email FROM "LiteLLM_UserTable" WHERE user_id = %s', (user,)) == [
                {"user_email": email}
            ]
            spend_rows: Final = eventually(
                lambda: read_rows(
                    'SELECT api_key, "user" FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
                    (str(response.json()["id"]),),
                ),
                lambda rows: len(rows) == 1,
                seconds=70,
            )
            assert spend_rows == [{"api_key": sha256(key.encode()).hexdigest(), "user": user}]
