import json
import time
import uuid
from pathlib import Path
from typing import Final

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa

from litellm.proxy.mcp_server.idp_token_exchange import REJECTED_SUBJECT_TOKEN
from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.shares_worker_state


@pytest.mark.parametrize(
    "enabled,mapping,claimed", ((True, False, True), (False, False, True), (True, True, False), (True, True, True))
)
def test_rules_add_models_to_jwt_and_scoped_virtual_key_without_opening_other_models(
    gateway: Gateway, tmp_path: Path, enabled: bool, mapping: bool, claimed: bool
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    key_id: Final = "rule-signing-key"

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=json.dumps({"keys": [{**public_key, "kid": key_id}]}).encode())

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        explicit: Final = scenario.model()
        group_model: Final = scenario.model()
        outside: Final = scenario.model()
        group_name: Final = f"integration-rule-{uuid.uuid4().hex}"
        created: Final = gateway.request(
            "POST", "/v1/access_group", {"access_group_name": group_name, "access_model_names": [group_model]}
        )
        assert created.status_code == 201, created.text
        group_id: Final = string_value(created.json()["access_group_id"])
        scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/access_group/{group_id}")
        virtual_user: Final = scenario.user()
        virtual_key: Final = scenario.key(models=[explicit], user_id=virtual_user)
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"] = {
            **config["general_settings"],
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "user_id_jwt_field": "sub",
                "team_ids_jwt_field": "groups",
                "user_id_upsert": True,
                "enforce_team_based_model_access": True,
                **(
                    {"virtual_key_claim_field": "sub", "unregistered_jwt_client_behavior": "auto_register"}
                    if mapping
                    else {}
                ),
            },
            **(
                {
                    "access_group_rules": [
                        {"attribute": "jwt_claims.groups", "contains": "science", "access_group": group_name},
                        {"attribute": "user_id", "equals": virtual_user, "access_group": group_name},
                    ]
                }
                if enabled
                else {}
            ),
        }
        path: Final = tmp_path / "access_group_rules.yaml"
        path.write_text(yaml.safe_dump(config))
        subject: Final = f"integration-rule-user-{uuid.uuid4().hex}"
        gateway.post("/user/new", {"user_id": subject, "user_role": "internal_user"})
        scenario.cleanups.callback(scenario.delete_user, subject)
        team_id: Final = (
            scenario.team(models=[explicit], members_with_roles=[{"role": "user", "user_id": subject}])
            if claimed
            else None
        )
        team_claims: Final = [team_id] if team_id is not None else []
        token: Final = jwt.encode(
            {"sub": subject, "groups": ["science", *team_claims], "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": key_id},
        )
        unmatched: Final = jwt.encode(
            {"sub": subject, "groups": ["other", *team_claims], "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": key_id},
        )
        with owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path, workers=2) as candidate:
            fallback_response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": group_model,
                    "fallbacks": [outside],
                    "messages": [{"role": "user", "content": "ungranted fallback control"}],
                },
                key=token,
            )
            assert fallback_response.status_code == 403, fallback_response.text
            for key, model, allowed in (
                (token, group_model, enabled and claimed),
                *(((token, group_model, enabled and claimed),) if mapping else ()),
                (token, outside, False),
                (unmatched, group_model, False),
                (virtual_key, explicit, True),
                (virtual_key, group_model, enabled),
                (virtual_key, outside, False),
            ):
                response: Final = candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "rule authorization control"}]},
                    key=key,
                )
                assert response.status_code == (200 if allowed else 403), response.text
                if allowed:
                    assert (
                        response.json()["choices"][0]["message"]["content"]
                        == "Hello! This is a mock response from the fake OpenAI endpoint."
                    ), response.text

            if mapping and claimed:
                rows: Final = read_rows(
                    'SELECT models, access_group_ids FROM "LiteLLM_VerificationToken" '
                    'WHERE user_id=%s AND team_id=%s',
                    (subject, team_id),
                )
                assert len(rows) == 1, rows
                assert group_model not in rows[0]["models"], rows
                assert not rows[0]["access_group_ids"], rows


@pytest.mark.parametrize("consumer", ("auto_register", "token_exchange"))
@pytest.mark.parametrize("matching", (True, False))
def test_teamless_jwt_cannot_provision_credentials_without_native_team_access(
    gateway: Gateway, tmp_path: Path, consumer: str, matching: bool
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_key: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key()))
    key_id: Final = "credential-rule-signing-key"

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=json.dumps({"keys": [{**public_key, "kid": key_id}]}).encode())

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        model: Final = scenario.model()
        subject: Final = f"integration-credential-user-{uuid.uuid4().hex}"
        gateway.post("/user/new", {"user_id": subject, "user_role": "internal_user"})
        scenario.cleanups.callback(scenario.delete_user, subject)
        group_name: Final = f"credential-rule-{uuid.uuid4().hex}"
        created_group: Final = gateway.request(
            "POST", "/v1/access_group", {"access_group_name": group_name, "access_model_names": [model]}
        )
        assert created_group.status_code == 201, created_group.text
        group_id: Final = string_value(created_group.json()["access_group_id"])
        scenario.cleanups.callback(gateway.request, "DELETE", f"/v1/access_group/{group_id}")
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"] = {
            **config["general_settings"],
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "user_id_jwt_field": "sub",
                "enforce_team_based_model_access": True,
                **(
                    {"virtual_key_claim_field": "sub", "unregistered_jwt_client_behavior": "auto_register"}
                    if consumer == "auto_register"
                    else {}
                ),
            },
            "access_group_rules": [
                {"attribute": "jwt_claims.groups", "contains": "science", "access_group": group_name}
            ],
        }
        path: Final = tmp_path / "credential_access_group_rules.yaml"
        path.write_text(yaml.safe_dump(config))
        token: Final = jwt.encode(
            {"sub": subject, "groups": ["science" if matching else "other"], "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": key_id},
        )
        with owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path, workers=2) as candidate:
            if consumer == "auto_register":
                for _ in range(2):
                    response: Final = candidate.request(
                        "POST",
                        "/v1/chat/completions",
                        {"model": model, "messages": [{"role": "user", "content": "credential rejection control"}]},
                        key=token,
                    )
                    assert response.status_code == 403, response.text
                    assert "No teams found in token" in response.text, response.text
            else:
                registered: Final = candidate.client.post(
                    "/register", json={"redirect_uris": ["http://localhost/callback"], "client_name": "rule-control"}
                )
                assert registered.status_code in (200, 201), registered.text
                exchanged: Final = candidate.client.post(
                    "/token",
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                        "client_id": registered.json()["client_id"],
                        "subject_token": token,
                        "subject_token_type": "urn:ietf:params:oauth:token-type:jwt",
                        "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
                    },
                )
                assert exchanged.status_code == 400, exchanged.text
                assert exchanged.json() == {"error": "invalid_request", "error_description": REJECTED_SUBJECT_TOKEN}, (
                    exchanged.text
                )
