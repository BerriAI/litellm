import json
import time
import uuid
from pathlib import Path
from typing import Final

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

KEY_ID: Final = "integration-jwt-signing-key"
TEAM_BUDGET: Final = 25.0


def _jwks_reply(public_jwk: str) -> Reply:
    return Reply(body=json.dumps({"keys": [{**json.loads(public_jwk), "kid": KEY_ID}]}).encode())


@pytest.mark.covers("authorization.jwt.new_subject_without_team_claim_joins_default_team")
def test_jwt_subject_without_team_claim_is_provisioned_into_configured_default_team(
    gateway: Gateway, tmp_path: Path
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key())

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply(public_jwk)

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        team: Final = scenario.team()
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"] = {
            **config["general_settings"],
            "enable_jwt_auth": True,
            "litellm_jwtauth": {"user_id_jwt_field": "sub", "user_id_upsert": True},
        }
        config["litellm_settings"] = {
            **config["litellm_settings"],
            "default_internal_user_params": {
                "user_role": "internal_user",
                "teams": [{"team_id": team, "user_role": "user", "max_budget_in_team": TEAM_BUDGET}],
            },
        }
        path: Final = tmp_path / "jwt_default_team.yaml"
        path.write_text(yaml.safe_dump(config))
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = jwt.encode(
            {"sub": subject, "iat": int(time.time()), "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": KEY_ID},
        )
        with owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path) as candidate:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, subject)
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "default team control"}]},
                key=token,
            )
            assert response.status_code == 200, response.text
            assert response.json()["choices"][0]["message"]["content"] == (
                "Hello! This is a mock response from the fake OpenAI endpoint."
            ), response.text
            assert read_rows(
                'SELECT user_id, user_role, teams FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,)
            ) == [{"user_id": subject, "user_role": "internal_user", "teams": [team]}]
            memberships: Final = eventually(
                lambda: read_rows(
                    'SELECT m.team_id, b.max_budget FROM "LiteLLM_TeamMembership" m '
                    'JOIN "LiteLLM_BudgetTable" b ON b.budget_id = m.budget_id WHERE m.user_id = %s',
                    (subject,),
                ),
                lambda rows: len(rows) == 1,
            )
            assert memberships == [{"team_id": team, "max_budget": TEAM_BUDGET}]
            roster: Final = read_rows('SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team,))
            assert {"user_id": subject, "role": "user", "user_email": None} in roster[0]["members_with_roles"], roster
