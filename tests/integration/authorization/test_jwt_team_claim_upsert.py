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

pytestmark: Final = pytest.mark.shares_worker_state

KEY_ID: Final = "integration-jwt-team-claim-signing-key"


def _jwks_reply(public_jwk: str) -> Reply:
    return Reply(body=json.dumps({"keys": [{**json.loads(public_jwk), "kid": KEY_ID}]}).encode())


@pytest.mark.parametrize(
    "team_id_upsert", [True, False], ids=["upsert-creates-claimed-teams", "no-upsert-rejects-them"]
)
def test_jwt_teams_array_claim_creates_and_joins_every_team_only_when_team_id_upsert_is_on(
    gateway: Gateway, tmp_path: Path, team_id_upsert: bool
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key())

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply(public_jwk)

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"] = {
            **config["general_settings"],
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "user_id_jwt_field": "sub",
                "user_id_upsert": True,
                "team_ids_jwt_field": "teams",
                "team_id_upsert": team_id_upsert,
                "sync_user_role_and_teams": True,
            },
        }
        path: Final = tmp_path / "jwt_team_claim_upsert.yaml"
        path.write_text(yaml.safe_dump(config))
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        claimed_teams: Final = sorted(f"jwt-team-{subject}-{suffix}" for suffix in ("a", "b"))
        token: Final = jwt.encode(
            {"sub": subject, "teams": claimed_teams, "iat": int(time.time()), "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": KEY_ID},
        )
        with owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path) as candidate:
            model: Final = scenario.model()
            scenario.cleanups.callback(gateway.request, "POST", "/team/delete", {"team_ids": claimed_teams})
            response: Final = candidate.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "teams array claim"}]},
                key=token,
            )
            team_rows: Final = read_rows(
                'SELECT team_id FROM "LiteLLM_TeamTable" WHERE team_id = ANY(%s) ORDER BY team_id', (claimed_teams,)
            )
            if not team_id_upsert:
                assert (response.status_code, team_rows) == (403, []), response.text
                return
            scenario.cleanups.callback(scenario.delete_user, subject)
            assert response.status_code == 200, response.text
            user_teams: Final = eventually(
                lambda: read_rows('SELECT teams FROM "LiteLLM_UserTable" WHERE user_id = %s', (subject,)),
                lambda rows: [sorted(row["teams"]) for row in rows] == [claimed_teams],
            )
            assert ([sorted(row["teams"]) for row in user_teams], [row["team_id"] for row in team_rows]) == (
                [claimed_teams],
                claimed_teams,
            )
