import json
import time
import uuid
from pathlib import Path
from typing import Final

import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

pytestmark: Final = pytest.mark.shares_worker_state


@pytest.mark.parametrize("enabled", (True, False))
def test_rules_add_models_to_jwt_and_scoped_virtual_key_without_opening_other_models(
    gateway: Gateway, tmp_path: Path, enabled: bool
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
        virtual_key: Final = scenario.key(models=[explicit], metadata={"department": "science"})
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"] = {
            **config["general_settings"],
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "user_id_jwt_field": "sub",
                "user_id_upsert": True,
                "enforce_team_based_model_access": True,
            },
            **(
                {
                    "access_group_rules": [
                        {"attribute": "jwt_claims.groups", "contains": "science", "access_group": group_name},
                        {"attribute": "metadata.department", "equals": "science", "access_group": group_name},
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
        token: Final = jwt.encode(
            {"sub": subject, "groups": ["science"], "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": key_id},
        )
        unmatched: Final = jwt.encode(
            {"sub": subject, "groups": ["other"], "exp": int(time.time()) + 300},
            private_key,
            algorithm="RS256",
            headers={"kid": key_id},
        )
        with owned_proxy(gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=path, workers=2) as candidate:
            for key, model, allowed in (
                (token, group_model, enabled),
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
