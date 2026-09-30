import json
import time
import uuid
from pathlib import Path
from typing import Final

import jwt
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa

from tests.integration._support.client import Gateway
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

KEY_ID: Final = "integration-routing-override-key"
ISSUER: Final = "https://idp.example.com"


def test_jwt_routing_override_sends_only_claim_matching_tokens_to_oauth2_introspection(
    gateway: Gateway, tmp_path: Path
) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key())
    subject: Final = f"oauth-machine-{uuid.uuid4().hex}"

    def jwks(request: Request) -> Reply:
        assert request.method == "GET", request
        return Reply(body=json.dumps({"keys": [{**json.loads(public_jwk), "kid": KEY_ID}]}).encode())

    def token_info(request: Request) -> Reply:
        assert request.method == "GET", request
        assert request.headers["authorization"].startswith("Bearer "), request.headers
        return Reply(body=json.dumps({"sub": subject, "role": "proxy_admin"}).encode())

    def token(**claims: object) -> str:
        now: Final = int(time.time())
        return jwt.encode(
            {"sub": subject, "iat": now, "exp": now + 300, **claims},
            private_key,
            algorithm="RS256",
            headers={"kid": KEY_ID},
        )

    with wire_server(jwks) as keys, wire_server(token_info) as introspection, gateway.scenario() as scenario:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["general_settings"] = {
            **config["general_settings"],
            "enable_jwt_auth": True,
            "litellm_jwtauth": {
                "user_id_jwt_field": "sub",
                "routing_overrides": [
                    {
                        "iss": ["https://other.example.com", "https://idp.*"],
                        "scope": "litellm.gateway",
                        "path": "oauth2",
                    }
                ],
            },
        }
        path: Final = tmp_path / "jwt_routing_override.yaml"
        path.write_text(yaml.safe_dump(config))
        overrides: Final = {"JWT_PUBLIC_KEY_URL": keys.url, "OAUTH_TOKEN_INFO_ENDPOINT": introspection.url}
        with owned_proxy(gateway, tmp_path, overrides, config=path) as candidate:
            model: Final = scenario.model()

            def chat(bearer: str) -> int:
                return candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {"model": model, "messages": [{"role": "user", "content": "routing override"}]},
                    key=bearer,
                ).status_code

            assert chat(token(iss=ISSUER, scope="openid litellm.gateway")) == 200
            assert len(introspection.drain()) == 1, "space-delimited scope must match the override"
            assert chat(token(iss=ISSUER, scope=["litellm.gateway", "openid"])) == 200
            assert len(introspection.drain()) == 1, "list scope must match the override"

            chat(token(iss=ISSUER, scope="openid"))
            chat(token(iss=f"{ISSUER} evil.example.com", scope="litellm.gateway"))
            chat(token(iss=ISSUER))
            assert introspection.drain() == (), "non-matching claims must stay on the JWT path"
            assert keys.drain(), "non-matching tokens must be validated against the JWKS"
