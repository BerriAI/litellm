import json
import time
import uuid
from pathlib import Path
from typing import Final

import jwt
import yaml
from cryptography.hazmat.primitives.asymmetric import ed25519, rsa

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.process import owned_proxy_process
from tests.integration._support.wire import Reply, Request, wire_server

KEY_ID: Final = "integration-jwt-allowlist-key"


def _jwks_reply(public_jwk: str) -> Reply:
    return Reply(body=json.dumps({"keys": [{**json.loads(public_jwk), "kid": KEY_ID}]}).encode())


def _jwt_auth_config(tmp_path: Path) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["general_settings"] = {
        **config["general_settings"],
        "enable_jwt_auth": True,
        "litellm_jwtauth": {"user_id_jwt_field": "sub", "user_id_upsert": True},
    }
    path: Final = tmp_path / "jwt_allowlist.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _token(key: object, algorithm: str, subject: str) -> str:
    return jwt.encode(
        {"sub": subject, "iat": int(time.time()), "exp": int(time.time()) + 300},
        key,  # pyright: ignore[reportArgumentType]  # jwt.encode takes Any key material
        algorithm=algorithm,
        headers={"kid": KEY_ID},
    )


def test_eddsa_signed_token_is_accepted_and_logged_as_deprecated_outside_fips_mode(
    gateway: Gateway, tmp_path: Path
) -> None:
    private_key: Final = ed25519.Ed25519PrivateKey.generate()
    public_jwk: Final = jwt.algorithms.OKPAlgorithm.to_jwk(private_key.public_key())

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply(public_jwk)

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _token(private_key, "EdDSA", subject)
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path)
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, subject)
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist control"}]},
                key=token,
            )
            assert response.status_code == 200, response.text
            deprecation: Final = eventually(
                lambda: owned.log.read_text(),
                lambda text: "EdDSA" in text and "deprecated" in text and "LITELLM_FIPS_MODE" in text,
                seconds=30,
            )
            assert "EdDSA" in deprecation and "deprecated" in deprecation and "LITELLM_FIPS_MODE" in deprecation


def test_rs256_signed_token_is_accepted_without_deprecation_log(gateway: Gateway, tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public_jwk: Final = jwt.algorithms.RSAAlgorithm.to_jwk(private_key.public_key())

    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return _jwks_reply(public_jwk)

    with wire_server(respond) as jwks, gateway.scenario() as scenario:
        subject: Final = f"integration-jwt-{uuid.uuid4().hex}"
        token: Final = _token(private_key, "RS256", subject)
        with owned_proxy_process(
            gateway, tmp_path, {"JWT_PUBLIC_KEY_URL": jwks.url}, config=_jwt_auth_config(tmp_path)
        ) as owned:
            model: Final = scenario.model()
            scenario.cleanups.callback(scenario.delete_user, subject)
            response: Final = owned.gateway.request(
                "POST",
                "/v1/chat/completions",
                {"model": model, "messages": [{"role": "user", "content": "jwt alg allowlist control"}]},
                key=token,
            )
            assert response.status_code == 200, response.text
            assert "EdDSA" not in owned.log.read_text(), owned.log.read_text()
