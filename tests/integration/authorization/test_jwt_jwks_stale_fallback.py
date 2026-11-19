import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import eventually, gateway_from_environment
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, wire_server

_KEY_ID: Final = "stale-fallback-signing-key"
_PUBLIC_KEY_TTL_SECONDS: Final = 2
_JWK: Final = TypeAdapter(dict[str, JsonValue])


def _proxy_config(directory: Path) -> Path:
    config: Final = directory / "jwt_jwks_stale_fallback_config.yaml"
    config.write_text(
        json.dumps(
            {
                "model_list": (),
                "general_settings": {
                    "master_key": "os.environ/LITELLM_MASTER_KEY",
                    "database_url": "os.environ/DATABASE_URL",
                    "enable_jwt_auth": True,
                    "litellm_jwtauth": {"public_key_ttl": _PUBLIC_KEY_TTL_SECONDS},
                },
            }
        )
    )
    return config


def _jwks_respond(private_key: rsa.RSAPrivateKey, failing: threading.Event) -> Callable[[Request], Reply]:
    public_jwk: Final = _JWK.validate_json(RSAAlgorithm.to_jwk(private_key.public_key()))
    keys: Final = json.dumps({"keys": [{**public_jwk, "kid": _KEY_ID, "use": "sig", "alg": "RS256"}]}).encode()

    def respond(request: Request) -> Reply:
        assert request.target == "/jwks"
        if failing.is_set():
            return Reply(status=500, body=b'{"error":"internal"}')
        return Reply(body=keys)

    return respond


def _admin_jwt(private_key: rsa.RSAPrivateKey) -> str:
    now: Final = int(time.time())
    claims: Final = {"sub": "stale-fallback-admin", "scope": "litellm_proxy_admin", "iat": now, "exp": now + 600}
    return jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": _KEY_ID})


def test_jwks_5xx_after_expiry_serves_stale_keys_and_backs_off(tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    failing: Final = threading.Event()
    token: Final = _admin_jwt(private_key)
    with (
        gateway_from_environment() as base,
        wire_server(_jwks_respond(private_key, failing)) as jwks,
        owned_proxy(
            base, tmp_path / "proxy", {"JWT_PUBLIC_KEY_URL": f"{jwks.url}/jwks"}, config=_proxy_config(tmp_path)
        ) as gateway,
    ):

        def models_then_fetches() -> tuple[httpx.Response, int]:
            response: Final = gateway.request("GET", "/v1/models", key=token)
            return response, jwks.received.qsize()

        warm, fetched = models_then_fetches()
        assert warm.status_code == 200, warm.text
        assert fetched >= 1
        failing.set()

        expired, refetched = eventually(
            models_then_fetches, lambda seen: seen[1] > fetched, seconds=_PUBLIC_KEY_TTL_SECONDS * 15
        )
        assert expired.status_code == 200, expired.text
        assert refetched == fetched + 1

        again, fetched_again = models_then_fetches()
        assert again.status_code == 200, again.text
        assert fetched_again == refetched
