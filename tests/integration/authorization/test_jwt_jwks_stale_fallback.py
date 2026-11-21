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
from tests.integration._support.ports import allocate_port
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


_DISCOVERY_TARGET: Final = "/.well-known/openid-configuration"


def _idp_respond(
    private_key: rsa.RSAPrivateKey, failing: threading.Event, failing_target: str, failure_status: int
) -> Callable[[Request], Reply]:
    public_jwk: Final = _JWK.validate_json(RSAAlgorithm.to_jwk(private_key.public_key()))
    keys: Final = json.dumps({"keys": [{**public_jwk, "kid": _KEY_ID, "use": "sig", "alg": "RS256"}]}).encode()

    def respond(request: Request) -> Reply:
        assert request.target in ("/jwks", _DISCOVERY_TARGET)
        if failing.is_set() and request.target == failing_target:
            return Reply(status=failure_status, body=b'{"error":"internal"}')
        if request.target == _DISCOVERY_TARGET:
            return Reply(body=json.dumps({"jwks_uri": f"http://{request.headers['host']}/jwks"}).encode())
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
        wire_server(_idp_respond(private_key, failing, "/jwks", 500)) as jwks,
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


def _expired_reply_after_idp_fails(
    tmp_path: Path, key_path: str, failing_target: str, failure_status: int
) -> tuple[httpx.Response, httpx.Response]:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    failing: Final = threading.Event()
    token: Final = _admin_jwt(private_key)
    with (
        gateway_from_environment() as base,
        wire_server(_idp_respond(private_key, failing, failing_target, failure_status)) as idp,
        owned_proxy(
            base, tmp_path / "proxy", {"JWT_PUBLIC_KEY_URL": f"{idp.url}{key_path}"}, config=_proxy_config(tmp_path)
        ) as gateway,
    ):

        def models_then_fetches() -> tuple[httpx.Response, int]:
            response: Final = gateway.request("GET", "/v1/models", key=token)
            return response, idp.received.qsize()

        warm, fetched = models_then_fetches()
        failing.set()
        expired, _ = eventually(
            models_then_fetches, lambda seen: seen[1] > fetched, seconds=_PUBLIC_KEY_TTL_SECONDS * 15
        )
        return warm, expired


def test_oidc_discovery_5xx_after_expiry_serves_stale_keys(tmp_path: Path) -> None:
    warm, expired = _expired_reply_after_idp_fails(tmp_path, _DISCOVERY_TARGET, _DISCOVERY_TARGET, 503)

    assert warm.status_code == 200, warm.text
    assert expired.status_code == 200, expired.text


def test_jwks_404_after_expiry_still_denies_with_401(tmp_path: Path) -> None:
    warm, expired = _expired_reply_after_idp_fails(tmp_path, "/jwks", "/jwks", 404)

    assert warm.status_code == 200, warm.text
    assert expired.status_code == 401, expired.text
    assert "returned status 404" in expired.text


def test_jwks_5xx_with_nothing_cached_answers_503_and_backs_off(tmp_path: Path) -> None:
    private_key: Final = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    failing: Final = threading.Event()
    failing.set()
    token: Final = _admin_jwt(private_key)
    with (
        gateway_from_environment() as base,
        wire_server(_idp_respond(private_key, failing, "/jwks", 500)) as jwks,
        owned_proxy(
            base, tmp_path / "proxy", {"JWT_PUBLIC_KEY_URL": f"{jwks.url}/jwks"}, config=_proxy_config(tmp_path)
        ) as gateway,
    ):
        first: Final = gateway.request("GET", "/v1/models", key=token)
        second: Final = gateway.request("GET", "/v1/models", key=token)

        assert (first.status_code, second.status_code) == (503, 503), (first.text, second.text)
        assert first.json()["error"]["type"] == "auth_provider_unavailable"
        assert jwks.received.qsize() == 1
        reason: Final = f"Error: JWT Auth: JWKS endpoint {jwks.url}/jwks returned status 500"
        for reply in (first, second):
            message = reply.json()["error"]["message"]
            assert message.endswith(reason), message
            assert "TransportError" not in message
            assert "internal" not in message


def test_refused_jwks_connection_with_nothing_cached_answers_503_naming_the_transport_error(tmp_path: Path) -> None:
    token: Final = _admin_jwt(rsa.generate_private_key(public_exponent=65537, key_size=2048))
    url: Final = f"http://127.0.0.1:{allocate_port()}/jwks"
    with (
        gateway_from_environment() as base,
        owned_proxy(base, tmp_path / "proxy", {"JWT_PUBLIC_KEY_URL": url}, config=_proxy_config(tmp_path)) as gateway,
    ):
        first: Final = gateway.request("GET", "/v1/models", key=token)
        second: Final = gateway.request("GET", "/v1/models", key=token)

        assert (first.status_code, second.status_code) == (503, 503), (first.text, second.text)
        assert first.json()["error"]["type"] == "auth_provider_unavailable"
        reason: Final = f"Error: TransportError fetching {url} after 3 attempts"
        assert first.json()["error"]["message"].endswith(reason), first.text
        assert second.json()["error"]["message"] == first.json()["error"]["message"]
