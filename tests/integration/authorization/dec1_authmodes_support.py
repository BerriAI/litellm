"""Shared helpers for the `test_dec1_authmodes_*` contracts: authentication modes other than virtual keys.

The proxy authenticates a caller in one of several ways besides a virtual key: a signed JWT checked against
an identity provider's JWKS (`general_settings.enable_jwt_auth`), OAuth2 token introspection
(`enable_oauth2_auth`), identity headers from a trusted reverse proxy (`enable_oauth2_proxy_auth`), or an
operator supplied Python function (`custom_auth`). These helpers start an owned proxy for each mode and a
local identity provider double, and read Postgres only to observe an effect. Nothing here imports or
patches proxy internals.

Every owned proxy here boots on its own private database and Redis (the harness default), so a test creates
the models, teams, users and keys it needs through that proxy's own `scenario()` and reads rows with that
proxy's `rows()`. Nothing is created on the shared gateway.

JWT auth, OAuth2 auth and `public_routes`/`admin_only_routes` are enterprise features. Like the existing
`test_jwt_*` files, these contracts assume the run supplies `LITELLM_LICENSE`, which the owned proxy inherits;
the unlicensed refusals are pinned by booting a proxy with `LITELLM_LICENSE` removed.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Literal

import httpx
import jwt
import yaml
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from pydantic import JsonValue

from tests.integration._support.client import (
    JSON_OBJECT,
    Gateway,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.authorization.dec1_keys_support import error_of

CHAT_PATH: Final = "/v1/chat/completions"
BASE_CONFIG: Final = Path("tests/integration/proxy_config.yaml")
MOCK_REPLY: Final = "Hello! This is a mock response from the fake OpenAI endpoint."
NOT_PREMIUM: Final = (
    "You must be a LiteLLM Enterprise user to use this feature. If you have a license please set "
    "`LITELLM_LICENSE` in your env. Get a 7 day trial key here: https://www.litellm.ai/enterprise#trial. "
    "\nPricing: https://www.litellm.ai/#pricing"
)
JWKS_UNAVAILABLE_PREFIX: Final = (
    "Service Unavailable, the identity provider's JWKS endpoint is temporarily unreachable, so the JWT "
    "signature could not be verified. Please retry shortly. Error: "
)
MODEL_DENIED_TEMPLATE: Final = (
    "The requested model '{model}' is not available for this API key, or the model name is invalid. "
    "Check the models available to you and try again."
)
ADMIN_ONLY_PREFIX: Final = (
    "Authentication Error, Only proxy admin can be used to generate, delete, update info for new "
    "keys/users/teams. Route={route}. Your role={role}. Your user_id="
)
JwksMode = Literal["serve", "error_500", "drop"]


def unique(prefix: str) -> str:
    return f"dec1-authmodes-{prefix}-{uuid.uuid4().hex[:12]}"


def marker() -> str:
    return unique("prompt")


def validation_failure(reason: str) -> str:
    """The 401 message every PyJWT decode failure other than expiry gets (litellm/proxy/auth/handle_jwt.py:1412)."""
    return f"Authentication Error, Validation fails: {reason}"


def no_matching_key(urls: tuple[str, ...], kid: str | None) -> str:
    """The 401 for a `kid` no configured JWKS publishes (handle_jwt.py:1003): the URL list is a Python repr."""
    return f"Authentication Error, No matching public key found. keys={list(urls)!r}, kid={kid}"


@dataclass(frozen=True, slots=True)
class SigningKey:
    kid: str
    private: rsa.RSAPrivateKey | ec.EllipticCurvePrivateKey
    algorithm: str

    def public_jwk(self) -> dict[str, JsonValue]:
        if isinstance(self.private, rsa.RSAPrivateKey):
            exported = jwt.algorithms.RSAAlgorithm.to_jwk(self.private.public_key())
        else:
            exported = jwt.algorithms.ECAlgorithm.to_jwk(self.private.public_key())
        return {**JSON_OBJECT.validate_json(exported), "kid": self.kid, "use": "sig", "alg": self.algorithm}


def rsa_key(kid: str | None = None) -> SigningKey:
    return SigningKey(kid or unique("rsa"), rsa.generate_private_key(public_exponent=65537, key_size=2048), "RS256")


def ec_key(kid: str | None = None) -> SigningKey:
    return SigningKey(kid or unique("ec"), ec.generate_private_key(ec.SECP256R1()), "ES256")


@dataclass(slots=True)
class IdentityProvider:
    """A JWKS and OIDC discovery double whose published keys and failure mode a test can change mid-run.

    `GET /jwks` answers the published keys (`mode="serve"`), a 500 (`"error_500"`), or drops the connection
    without a response (`"drop"`, which httpx reports as a transport error). `GET
    /.well-known/openid-configuration` answers a discovery document whose `jwks_uri` is `/jwks`.
    """

    published: list[SigningKey]
    mode: JwksMode = "serve"
    jwks_gets: int = 0
    discovery_gets: int = 0
    wire: Wire | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        assert self.wire is not None
        return self.wire.url

    @property
    def jwks_url(self) -> str:
        return f"{self.url}/jwks"

    @property
    def discovery_url(self) -> str:
        return f"{self.url}/.well-known/openid-configuration"

    @property
    def key(self) -> SigningKey:
        return self.published[0]

    def publish(self, *keys: SigningKey) -> None:
        with self.lock:
            self.published = list(keys)

    def set_mode(self, mode: JwksMode) -> None:
        with self.lock:
            self.mode = mode

    def fetches(self) -> int:
        with self.lock:
            return self.jwks_gets

    def respond(self, request: Request) -> Reply:
        path: Final = request.target.split("?", 1)[0]
        with self.lock:
            if path == "/.well-known/openid-configuration":
                self.discovery_gets += 1
                return Reply(body=json.dumps({"issuer": self.url, "jwks_uri": self.jwks_url}).encode())
            if path != "/jwks":
                return Reply(status=404, body=b'{"error": "not found"}')
            self.jwks_gets += 1
            if self.mode == "drop":
                return Reply(drop_connection=True)
            if self.mode == "error_500":
                return Reply(status=500, body=b'{"error": "dec1 identity provider down"}')
            return Reply(body=json.dumps({"keys": [key.public_jwk() for key in self.published]}).encode())

    def token(
        self,
        key: SigningKey | None = None,
        *,
        omit: tuple[str, ...] = (),
        headers: Mapping[str, JsonValue] | None = None,
        algorithm: str | None = None,
        **claims: JsonValue,
    ) -> str:
        """A token signed by `key` (default: the first published key) with `iat` now and `exp` in five minutes.

        `claims` override the defaults, `omit` drops claims, `headers` override the JOSE header (`kid` included).
        """
        signer: Final = key or self.key
        now: Final = int(time.time())
        payload: Final = {
            name: value for name, value in {"iat": now, "exp": now + 300, **claims}.items() if name not in omit
        }
        return jwt.encode(
            payload,
            signer.private,
            algorithm=algorithm or signer.algorithm,
            headers={"kid": signer.kid, **(headers or {})},
        )


@contextmanager
def identity_provider(*keys: SigningKey) -> Iterator[IdentityProvider]:
    provider: Final = IdentityProvider(list(keys) or [rsa_key()])
    with wire_server(provider.respond) as wire:
        provider.wire = wire
        yield provider


def config_file(
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    name: str = "dec1-authmodes.yaml",
) -> Path:
    """A copy of the suite's proxy config with the given sections merged in, written into `directory`."""
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(BASE_CONFIG.read_text()))
    config: Final = {
        **base,
        "general_settings": {**object_value(base["general_settings"]), **(general_settings or {})},
        "litellm_settings": {**object_value(base["litellm_settings"]), **(litellm_settings or {})},
        "router_settings": {**object_value(base["router_settings"]), "num_retries": 0},
    }
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / name
    path.write_text(yaml.safe_dump(config))
    return path


def jwt_settings(**jwtauth: JsonValue) -> dict[str, JsonValue]:
    return {"enable_jwt_auth": True, "litellm_jwtauth": dict(jwtauth)}


@contextmanager
def authmodes_proxy(
    directory: Path,
    *,
    environment: Mapping[str, str] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    remove_environment: tuple[str, ...] = (),
) -> Iterator[Gateway]:
    """An owned proxy on a private database and Redis, started with the given settings."""
    config: Final = config_file(directory, general_settings=general_settings, litellm_settings=litellm_settings)
    with (
        gateway_from_environment() as shared,
        owned_proxy(
            shared,
            directory / "proxy",
            dict(environment or {}),
            config=config,
            remove_environment=remove_environment,
        ) as owned,
    ):
        yield owned


@dataclass(frozen=True, slots=True)
class JwtRig:
    idp: IdentityProvider
    proxy: Gateway


@contextmanager
def jwt_rig(
    directory: Path,
    *,
    keys: tuple[SigningKey, ...] = (),
    environment: Mapping[str, str] | None = None,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    discovery: bool = False,
    remove_environment: tuple[str, ...] = (),
    **jwtauth: JsonValue,
) -> Iterator[JwtRig]:
    """An identity provider plus a proxy with JWT auth on, `JWT_PUBLIC_KEY_URL` naming the provider's JWKS.

    `discovery=True` points `JWT_PUBLIC_KEY_URL` at the provider's OIDC discovery document instead.
    """
    with identity_provider(*keys) as idp:
        url: Final = idp.discovery_url if discovery else idp.jwks_url
        with authmodes_proxy(
            directory,
            environment={"JWT_PUBLIC_KEY_URL": url, **(environment or {})},
            general_settings={**jwt_settings(**jwtauth), **(general_settings or {})},
            litellm_settings=litellm_settings,
            remove_environment=remove_environment,
        ) as proxy:
            yield JwtRig(idp, proxy)


def chat_as(
    gateway: Gateway,
    credential: str,
    model: str,
    *,
    text: str | None = None,
    headers: Mapping[str, str] | None = None,
    **extra: JsonValue,
) -> httpx.Response:
    """A chat completion sent with `Authorization: Bearer <credential>` and a fresh prompt (the suite caches)."""
    return gateway.request(
        "POST",
        CHAT_PATH,
        {"model": model, "messages": [{"role": "user", "content": text or marker()}], **extra},
        key=credential,
        headers=headers,
    )


def without_credentials(
    gateway: Gateway, method: str, path: str, body: Mapping[str, JsonValue] | None = None, **headers: str
) -> httpx.Response:
    """A request carrying no `Authorization` header at all (`Gateway.request` always adds one)."""
    return gateway.client.request(method, path, json=dict(body) if body is not None else None, headers=headers)


def served_body(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    body: Final = JSON_OBJECT.validate_json(response.content)
    choices: Final = body["choices"]
    assert isinstance(choices, list) and len(choices) == 1, response.text
    assert object_value(object_value(choices[0])["message"])["content"] == MOCK_REPLY, response.text
    return body


def message_of(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def spend_row(gateway: Gateway, response: httpx.Response) -> dict[str, JsonValue]:
    """The `LiteLLM_SpendLogs` row of a served request, found by its response id. Only text columns are read."""
    request_id: Final = string_value(JSON_OBJECT.validate_json(response.content)["id"])
    rows: Final = eventually(
        lambda: gateway.rows(
            'SELECT api_key, "user", team_id, organization_id, end_user, model_group '
            'FROM "LiteLLM_SpendLogs" WHERE request_id = %s',
            (request_id,),
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return rows[0]


def user_row(gateway: Gateway, user_id: str) -> dict[str, JsonValue] | None:
    rows: Final = gateway.rows(
        'SELECT user_id, user_email, user_role, teams FROM "LiteLLM_UserTable" WHERE user_id = %s', (user_id,)
    )
    return rows[0] if rows else None


def team_member_ids(gateway: Gateway, team_id: str) -> frozenset[str]:
    rows: Final = gateway.rows('SELECT members_with_roles FROM "LiteLLM_TeamTable" WHERE team_id = %s', (team_id,))
    assert len(rows) == 1, rows
    members: Final = rows[0]["members_with_roles"]
    assert isinstance(members, list), rows
    return frozenset(str(object_value(member).get("user_id")) for member in members)
