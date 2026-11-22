import json
import time
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import jwt
import pytest
import yaml
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import JsonValue

from tests.integration._support import claude_code
from tests.integration._support.client import Gateway, Scenario, list_value, object_value, string_value
from tests.integration._support.process import owned_proxy
from tests.integration._support.wire import Reply, Request, Wire, wire_server

pytestmark: Final = pytest.mark.shares_worker_state

KEY_ID: Final = "entra-signing-key"
TENANT_ID: Final = "00000000-0000-4000-8000-00000000c0de"
CLIENT_ID: Final = "11111111-2222-4333-8444-555555555555"
ISSUER: Final = f"https://login.microsoftonline.com/{TENANT_ID}/v2.0"
JWT_ENVIRONMENT: Final = ("JWT_AUDIENCE", "JWT_PUBLIC_KEY_URL", "JWT_ISSUER")
ROUTES: Final = ("/v1/models", "/v1/chat/completions", "/v1/messages")
ANTHROPIC_MODEL: Final = "claude-opus-5-5"
ANTHROPIC_KEY: Final = "synthetic-anthropic-key"


@dataclass(frozen=True, slots=True)
class Signer:
    key: rsa.RSAPrivateKey

    def jwks(self) -> Reply:
        public: Final = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(self.key.public_key()))
        return Reply(body=json.dumps({"keys": [{**public, "kid": KEY_ID}]}).encode())

    def token(self, audience: str, oid: str) -> str:
        now: Final = int(time.time())
        return jwt.encode(
            {
                "iss": ISSUER,
                "aud": audience,
                "tid": TENANT_ID,
                "oid": oid,
                "sub": f"pairwise-{oid}",
                "email": f"{oid}@example.com",
                "ver": "2.0",
                "iat": now,
                "nbf": now,
                "exp": now + 300,
            },
            self.key,
            algorithm="RS256",
            headers={"kid": KEY_ID},
        )


def _signer() -> Signer:
    return Signer(rsa.generate_private_key(public_exponent=65537, key_size=2048))


@contextmanager
def _jwks_server(signer: Signer) -> Iterator[Wire]:
    def respond(request: Request) -> Reply:
        assert request.method == "GET", request
        return signer.jwks()

    with wire_server(respond) as served:
        yield served


@contextmanager
def _anthropic_server() -> Iterator[Wire]:
    def respond(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/messages", request.target
        assert request.headers["x-api-key"] == ANTHROPIC_KEY, request.headers
        return Reply(
            body=claude_code.message_reply(
                f"msg_{uuid.uuid4().hex}",
                ANTHROPIC_MODEL,
                ({"type": "text", "text": "served"},),
                {"input_tokens": 5, "output_tokens": 1},
            )
        )

    with wire_server(respond) as served:
        yield served


@dataclass(frozen=True, slots=True)
class Models:
    chat: str
    messages: str
    anthropic: Wire

    @classmethod
    def registered(cls, scenario: Scenario, anthropic: Wire) -> "Models":
        return cls(
            chat=scenario.model(),
            messages=scenario.model(
                model=f"anthropic/{ANTHROPIC_MODEL}", api_base=anthropic.url, api_key=ANTHROPIC_KEY
            ),
            anthropic=anthropic,
        )


def _config(directory: Path, jwtauth: Mapping[str, JsonValue]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    path: Final = directory / f"jwt-auth-{uuid.uuid4().hex}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                **config,
                "general_settings": {
                    **config["general_settings"],
                    "enable_jwt_auth": True,
                    "litellm_jwtauth": {
                        "user_id_jwt_field": "oid",
                        "user_email_jwt_field": "email",
                        "user_id_upsert": True,
                        **jwtauth,
                    },
                },
            }
        )
    )
    return path


def _user_rows(gateway: Gateway, oid: str) -> list[dict[str, JsonValue]]:
    return gateway.rows('SELECT user_id, user_email FROM "LiteLLM_UserTable" WHERE user_id = %s', (oid,))


def _delete_user_if_present(scenario: Scenario, oid: str) -> None:
    if _user_rows(scenario.gateway, oid):
        scenario.delete_user(oid)


def _new_subject(scenario: Scenario) -> str:
    oid: Final = str(uuid.uuid4())
    scenario.cleanups.callback(_delete_user_if_present, scenario, oid)
    return oid


def _call(proxy: Gateway, route: str, models: Models, token: str) -> httpx.Response:
    if route == "/v1/models":
        return proxy.request("GET", route, key=token)
    messages: Final[list[JsonValue]] = [{"role": "user", "content": "config audience"}]
    if route == "/v1/messages":
        return proxy.request(
            "POST", route, {"model": models.messages, "max_tokens": 32, "messages": messages}, key=token
        )
    return proxy.request("POST", route, {"model": models.chat, "messages": messages}, key=token)


def _assert_served(response: httpx.Response, route: str, models: Models) -> None:
    assert response.status_code == 200, response.text
    body: Final = object_value(response.json())
    if route == "/v1/models":
        listed: Final = {string_value(object_value(entry)["id"]) for entry in list_value(body["data"])}
        assert {models.chat, models.messages} <= listed, body
    elif route == "/v1/messages":
        assert body["content"] == [{"type": "text", "text": "served"}], body
        assert len(models.anthropic.drain()) == 1, "the Anthropic upstream did not get exactly one request"
    else:
        assert string_value(object_value(object_value(list_value(body["choices"])[0])["message"])["content"]), body


def _assert_refused(proxy: Gateway, response: httpx.Response, oid: str) -> None:
    assert response.status_code == 401, response.text
    assert _user_rows(proxy, oid) == []


@dataclass(frozen=True, slots=True)
class Rig:
    proxy: Gateway
    scenario: Scenario
    models: Models
    signer: Signer
    environment_signer: Signer
    environment_jwks: Wire


@contextmanager
def _rig(
    session_gateway: Gateway,
    directory: Path,
    *,
    config_keys: bool,
    environment_audience: str | None,
    environment_keys: str,
) -> Iterator[Rig]:
    signer: Final = _signer()
    environment_signer: Final = _signer()
    with (
        session_gateway.scenario() as scenario,
        _jwks_server(signer) as jwks,
        _jwks_server(environment_signer if environment_keys == "own" else signer) as environment_jwks,
        _anthropic_server() as anthropic,
    ):
        models: Final = Models.registered(scenario, anthropic)
        config: Final = _config(directory, {"public_key_url": jwks.url, "audience": CLIENT_ID} if config_keys else {})
        overrides: Final = (
            {}
            if environment_audience is None
            else {"JWT_PUBLIC_KEY_URL": environment_jwks.url, "JWT_AUDIENCE": environment_audience}
        )
        with owned_proxy(
            session_gateway,
            directory,
            overrides,
            config=config,
            remove_environment=tuple(name for name in JWT_ENVIRONMENT if name not in overrides),
            workers=2,
        ) as proxy:
            yield Rig(proxy, scenario, models, signer, environment_signer, environment_jwks)


@pytest.fixture(scope="module")
def config_only(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        session_gateway,
        tmp_path_factory.mktemp("jwt-config-only"),
        config_keys=True,
        environment_audience=None,
        environment_keys="own",
    ) as rig:
        yield rig


@pytest.fixture(scope="module")
def config_and_environment(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        session_gateway,
        tmp_path_factory.mktemp("jwt-config-and-environment"),
        config_keys=True,
        environment_audience="environment-application",
        environment_keys="own",
    ) as rig:
        yield rig


@pytest.fixture(scope="module")
def environment_only(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    with _rig(
        session_gateway,
        tmp_path_factory.mktemp("jwt-environment-only"),
        config_keys=False,
        environment_audience=CLIENT_ID,
        environment_keys="shared",
    ) as rig:
        yield rig


def _assert_served_and_upserted(rig: Rig, route: str, token_for: Callable[[str], str]) -> None:
    oid: Final = _new_subject(rig.scenario)

    response: Final = _call(rig.proxy, route, rig.models, token_for(oid))

    _assert_served(response, route, rig.models)
    assert _user_rows(rig.proxy, oid) == [{"user_id": oid, "user_email": f"{oid}@example.com"}]


def _assert_token_refused(rig: Rig, route: str, token_for: Callable[[str], str]) -> None:
    oid: Final = _new_subject(rig.scenario)

    response: Final = _call(rig.proxy, route, rig.models, token_for(oid))

    _assert_refused(rig.proxy, response, oid)


@pytest.mark.parametrize("route", ROUTES)
def test_config_audience_and_jwks_serve_a_token_for_that_audience(config_only: Rig, route: str) -> None:
    _assert_served_and_upserted(config_only, route, lambda oid: config_only.signer.token(CLIENT_ID, oid))


@pytest.mark.parametrize("route", ROUTES)
def test_config_audience_refuses_a_token_for_another_audience(config_only: Rig, route: str) -> None:
    _assert_token_refused(config_only, route, lambda oid: config_only.signer.token("another-application", oid))


@pytest.mark.parametrize("route", ROUTES)
def test_config_jwks_refuses_a_token_signed_by_another_key(config_only: Rig, route: str) -> None:
    _assert_token_refused(config_only, route, lambda oid: _signer().token(CLIENT_ID, oid))


@pytest.mark.parametrize("route", ROUTES)
def test_config_audience_and_jwks_win_over_the_environment(config_and_environment: Rig, route: str) -> None:
    _assert_served_and_upserted(
        config_and_environment, route, lambda oid: config_and_environment.signer.token(CLIENT_ID, oid)
    )


@pytest.mark.parametrize("route", ROUTES)
def test_environment_audience_and_jwks_are_ignored_when_config_sets_them(
    config_and_environment: Rig, route: str
) -> None:
    _assert_token_refused(
        config_and_environment,
        route,
        lambda oid: config_and_environment.environment_signer.token("environment-application", oid),
    )
    assert config_and_environment.environment_jwks.drain() == (), "the JWT_PUBLIC_KEY_URL endpoint was contacted"


@pytest.mark.parametrize("route", ROUTES)
def test_environment_audience_and_jwks_still_serve_without_config_keys(environment_only: Rig, route: str) -> None:
    _assert_served_and_upserted(environment_only, route, lambda oid: environment_only.signer.token(CLIENT_ID, oid))


@pytest.mark.parametrize("route", ROUTES)
def test_environment_audience_still_refuses_another_audience(environment_only: Rig, route: str) -> None:
    _assert_token_refused(
        environment_only, route, lambda oid: environment_only.signer.token("another-application", oid)
    )
