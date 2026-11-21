"""Shared helpers for the `test_dec1_authsec_*` security contract files.

These files pin the security properties of the proxy's auth path while it moves to Rust (the Rust auth core
reached through `PythonAuthHost`, the route and plan rules already in `gateway-auth-core`, and the Rust front
door in `gateway-forward`). Every helper talks to the proxy over HTTP like a customer would, and reads Postgres,
the proxy log or a recording peer only to observe an effect. Nothing here patches or imports proxy internals.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import yaml
from integration._support.client import (
    JSON_OBJECT,
    Gateway,
    Scenario,
    eventually,
    gateway_from_environment,
    object_value,
    string_value,
)
from integration._support.database import scratch_database
from integration._support.database_relay import (
    DroppedConnectionRelay,
    dropped_connection_relay,
    plaintext_database_url,
)
from integration._support.process import OwnedProxy, owned_proxy, owned_proxy_process
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

CHAT: Final = "/v1/chat/completions"
BASE_CONFIG: Final = Path("tests/integration/proxy_config.yaml")
DEPLOYMENT_KEY: Final = "sk-dec1-authsec-deployment-key"
OWNED_MODEL: Final = "dec1-authsec-chat"
SECOND_MODEL: Final = "dec1-authsec-second"
GEMINI_MODEL: Final = "dec1-authsec-gemini"

NO_KEY_MESSAGE: Final = "Authentication Error, No api key passed in."
UNKNOWN_KEY_MESSAGE: Final = "Authentication Error, Invalid API key provided."
UNKNOWN_KEY_TYPE: Final = "token_not_found_in_db"
INVALID_VIRTUAL_KEY_PREFIX: Final = "LiteLLM Virtual Key expected. Received="
DB_UNAVAILABLE_USER: Final = "__db_unavailable_fallback__"
ADMIN_ONLY_PREFIX: Final = (
    "Authentication Error, Only proxy admin can be used to generate, delete, update info for new keys/users/teams. "
)
MASTER_ONLY_TEMPLATE: Final = "Authentication Error, Tried to access route={route}, which is only for MASTER KEY"
REJECTED_PARAM_TEMPLATE: Final = "Rejected Request: {param} is not allowed in request body."


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def marker(prefix: str = "dec1-authsec") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def chat_body(model: str, text: str | None = None, **extra: JsonValue) -> dict[str, JsonValue]:
    """A chat request no response cache can answer, so every call reaches the provider double."""
    return {
        "model": model,
        "messages": [{"role": "user", "content": text or marker()}],
        "cache": {"no-cache": True},
        **extra,
    }


def raw_post(
    gateway: Gateway,
    path: str,
    body: Mapping[str, JsonValue] | None,
    headers: Sequence[tuple[str, str | bytes]] | Mapping[str, str],
    *,
    params: Mapping[str, str] | None = None,
) -> httpx.Response:
    """POST with exactly the headers given, without the `Authorization` that `Gateway.request` always adds.

    A sequence of pairs keeps duplicates and byte values, so a test can send two `Authorization` headers or a
    header value that is not ASCII."""
    return gateway.client.post(path, json=dict(body or {}), headers=headers, params=params)


def raw_get(
    gateway: Gateway, path: str, headers: Mapping[str, str], params: Mapping[str, str] | None = None
) -> httpx.Response:
    return gateway.client.get(path, headers=dict(headers), params=params)


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def message_of(response: httpx.Response) -> str:
    return string_value(error_of(response)["message"])


def assert_served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    return JSON_OBJECT.validate_json(response.content)


def assert_auth_error(
    response: httpx.Response, status: int, message: str | None = None, *, error_type: str = "auth_error"
) -> str:
    """The OpenAI shaped auth refusal `{"error": {"message", "type", "param", "code"}}`.

    `type` is "auth_error" for most refusals; an unknown virtual key is "token_not_found_in_db" with param "key"
    (litellm/proxy/auth/resolvers/exceptions.py:33-39, litellm/proxy/auth/auth_checks.py:4526-4531).
    """
    assert response.status_code == status, f"{response.status_code} {response.text}"
    error: Final = error_of(response)
    assert error["type"] == error_type, response.text
    assert error["code"] == str(status), response.text
    text: Final = string_value(error["message"])
    if message is not None:
        assert text == message, response.text
    return text


def assert_not_echoed(response: httpx.Response, *secrets: str) -> None:
    """Neither the secret nor its SHA-256 appears in the body or any response header."""
    blob: Final = response.text + json.dumps(dict(response.headers))
    for secret in secrets:
        assert secret not in blob, f"response echoed a credential: {response.text}"
        assert digest(secret) not in blob, f"response echoed a credential hash: {response.text}"


def spend_row_text(gateway: Gateway, request_id: str) -> str:
    """The whole `LiteLLM_SpendLogs` row of one request as JSON text, once it has landed."""
    rows: Final = eventually(
        lambda: gateway.rows(
            'SELECT to_jsonb(s)::text AS row FROM "LiteLLM_SpendLogs" s WHERE request_id=%s', (request_id,)
        ),
        lambda found: len(found) == 1,
        seconds=70,
    )
    return string_value(rows[0]["row"])


def spend_row(gateway: Gateway, request_id: str) -> dict[str, JsonValue]:
    return object_value(json.loads(spend_row_text(gateway, request_id)))


def chat_completion(text: str = "authsec answer") -> bytes:
    return json.dumps(
        {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    ).encode()


GEMINI_REPLY: Final = json.dumps(
    {
        "candidates": [
            {"content": {"parts": [{"text": "green"}], "role": "model"}, "finishReason": "STOP", "index": 0}
        ],
        "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 1, "totalTokenCount": 4},
    }
).encode()


def provider_reply(request: Request) -> Reply:
    """What a healthy provider answers: Gemini generateContent, embeddings, or a chat completion."""
    path: Final = request.target.split("?", 1)[0]
    if "generateContent" in path:
        return Reply(body=GEMINI_REPLY)
    if path.endswith("/embeddings"):
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 2, "total_tokens": 2},
                }
            ).encode()
        )
    return Reply(body=chat_completion())


def posts(upstream: Wire) -> tuple[Request, ...]:
    return tuple(request for request in upstream.drain() if request.method == "POST")


def header_values_carrying(request: Request, secret: str) -> list[str]:
    """Names of provider request headers whose value contains `secret` or its SHA-256."""
    return [name for name, value in request.headers.items() if secret in value or digest(secret) in value]


def owned_config(
    directory: Path,
    upstream_url: str,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    extra_models: Sequence[Mapping[str, JsonValue]] = (),
    name: str | None = None,
) -> Path:
    """The suite config with two chat models and one Gemini model served by `upstream_url`, plus the given sections."""
    base: Final = JSON_OBJECT.validate_python(yaml.safe_load(BASE_CONFIG.read_text()))
    configuration: Final = {
        **base,
        "model_list": [
            {
                "model_name": OWNED_MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{upstream_url}/v1",
                    "api_key": DEPLOYMENT_KEY,
                },
            },
            {
                "model_name": SECOND_MODEL,
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_base": f"{upstream_url}/v1",
                    "api_key": DEPLOYMENT_KEY,
                },
            },
            {
                "model_name": GEMINI_MODEL,
                "litellm_params": {
                    "model": "gemini/gemini-2.5-flash",
                    "api_base": upstream_url,
                    "api_key": DEPLOYMENT_KEY,
                },
            },
            *extra_models,
        ],
        "general_settings": {**object_value(base.get("general_settings", {})), **(general_settings or {})},
        "litellm_settings": {**object_value(base.get("litellm_settings", {})), **(litellm_settings or {})},
        "router_settings": {**object_value(base.get("router_settings", {})), "num_retries": 0},
    }
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / (name or f"dec1-authsec-{uuid.uuid4().hex}.yaml")
    path.write_text(yaml.safe_dump(configuration))
    return path


@dataclass(frozen=True, slots=True)
class Door:
    """An owned proxy with its recording provider and the process handle, whose `log` holds stdout and stderr."""

    gateway: Gateway
    upstream: Wire
    owned: OwnedProxy

    def calls(self) -> tuple[Request, ...]:
        return posts(self.upstream)


@contextmanager
def owned_door(
    directory: Path,
    *,
    general_settings: Mapping[str, JsonValue] | None = None,
    litellm_settings: Mapping[str, JsonValue] | None = None,
    environment: Mapping[str, str] | None = None,
    extra_models: Sequence[Mapping[str, JsonValue]] = (),
) -> Iterator[Door]:
    """A proxy booted for one module or test with its own settings, in front of a recording provider."""
    with gateway_from_environment() as root, wire_server(provider_reply) as upstream:
        config: Final = owned_config(
            directory,
            upstream.url,
            general_settings=general_settings,
            litellm_settings=litellm_settings,
            extra_models=extra_models,
        )
        with owned_proxy_process(root, directory, dict(environment or {}), config=config, workers=1) as owned:
            yield Door(owned.gateway, upstream, owned)


@dataclass(frozen=True, slots=True)
class Outage:
    """An owned proxy whose database connection drops whenever a query names one specific key's hash."""

    proxy: Gateway
    relay: DroppedConnectionRelay
    scenario: Scenario
    key: str


@contextmanager
def outage_rig(directory: Path, *, allow_requests: bool, key: str | None = None) -> Iterator[Outage]:
    """`allow_requests_on_db_unavailable` set as given; the relay stays disarmed until the test arms it."""
    chosen: Final = key or f"sk-{marker()}"
    with ExitStack() as stack:
        root: Final = stack.enter_context(gateway_from_environment())
        database_url: Final = stack.enter_context(scratch_database())
        relay, relayed_url = stack.enter_context(dropped_connection_relay(database_url, digest(chosen).encode()))
        config: Final = owned_config(
            directory, root.upstream_url, general_settings={"allow_requests_on_db_unavailable": allow_requests}
        )
        proxy: Final = stack.enter_context(
            owned_proxy(
                root,
                directory / "proxy",
                {"DATABASE_URL": plaintext_database_url(relayed_url), "PRISMA_HEALTH_WATCHDOG_ENABLED": "false"},
                config=config,
                remove_environment=("DATABASE_URL_READ_REPLICA",),
            )
        )
        scenario: Final = stack.enter_context(proxy.scenario())
        stack.callback(relay.disarm)
        yield Outage(proxy, relay, scenario, chosen)


@dataclass
class TokenInfo:
    """An OAuth2 token-info double: a known token gets its JSON answer, any other token a 401."""

    answers: dict[str, JsonValue] = field(default_factory=dict)
    seen: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def grant(self, token: str, answer: JsonValue) -> None:
        with self.lock:
            self.answers[token] = answer

    def revoke(self, token: str) -> None:
        with self.lock:
            self.answers.pop(token, None)

    def calls_for(self, token: str) -> int:
        with self.lock:
            return sum(1 for presented in self.seen if presented == token)

    def respond(self, request: Request) -> Reply:
        presented: Final = request.headers.get("authorization", "").removeprefix("Bearer ")
        with self.lock:
            self.seen.append(presented)
            answer: Final = self.answers.get(presented)
        if answer is None:
            return Reply(status=401, body=b'{"error": "invalid_token"}')
        return Reply(body=json.dumps(answer).encode())


@dataclass(frozen=True, slots=True)
class OAuthDoor:
    door: Door
    info: TokenInfo


@contextmanager
def oauth_door(directory: Path, *, environment: Mapping[str, str] | None = None) -> Iterator[OAuthDoor]:
    """An owned proxy with `enable_oauth2_auth` whose token-info endpoint is a local double.

    Info routes are introspected too, so the double vouches for the master key as a proxy admin (the harness
    reads `/model/info` with it)."""
    info: Final = TokenInfo()
    with (
        wire_server(info.respond) as idp,
        owned_door(
            directory,
            general_settings={"enable_oauth2_auth": True},
            environment={"OAUTH_TOKEN_INFO_ENDPOINT": f"{idp.url}/tokeninfo", **(environment or {})},
        ) as door,
    ):
        info.grant(door.gateway.key, {"sub": "dec1-authsec-oauth-admin", "role": "proxy_admin"})
        yield OAuthDoor(door, info)


def login(gateway: Gateway, username: str, password: str) -> httpx.Response:
    """POST `/v2/login` from a fresh cookie jar, the JSON login the dashboard uses."""
    with httpx.Client(base_url=str(gateway.client.base_url), timeout=30, trust_env=False) as browser:
        return browser.post("/v2/login", json={"username": username, "password": password})
