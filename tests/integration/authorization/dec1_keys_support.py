"""Shared helpers for the `test_dec1_keys_*` virtual key contract files.

Every helper here talks to the proxy over HTTP exactly like a customer would, and reads Postgres or the
scripted upstream only to observe an effect. Nothing here patches or imports proxy internals.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from hashlib import sha256
from pathlib import Path
from typing import Final

import httpx
import yaml
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, list_value, object_value, string_value

CHAT_PATH: Final = "/v1/chat/completions"
MODEL_DENIED_TEMPLATE: Final = (
    "The requested model '{model}' is not available for this API key, or the model name is invalid. "
    "Check the models available to you and try again."
)
BLOCKED_KEY_MESSAGE: Final = "Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin."
MALFORMED_KEY_MESSAGE: Final = "Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix."
NO_KEY_MESSAGE: Final = "Authentication Error, No api key passed in."
UNKNOWN_KEY_MESSAGE: Final = "Authentication Error, Invalid API key provided."
DB_UNAVAILABLE_MESSAGE: Final = (
    "Service Unavailable, the authentication database is temporarily unreachable. Please retry shortly."
)
BASE_CONFIG: Final = Path("tests/integration/proxy_config.yaml")


def digest(key: str) -> str:
    return sha256(key.encode()).hexdigest()


def marker() -> str:
    return f"dec1-keys-{uuid.uuid4().hex}"


def chat_body(model: str, text: str | None = None, **extra: JsonValue) -> dict[str, JsonValue]:
    return {"model": model, "messages": [{"role": "user", "content": text or marker()}], **extra}


def chat(gateway: Gateway, key: str, model: str, text: str | None = None, **extra: JsonValue) -> httpx.Response:
    return gateway.request("POST", CHAT_PATH, chat_body(model, text, **extra), key=key)


def send_with_headers(
    gateway: Gateway,
    headers: Mapping[str, str],
    body: Mapping[str, JsonValue],
    *,
    path: str = CHAT_PATH,
    params: Mapping[str, str] | None = None,
) -> httpx.Response:
    """POST without the `Authorization` header `Gateway.request` always adds, so a test owns every credential."""
    return gateway.client.post(path, json=dict(body), headers=dict(headers), params=params)


def error_of(response: httpx.Response) -> dict[str, JsonValue]:
    return object_value(JSON_OBJECT.validate_json(response.content)["error"])


def assert_served(response: httpx.Response) -> dict[str, JsonValue]:
    assert response.status_code == 200, f"{response.status_code} {response.text}"
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert len(list_value(body["choices"])) == 1, response.text
    return body


def assert_denied(
    response: httpx.Response,
    *,
    status: int,
    error_type: str,
    message: str | None = None,
    param: str | None = None,
) -> dict[str, JsonValue]:
    """The OpenAI-shaped denial every non-Anthropic route answers: `{"error": {message, type, param, code}}`."""
    assert response.status_code == status, f"{response.status_code} {response.text}"
    error: Final = error_of(response)
    assert error["type"] == error_type, response.text
    assert error["code"] == str(status), response.text
    assert error["param"] == param, response.text
    if message is not None:
        assert error["message"] == message, response.text
    return error


def route_denied_message(allowed: Sequence[str], route: str) -> str:
    return (
        "Virtual key is not allowed to call this route. "
        f"Only allowed to call routes: {list(allowed)}. Tried to call route: {route}"
    )


def assert_route_denied(response: httpx.Response, allowed: Sequence[str], route: str) -> None:
    """The `allowed_routes` refusal: HTTP 403 with exactly `{"detail": <route message>}`.

    Pinned current behaviour: `RouteChecks.should_call_route` runs in `_authorize_authenticated_request`
    (litellm/proxy/auth/user_api_key_auth.py) before the `try` that converts failures into ProxyException,
    so its HTTPException reaches FastAPI's default handler and the body is the bare `detail` shape rather
    than the OpenAI `{"error": {...}}` shape every other auth refusal uses.
    """
    assert response.status_code == 403, f"{response.status_code} {response.text}"
    assert JSON_OBJECT.validate_json(response.content) == {"detail": route_denied_message(allowed, route)}, (
        response.text
    )


def assert_model_denied(response: httpx.Response, model: str, error_type: str) -> None:
    assert_denied(
        response, status=403, error_type=error_type, param="model", message=MODEL_DENIED_TEMPLATE.format(model=model)
    )


def upstream_requests(gateway: Gateway, text: str) -> list[dict[str, JsonValue]]:
    """Requests the scripted upstream received whose body mentions `text`."""
    observed: Final = httpx.get(f"{gateway.upstream_url}/__observations", timeout=15, trust_env=False)
    observed.raise_for_status()
    requests: Final = list_value(JSON_OBJECT.validate_json(observed.content)["requests"])
    return [object_value(request) for request in requests if text in str(object_value(request)["body"])]


def spend_rows(gateway: Gateway, key: str, *, at_least: int = 1) -> list[dict[str, JsonValue]]:
    return eventually(
        lambda: gateway.rows(
            'SELECT api_key, "user", team_id, organization_id, end_user, request_tags, metadata, status '
            'FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (digest(key),),
        ),
        lambda rows: len(rows) >= at_least,
        seconds=70,
    )


def key_row(gateway: Gateway, key: str) -> dict[str, JsonValue]:
    rows: Final = gateway.rows(
        'SELECT token, key_name, blocked, models, allowed_routes, metadata FROM "LiteLLM_VerificationToken" '
        "WHERE token=%s",
        (digest(key),),
    )
    assert len(rows) == 1, rows
    return rows[0]


def model_ids(gateway: Gateway, key: str) -> frozenset[str]:
    response: Final = gateway.request("GET", "/v1/models", key=key)
    assert response.status_code == 200, response.text
    return frozenset(
        string_value(object_value(entry)["id"])
        for entry in list_value(JSON_OBJECT.validate_json(response.content)["data"])
    )


def config_with_general_settings(directory: Path, settings: Mapping[str, JsonValue], name: str = "dec1.yaml") -> Path:
    config: Final = yaml.safe_load(BASE_CONFIG.read_text())
    config["general_settings"] = {**config.get("general_settings", {}), **settings}
    directory.mkdir(parents=True, exist_ok=True)
    path: Final = directory / name
    path.write_text(yaml.safe_dump(config))
    return path


def statuses(responses: Sequence[httpx.Response]) -> list[int]:
    return [response.status_code for response in responses]
