import json
from typing import Final

import pytest
from starlette.requests import Request

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.proxy_server import openai_exception_handler

_MISSING_KEY: Final = "Authentication Error, No api key passed in."
_MALFORMED_KEY: Final = "Authentication Error, Malformed API Key passed in. Ensure Key has `Bearer ` prefix."
_UNKNOWN_TOKEN: Final = "Authentication Error, Invalid proxy server token passed."
_BLOCKED_KEY: Final = "Authentication Error, Key is blocked. Update via `/key/unblock` if you're an admin."


def _request(path: str, root_path: str = "") -> Request:
    scope: dict[str, object] = {"type": "http", "method": "POST", "path": path, "headers": []}
    if root_path:
        scope["root_path"] = root_path
    return Request(scope)


def _proxy_error(message: str, error_type: str, code: int) -> ProxyException:
    return ProxyException(message=message, type=error_type, param=None, code=code)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "root_path"),
    [
        ("/cn.tripo/v3/account/balance", ""),
        ("/tripo/v3/account/balance", ""),
        ("/litellm/cn.tripo/v3/account/balance", "/litellm"),
    ],
)
async def test_missing_key_on_tripo_route_uses_tripo_auth_required_body(path: str, root_path: str) -> None:
    response: Final = await openai_exception_handler(
        _request(path, root_path),
        _proxy_error(_MISSING_KEY, ProxyErrorTypes.auth_error, 401),
    )
    body: Final = json.loads(response.body)

    assert response.status_code == 401
    assert body["code"] == 2
    assert body["status"] == "error"
    assert body["message"] == "Authentication required"
    assert body["suggestion"] == "Check the Bearer API key"
    assert len(body["request_id"]) == 32


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "error_type"),
    [
        ("Authentication Error, Invalid API key provided.", ProxyErrorTypes.token_not_found_in_db),
        (_UNKNOWN_TOKEN, ProxyErrorTypes.auth_error),
    ],
)
async def test_unknown_key_on_tripo_route_uses_tripo_auth_failed_body(message: str, error_type: str) -> None:
    response: Final = await openai_exception_handler(
        _request("/cn.tripo/v3/files/presign"),
        _proxy_error(message, error_type, 401),
    )
    body: Final = json.loads(response.body)

    assert response.status_code == 401
    assert body["code"] == 1002
    assert body["message"] == "Authentication failed"
    assert body["suggestion"] == "Check if your credentials is valid, and ensure you set it correctly"


@pytest.mark.asyncio
async def test_malformed_bearer_on_tripo_route_uses_auth_required_body() -> None:
    response: Final = await openai_exception_handler(
        _request("/tripo/v3/generation/text-to-model"),
        _proxy_error(_MALFORMED_KEY, ProxyErrorTypes.auth_error, 401),
    )

    assert json.loads(response.body)["code"] == 2


@pytest.mark.asyncio
async def test_malformed_json_on_tripo_route_uses_tripo_body() -> None:
    response: Final = await openai_exception_handler(
        _request("/cn.tripo/v3/generation/text-to-model"),
        _proxy_error("Invalid JSON payload: unexpected end of data", "invalid_request_error", 400),
    )
    body: Final = json.loads(response.body)

    assert response.status_code == 400
    assert body["code"] == 1003
    assert body["status"] == "error"
    assert body["message"] == "The request body is malformed"
    assert body["suggestion"] == "Refer to the API documentation for request schema"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "message", "error_type", "code"),
    [
        ("/v1/chat/completions", _MISSING_KEY, ProxyErrorTypes.auth_error, 401),
        ("/cn.tripo/v3/generation/text-to-model", _BLOCKED_KEY, ProxyErrorTypes.auth_error, 401),
        ("/cn.tripo/v3/generation/text-to-model", "Budget exceeded", ProxyErrorTypes.budget_exceeded, 429),
        ("/tripo/v3/generation/text-to-model", "Key expired", ProxyErrorTypes.expired_key, 401),
        ("/v1/chat/completions", "Invalid JSON payload: bad", "invalid_request_error", 400),
    ],
)
async def test_unrelated_proxy_errors_keep_the_proxy_envelope(
    path: str, message: str, error_type: str, code: int
) -> None:
    response: Final = await openai_exception_handler(_request(path), _proxy_error(message, error_type, code))
    body: Final = json.loads(response.body)

    assert response.status_code == code
    assert body["error"]["message"] == message
    assert body["error"]["type"] == error_type
    assert "status" not in body


@pytest.mark.asyncio
async def test_messages_route_keeps_the_anthropic_envelope() -> None:
    response: Final = await openai_exception_handler(
        _request("/v1/messages"),
        _proxy_error(_MISSING_KEY, ProxyErrorTypes.auth_error, 401),
    )
    body: Final = json.loads(response.body)

    assert response.status_code == 401
    assert body["type"] == "error"
    assert body["error"]["message"] == _MISSING_KEY
