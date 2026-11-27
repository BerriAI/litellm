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
_OTHER_AUTH: Final = "Authentication Error, Key is over a limit this mapper does not own."
_BUDGET: Final = "Budget has been exceeded! Key=alias Current cost: 1.5, Max budget: 1.0"
_EXPIRED: Final = "Authentication Error - Expired Key. Key Expiry time 2020-01-01 and current time 2026-01-01"
_MODEL_ACCESS: Final = (
    "The requested model 'tripo/task' is not available for this API key, or the model name is invalid. "
    "Check the models available to you and try again."
)


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
@pytest.mark.parametrize("path", ["/cn.tripo/v3/generation/text-to-model", "/tripo/v3/generation/text-to-model"])
@pytest.mark.parametrize(
    ("message", "error_type", "status_code", "code", "suggestion"),
    [
        (_BUDGET, ProxyErrorTypes.budget_exceeded, 422, 9101, "Check the configured budget"),
        (_BLOCKED_KEY, ProxyErrorTypes.auth_error, 401, 9102, "Unblock the key or use a different key"),
        (_EXPIRED, ProxyErrorTypes.expired_key, 401, 9103, "Use a key that has not expired"),
        (_MODEL_ACCESS, ProxyErrorTypes.key_model_access_denied, 403, 9104, "Grant this caller access to the model"),
        (_MODEL_ACCESS, ProxyErrorTypes.team_model_access_denied, 403, 9104, "Grant this caller access to the model"),
    ],
)
async def test_proxy_gate_failures_on_tripo_routes_keep_the_status_and_original_message(
    path: str, message: str, error_type: str, status_code: int, code: int, suggestion: str
) -> None:
    response: Final = await openai_exception_handler(_request(path), _proxy_error(message, error_type, status_code))
    body: Final = json.loads(response.body)

    assert response.status_code == status_code
    assert body["code"] == code
    assert body["status"] == "error"
    assert body["message"] == message
    assert body["suggestion"] == suggestion
    assert len(body["request_id"]) == 32
    assert "error" not in body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "message", "error_type", "code"),
    [
        ("/v1/chat/completions", _MISSING_KEY, ProxyErrorTypes.auth_error, 401),
        ("/v1/chat/completions", _BLOCKED_KEY, ProxyErrorTypes.auth_error, 401),
        ("/v1/chat/completions", _BUDGET, ProxyErrorTypes.budget_exceeded, 422),
        ("/v1/chat/completions", _EXPIRED, ProxyErrorTypes.expired_key, 401),
        ("/v1/chat/completions", _MODEL_ACCESS, ProxyErrorTypes.key_model_access_denied, 403),
        ("/cn.tripo/v3/generation/text-to-model", _OTHER_AUTH, ProxyErrorTypes.auth_error, 401),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/cn.tripo/v3/generation/text-to-model", "/tripo/v3/generation/text-to-model"])
async def test_proxy_rate_limit_on_tripo_route_uses_tripo_body(path: str) -> None:
    from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
    from litellm.proxy.proxy_server import otlp_http_exception_handler

    detail: Final = "Rate limit exceeded for api_key: alias. Limit type: requests. Current limit: 2, Remaining: 0."
    response: Final = await otlp_http_exception_handler(
        _request(path),
        ProxyRateLimitError(detail=detail, headers={"retry-after": "60"}),
    )
    body: Final = json.loads(response.body)

    assert response is not None
    assert response.status_code == 429
    assert body["code"] == 9105
    assert body["status"] == "error"
    assert body["message"] == detail
    assert body["suggestion"] == "Retry after the limit resets"
    assert response.headers["retry-after"] == "60"
    assert "detail" not in body


@pytest.mark.asyncio
async def test_rate_limit_on_other_routes_and_other_http_errors_on_tripo_keep_their_body() -> None:
    from fastapi import HTTPException

    from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
    from litellm.proxy.proxy_server import otlp_http_exception_handler

    limited: Final = await otlp_http_exception_handler(
        _request("/v1/chat/completions"),
        ProxyRateLimitError(detail="Rate limit exceeded for api_key: alias."),
    )
    unavailable: Final = await otlp_http_exception_handler(
        _request("/cn.tripo/v3/generation/text-to-model"),
        HTTPException(status_code=503, detail="No Tripo use_in_pass_through deployment configured for cn"),
    )

    assert limited is not None and unavailable is not None
    assert json.loads(limited.body) == {"detail": "Rate limit exceeded for api_key: alias."}
    assert json.loads(unavailable.body) == {
        "detail": "No Tripo use_in_pass_through deployment configured for cn"
    }
