"""Unit tests for the internal data-plane auth endpoints."""

import pytest
from fastapi import HTTPException, Request

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.auth.internal_auth_endpoints import (
    DATA_PLANE_KEY_ENV_VAR,
    DATA_PLANE_KEY_HEADER,
    VerifyKeyRequest,
    require_data_plane_key,
    verify_key,
)


def _make_request(headers: dict) -> Request:
    """Build a minimal ASGI Request with the given headers."""
    raw_headers = [
        (k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()
    ]
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/internal/v1/auth/verify",
        "headers": raw_headers,
    }
    return Request(scope)


def test_require_data_plane_key_500_when_env_unset(monkeypatch):
    monkeypatch.delenv(DATA_PLANE_KEY_ENV_VAR, raising=False)
    request = _make_request({DATA_PLANE_KEY_HEADER: "anything"})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "data-plane auth not configured"


def test_require_data_plane_key_500_when_env_empty(monkeypatch):
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "")
    request = _make_request({DATA_PLANE_KEY_HEADER: "anything"})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 500


def test_require_data_plane_key_401_when_header_missing(monkeypatch):
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "secret-dp-key")
    request = _make_request({})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 401


def test_require_data_plane_key_401_when_header_wrong(monkeypatch):
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "secret-dp-key")
    request = _make_request({DATA_PLANE_KEY_HEADER: "wrong-key"})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 401


def test_require_data_plane_key_does_not_accept_master_key(monkeypatch):
    """The data-plane key must be a dedicated secret, not the master key."""
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "secret-dp-key")
    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-master-1234")
    request = _make_request({DATA_PLANE_KEY_HEADER: "sk-master-1234"})
    with pytest.raises(HTTPException) as exc_info:
        require_data_plane_key(request)
    assert exc_info.value.status_code == 401


def test_require_data_plane_key_passes_when_correct(monkeypatch):
    monkeypatch.setenv(DATA_PLANE_KEY_ENV_VAR, "secret-dp-key")
    request = _make_request({DATA_PLANE_KEY_HEADER: "secret-dp-key"})
    # Should not raise.
    assert require_data_plane_key(request) is None


@pytest.mark.asyncio
async def test_verify_key_returns_model_dump(monkeypatch):
    expected_auth = UserAPIKeyAuth(
        api_key="hashed-key", user_id="user-123", max_budget=100.0
    )

    captured = {}

    async def fake_user_api_key_auth(request, api_key):
        captured["api_key"] = api_key
        captured["request"] = request
        return expected_auth

    monkeypatch.setattr(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-test-key", route="/v1/realtime")
    result = await verify_key(body=body)

    # The key is forwarded WITH the Bearer prefix (user_api_key_auth strips it).
    assert captured["api_key"] == "Bearer sk-test-key"
    # Validation runs against a synthetic request carrying the gateway's route.
    assert captured["request"].url.path == "/v1/realtime"
    assert result == expected_auth.model_dump(exclude_none=True, mode="json")
    assert result["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_verify_key_401_on_proxy_exception(monkeypatch):
    async def fake_user_api_key_auth(request, api_key):
        raise ProxyException(
            message="bad key",
            type="auth_error",
            param=None,
            code="401",
        )

    monkeypatch.setattr(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-bad-key", route="/v1/realtime")
    with pytest.raises(HTTPException) as exc_info:
        await verify_key(body=body)
    assert exc_info.value.status_code == 401
    assert exc_info.value.detail == "invalid api key"


@pytest.mark.asyncio
async def test_verify_key_401_on_http_exception(monkeypatch):
    async def fake_user_api_key_auth(request, api_key):
        raise HTTPException(status_code=403, detail="forbidden internals")

    monkeypatch.setattr(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-bad-key", route="/v1/realtime")
    with pytest.raises(HTTPException) as exc_info:
        await verify_key(body=body)
    assert exc_info.value.status_code == 401
    # Internals must not leak.
    assert exc_info.value.detail == "invalid api key"
