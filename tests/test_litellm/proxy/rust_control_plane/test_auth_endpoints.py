"""Unit tests for the Rust control-plane auth endpoints."""

import pytest
from fastapi import HTTPException, Request

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.rust_control_plane.auth_endpoints import (
    DATA_PLANE_KEY_ENV_VAR,
    DATA_PLANE_KEY_HEADER,
    VerifyKeyRequest,
    _synthetic_request,
    require_data_plane_key,
    router,
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
        "path": "/v1/rust_control_plane/authentication",
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


def test_router_mounts_auth_verify_under_rust_control_plane():
    assert any(
        getattr(route, "path", None) == "/v1/rust_control_plane/authentication"
        for route in router.routes
    )


@pytest.mark.asyncio
async def test_synthetic_request_skips_budget_reservation():
    request = _synthetic_request(
        route="/v1/realtime",
        authorization_header="Bearer sk-test-key",
        model="gpt-realtime",
    )

    assert request.url.path == "/v1/realtime"
    assert request.state.skip_budget_reservation is True
    assert (await request.json()) == {"model": "gpt-realtime"}


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

    body = VerifyKeyRequest(
        api_key="sk-test-key", route="/v1/realtime", model="gpt-realtime"
    )
    result = await verify_key(body=body)

    # The key is forwarded WITH the Bearer prefix (user_api_key_auth strips it).
    assert captured["api_key"] == "Bearer sk-test-key"
    # Validation runs against a synthetic request carrying the gateway's route...
    assert captured["request"].url.path == "/v1/realtime"
    assert captured["request"].headers["authorization"] == "Bearer sk-test-key"
    # ...and the requested model in the body, so model-access checks enforce it.
    assert (await captured["request"].json())["model"] == "gpt-realtime"
    assert result == expected_auth.model_dump(exclude_none=True, mode="json")
    assert result["user_id"] == "user-123"


@pytest.mark.asyncio
async def test_verify_key_omits_model_when_absent(monkeypatch):
    captured = {}

    async def fake_user_api_key_auth(request, api_key):
        captured["request"] = request
        return UserAPIKeyAuth(api_key="hashed-key")

    monkeypatch.setattr(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-test-key", route="/v1/realtime")
    await verify_key(body=body)
    # No model requested → empty body, not {"model": null}.
    assert (await captured["request"].json()) == {}


@pytest.mark.asyncio
async def test_verify_key_does_not_double_prefix_existing_bearer(monkeypatch):
    captured = {}

    async def fake_user_api_key_auth(request, api_key):
        captured["api_key"] = api_key
        captured["request"] = request
        return UserAPIKeyAuth(api_key="hashed-key")

    monkeypatch.setattr(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="Bearer sk-test-key", route="/v1/realtime")
    await verify_key(body=body)

    assert captured["api_key"] == "Bearer sk-test-key"
    assert captured["request"].headers["authorization"] == "Bearer sk-test-key"


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


@pytest.mark.asyncio
async def test_verify_key_propagates_http_5xx(monkeypatch):
    # A 5xx (e.g. DB outage) must NOT be masked as 401 — operators need the real error.
    async def fake_user_api_key_auth(request, api_key):
        raise HTTPException(status_code=503, detail="db unavailable")

    monkeypatch.setattr(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-key", route="/v1/realtime")
    with pytest.raises(HTTPException) as exc_info:
        await verify_key(body=body)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
async def test_verify_key_propagates_proxy_5xx(monkeypatch):
    # A ProxyException carrying a 5xx code propagates too (not converted to 401).
    async def fake_user_api_key_auth(request, api_key):
        raise ProxyException(
            message="internal", type="internal_error", param=None, code="500"
        )

    monkeypatch.setattr(
        "litellm.proxy.auth.user_api_key_auth.user_api_key_auth",
        fake_user_api_key_auth,
    )

    body = VerifyKeyRequest(api_key="sk-key", route="/v1/realtime")
    with pytest.raises(ProxyException):
        await verify_key(body=body)
