"""
Tests for LiteLLM proxy realtime WebRTC HTTP endpoints:
- POST /v1/realtime/client_secrets
- POST /v1/realtime/calls
"""

import json
import os
import sys
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.proxy.realtime_endpoints.endpoints import (
    _decode_realtime_token_payload,
    _encode_realtime_token_payload,
)

# --- Unit tests: token encode/decode helpers ---


def test_encode_realtime_token_payload():
    payload = _encode_realtime_token_payload(
        ephemeral_key="epk_abc123",
        model_id="gpt-4o-realtime-preview",
        user_id="user-1",
        team_id="team-1",
        expires_at=1234567890,
    )
    decoded = json.loads(payload)
    assert decoded["v"] == "realtime_v1"
    assert decoded["ephemeral_key"] == "epk_abc123"
    assert decoded["model_id"] == "gpt-4o-realtime-preview"
    assert decoded["user_id"] == "user-1"
    assert decoded["team_id"] == "team-1"
    assert decoded["expires_at"] == 1234567890


def test_encode_realtime_token_payload_none_optional_fields():
    payload = _encode_realtime_token_payload(
        ephemeral_key="epk_xyz",
        model_id="gpt-4o-realtime",
        user_id=None,
        team_id=None,
        expires_at=None,
    )
    decoded = json.loads(payload)
    assert decoded["user_id"] == ""
    assert decoded["team_id"] == ""
    assert decoded["expires_at"] is None


def test_decode_realtime_token_payload_valid():
    future_expires_at = int(time.time()) + 3600
    payload = _encode_realtime_token_payload(
        ephemeral_key="epk_abc",
        model_id="gpt-4o",
        user_id=None,
        team_id=None,
        expires_at=future_expires_at,
    )
    decrypted = json.loads(payload)  # simulate decrypted value
    result = _decode_realtime_token_payload(json.dumps(decrypted))
    assert result is not None
    assert result["ephemeral_key"] == "epk_abc"
    assert result["model_id"] == "gpt-4o"
    assert result["expires_at"] == future_expires_at


def test_decode_realtime_token_payload_invalid_version():
    payload = json.dumps(
        {
            "v": "realtime_v2",
            "ephemeral_key": "epk",
            "model_id": "gpt-4o",
        }
    )
    assert _decode_realtime_token_payload(payload) is None


def test_decode_realtime_token_payload_invalid_json():
    assert _decode_realtime_token_payload("not-json") is None


def test_decode_realtime_token_payload_missing_ephemeral_key():
    payload = json.dumps({"v": "realtime_v1", "model_id": "gpt-4o"})
    assert _decode_realtime_token_payload(payload) is None


def test_decode_realtime_token_payload_ephemeral_key_not_string():
    payload = json.dumps(
        {
            "v": "realtime_v1",
            "ephemeral_key": 123,
            "model_id": "gpt-4o",
        }
    )
    assert _decode_realtime_token_payload(payload) is None


# --- Integration tests: proxy endpoints (mocked upstream) ---


@pytest.fixture
def proxy_app(monkeypatch):
    from litellm.proxy import proxy_server

    monkeypatch.setattr(proxy_server, "master_key", "sk-test-master-key")
    return proxy_server.app


@pytest.fixture
def mock_route_request_client_secrets():
    """Mock route_request to return a fake upstream client_secrets response."""
    future_expires_at = int(time.time()) + 3600
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 200
    mock_resp.text = (
        f'{{"value":"upstream_ephemeral_key","expires_at":{future_expires_at}}}'
    )
    mock_resp.content = f'{{"value":"upstream_ephemeral_key","expires_at":{future_expires_at}}}'.encode()
    mock_resp.headers = {}
    mock_resp.json.return_value = {
        "value": "upstream_ephemeral_key",
        "expires_at": future_expires_at,
    }

    async def _mock_route(*args, **kwargs):
        async def _inner():
            return mock_resp

        return _inner()

    return _mock_route


@pytest.fixture
def mock_route_request_realtime_calls():
    """Mock route_request to return a fake SDP answer."""
    mock_resp = MagicMock(spec=httpx.Response)
    mock_resp.status_code = 201
    mock_resp.content = b"v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\n"
    mock_resp.headers = {"content-type": "application/sdp"}

    async def _mock_route(*args, **kwargs):
        async def _inner():
            return mock_resp

        return _inner()

    return _mock_route


@pytest.fixture
def mock_add_litellm_data():
    async def _mock(data, **kwargs):
        return data

    return _mock


@pytest.fixture
def mock_pre_call_hook():
    async def _mock(user_api_key_dict, data, call_type):
        return data

    return _mock


def test_client_secrets_requires_auth(proxy_app):
    """POST /v1/realtime/client_secrets returns 401 without Authorization."""
    from fastapi import HTTPException

    def _raise_401():
        raise HTTPException(status_code=401, detail="Unauthorized")

    proxy_app.dependency_overrides[user_api_key_auth] = _raise_401
    try:
        client = TestClient(proxy_app, raise_server_exceptions=False)
        response = client.post(
            "/v1/realtime/client_secrets",
            json={"model": "gpt-4o-realtime-preview"},
        )
        assert response.status_code == 401
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


@pytest.mark.asyncio
async def test_client_secrets_success_with_mock(
    proxy_app,
    mock_route_request_client_secrets,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    """POST /v1/realtime/client_secrets returns 200 with valid auth and mocked upstream."""
    proxy_app.dependency_overrides[user_api_key_auth] = lambda: UserAPIKeyAuth(
        user_id="test-user", team_id="test-team"
    )
    try:
        client = TestClient(proxy_app)
        with (
            patch(
                "litellm.proxy.proxy_server.route_request",
                side_effect=mock_route_request_client_secrets,
            ),
            patch(
                "litellm.proxy.proxy_server.add_litellm_data_to_request",
                side_effect=mock_add_litellm_data,
            ),
            patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
        ):
            mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
            mock_logging.post_call_failure_hook = AsyncMock()

            response = client.post(
                "/v1/realtime/client_secrets",
                headers={"Authorization": "Bearer sk-test-master-key"},
                json={"model": "gpt-4o-realtime-preview"},
            )

        assert response.status_code == 200
        data = response.json()
        assert "value" in data
        assert data["expires_at"] is not None
        assert data["expires_at"] > int(time.time())  # Should be in the future
        # Proxy encrypts the upstream value, so returned value should differ
        assert data["value"] != "upstream_ephemeral_key"
    finally:
        proxy_app.dependency_overrides.pop(user_api_key_auth, None)


def test_realtime_calls_requires_auth(proxy_app):
    """POST /v1/realtime/calls returns 401 without Authorization.

    Note: /realtime/calls does NOT use the user_api_key_auth dependency —
    it checks the Bearer token manually (an encrypted ephemeral key from
    /realtime/client_secrets).  So no dependency override is needed here.
    """
    client = TestClient(proxy_app)
    response = client.post(
        "/v1/realtime/calls",
        content=b"v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\n",
    )
    assert response.status_code == 401


def test_realtime_calls_invalid_token_returns_401(proxy_app):
    """POST /v1/realtime/calls returns 401 with invalid Bearer token."""
    client = TestClient(proxy_app)
    response = client.post(
        "/v1/realtime/calls",
        headers={"Authorization": "Bearer invalid-token-not-encrypted"},
        content=b"v=0\r\n",
    )
    assert response.status_code == 401
    assert "Invalid or expired token" in response.json().get("error", "")


@pytest.mark.asyncio
async def test_realtime_calls_success_with_valid_encrypted_token(
    proxy_app,
    mock_route_request_realtime_calls,
    mock_add_litellm_data,
    mock_pre_call_hook,
):
    """POST /v1/realtime/calls returns 201 with valid encrypted token from client_secrets."""
    # Build a valid encrypted token (same format as client_secrets returns)
    future_expires_at = int(time.time()) + 3600
    token_payload = _encode_realtime_token_payload(
        ephemeral_key="fake_upstream_epk",
        model_id="gpt-4o-realtime-preview",
        user_id=None,
        team_id=None,
        expires_at=future_expires_at,
    )
    encrypted_token = encrypt_value_helper(token_payload)

    client = TestClient(proxy_app)
    with (
        patch(
            "litellm.proxy.proxy_server.route_request",
            side_effect=mock_route_request_realtime_calls,
        ),
        patch(
            "litellm.proxy.proxy_server.add_litellm_data_to_request",
            side_effect=mock_add_litellm_data,
        ),
        patch("litellm.proxy.proxy_server.proxy_logging_obj") as mock_logging,
    ):
        mock_logging.pre_call_hook = AsyncMock(side_effect=mock_pre_call_hook)
        mock_logging.post_call_failure_hook = AsyncMock()

        response = client.post(
            "/v1/realtime/calls",
            headers={"Authorization": f"Bearer {encrypted_token}"},
            content=b"v=0\r\no=- 0 0 IN IP4 0.0.0.0\r\ns=-\r\n",
        )

    assert response.status_code == 201
    assert response.content.startswith(b"v=0")
    assert b"application/sdp" in response.headers.get("content-type", "").encode()
